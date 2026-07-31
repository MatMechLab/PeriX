//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Author  : Yang Bai
//+++ Date    : 2026.04.15
//+++ Function: define the pdproblem to be solved in PeriX
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDProblem/PDProblem.h"

#include "Utils/MessagePrinter.h"

PDProblem::PDProblem()=default;

void PDProblem::init(int args,char *argv[]) {
    MessagePrinter::printNormalTxt("Start to read the input file ...");
    m_InputSystem.init(args,argv);

    m_Timer.startTimer();
    m_InputSystem.readInputFile(m_Mesh,m_PDMesh,m_MeshModify,m_Operators,m_ElmtSystem,m_BCSystem,
                                m_ICSystem,m_JobSystem,m_TimeStepping,m_NonlinearSolver,
                                m_LinearSolver,m_OutputSystem);
    m_Timer.endTimer();
    m_Timer.printElapseTime("Input file reading is done",true);

    if (m_InputSystem.isReadOnly()) return;

    // The assembly back-end is a run-wide choice owned by JobSystem
    // ("JobSystem.assemble"); hand it to the Newton solver, which is the
    // object that actually dispatches the residual/Jacobian assembly.
    m_NonlinearSolver.setAssembleType(m_JobSystem.getAssembleType());

    // Explicit (forward-Euler) kernels switch the transient driver to the
    // matrix-free integrator. The choice is intrinsic to the registered
    // element (ExplicitXXXElement), so it is read straight from ElmtSystem.
    m_TimeStepping.setExplicit(m_ElmtSystem.isExplicit());
    // 2nd-order explicit kernels (elastodynamics, rho*u''=F) use the
    // central-difference integrator; 1st-order ones use forward Euler;
    // quasi-static ones relax to equilibrium with ADR.
    if (m_ElmtSystem.isExplicit()) {
        m_TimeStepping.setTimeOrder(m_ElmtSystem.getTimeOrder());
        m_TimeStepping.setQuasiStatic(m_ElmtSystem.isQuasiStatic());
        // The matrix-free explicit integrators never enter NonlinearSolver, so
        // they read the run-wide assembly back-end straight from JobSystem.
        // assemble==openmp turns on the OpenMP node loops in the rate routines
        // and the explicit kernels' preprocessIteration (OMP_NUM_THREADS wide);
        // serial keeps the bitwise-identical single-thread path.
        m_TimeStepping.setParallelAssemble(m_JobSystem.getAssembleType()==AssembleType::OPENMP);
        // assemble=cuda enables the explicit PDDO fracture device path.
        m_TimeStepping.setCudaAssemble(m_JobSystem.getAssembleType()==AssembleType::CUDA);
    }

    MessagePrinter::printNormalTxt("Start to create pd mesh ...");
    m_Timer.startTimer();
    if (m_MeshModify.hasOperations()) {
        m_MeshModify.apply(m_Mesh.getMeshDataRef(),m_PDMesh);
    }
    m_PDMesh.createPDMesh(m_Mesh.getMeshDataRef());
    // Opt-in per-node horizon for non-uniform meshes. Compute it after PD-point
    // creation and before building the neighbour families.
    if (m_PDMesh.getVariableHorizon()) {
        m_PDMesh.computeVariableHorizon();
    }
    m_PDMesh.createNeighborNodes();
    m_Timer.endTimer();
    m_Timer.printElapseTime("PD mesh is generated",false);
    m_PDMesh.savePDMesh(m_InputSystem.getInputFileName());

    // Reject a Dirichlet BC that pins a Cahn-Hilliard species DoF (c/mu) on
    // boundary ghosts: it is inert (the CH species flux drops ghost bonds) and
    // would silently seal the domain. Needs the PD mesh (ghost flags + phygroup
    // node lists), so it runs here, right after the mesh is built.
    if (!m_BCSystem.validateSpeciesDirichletOnGhosts(m_PDMesh,m_ElmtSystem)) {
        MessagePrinter::printErrorTxt("PDProblem::init: invalid boundary-condition setup (see above)");
        MessagePrinter::exitPeriX();
    }

    // Report ghost walls left without any boundary condition. Their one-sided
    // PD rows are the single most common cause of a Jacobian that a direct
    // solver only just survives and an iterative solver cannot precondition at
    // all -- and the symptom (a solve that stops converging once the mesh is
    // refined) points nowhere near the input file, so say it here.
    m_BCSystem.warnUnconstrainedGhostRows(m_PDMesh,m_ElmtSystem.getMaxDofsPerNode());

    MessagePrinter::printNormalTxt("Start to setup pd operators ...");
    m_Timer.startTimer();
    m_Operators.setDim(m_Mesh.getMeshDataRef().MeshDim);
    m_Operators.setup();
    // Geometry-operator cache (PDMesh.op_cache, default on): the per-bond PDDO
    // operators depend only on the now-frozen reference geometry, so compute
    // them once here and replay them on every later calcAMatrix/calcOperators
    // call (assembly drivers, element preprocess sweeps, strong-form BC rows,
    // output projections). The matrix-free explicit drivers already replay
    // their own operator arrays (loadOperatorValues), so the cache is only
    // built for the implicit path. An oversized cache (> 8 GiB estimated)
    // skips itself and every call keeps the live-recompute behaviour.
    if (!m_ElmtSystem.isExplicit() && m_PDMesh.getOpCache()) {
        m_Operators.buildGeometryCache(m_PDMesh.getDataConstRef(),8.0);
    }
    m_Timer.endTimer();
    m_Timer.printElapseTime("PD operators is initialized",false);

    // resolve user-requested output projections against registered elements
    m_OutputSystem.resolve(m_ElmtSystem);

    // forward the output cadence (configured under "OutputSystem.interval")
    // into the transient driver, which gates SaveResults calls on it.
    m_TimeStepping.setOutputInterval(m_OutputSystem.getInterval());

    const int ndof=m_ElmtSystem.getMaxDofsPerNode();
    if (ndof<1) {
        MessagePrinter::printErrorTxt("PDProblem::init: ElmtSystem has no element registered (or 0 dofs/node)");
        MessagePrinter::exitPeriX();
    }
    const int totalDofs=m_PDMesh.getNodesNum()*ndof;

    // The explicit/ADR drivers are matrix-free: RunPDProblem routes an
    // all-explicit element set to solveExplicit*/solveADR, which never
    // assemble K nor call the linear solver. Skipping both here is
    // load-bearing, not cosmetic: a 3D run at ~2e5 nodes carries ~1.4e8
    // pattern nonzeros (GBs of CSR), and the default profile-LDU solver
    // eagerly allocates its skyline AU/AL arrays for the whole system at
    // init (~70 GB for a 200x100x8 grid) -- an instant OOM for a run that
    // never solves a linear system. (An explicit kernel with a static job
    // is rejected in run() and reaches neither driver.)
    if (m_ElmtSystem.isExplicit()) {
        MessagePrinter::printNormalTxt("Start to setup vectors (matrix-free explicit run: "
                                       "sparse matrix and linear solver are skipped) ...");
        m_Timer.startTimer();
        m_RHS.resize(totalDofs);
        m_U.resize(totalDofs);
        m_dU.resize(totalDofs);
        m_Uold.resize(totalDofs);
        m_InitialVelocity.resize(totalDofs);
        m_Timer.endTimer();
        m_Timer.printElapseTime("Vectors are initialized",true);
        return;
    }

    MessagePrinter::printNormalTxt("Start to setup sparse matrix and vectors ...");
    m_Timer.startTimer();
    m_Matrix.initFromPDMesh(m_PDMesh.getDataConstRef(),ndof);
    m_RHS.resize(totalDofs);
    m_U.resize(totalDofs);
    m_dU.resize(totalDofs);
    m_Uold.resize(totalDofs);
    m_InitialVelocity.resize(totalDofs);
    m_Timer.endTimer();
    m_Timer.printElapseTime("Sparse matrix and vectors are initialized",false);

    MessagePrinter::printNormalTxt("Start to setup linear solver ("+m_LinearSolver.getLinearSolverName()+") ...");
    m_Timer.startTimer();
    m_LinearSolver.init(m_Matrix);
    m_Timer.endTimer();
    m_Timer.printElapseTime("Linear solver ("+m_LinearSolver.getLinearSolverName()+") is initialized",true);
}

void PDProblem::printBasicInfo() {
    m_Mesh.printInfo();
    m_MeshModify.printMeshModifyInfo();
    m_PDMesh.printPDMeshInfo();
    m_ElmtSystem.printElmtSystemInfo();
    m_BCSystem.printBCSystemInfo();
    if (m_ICSystem.getICsNum()>0) m_ICSystem.printICSystemInfo();
    m_JobSystem.printJobSystemInfo();
    if (m_JobSystem.isTransient()) m_TimeStepping.printTimeSteppingInfo();
    m_NonlinearSolver.printNonlinearSolverInfo();
    // Matrix-free explicit runs skip the sparse matrix and the linear solver
    // (see init()); don't advertise an empty matrix / an unused solver backend.
    if (!m_ElmtSystem.isExplicit()) {
        m_LinearSolver.printLinearSolverInfo();
    }
    m_OutputSystem.printOutputSystemInfo();
    if (!m_ElmtSystem.isExplicit()) {
        m_Matrix.print();
    }
}
