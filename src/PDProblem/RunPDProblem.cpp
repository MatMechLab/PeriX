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
//+++ Function: drive the analysis (static or transient).
//+++           For static  : a single Newton solve.
//+++           For transient: delegate the outer time loop to
//+++                          TimeStepping (Backward Euler), which
//+++                          calls NonlinearSolver per step.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDProblem/PDProblem.h"

void PDProblem::run(int args,char *argv[]) {
    init(args,argv);
    if (m_InputSystem.isReadOnly()) return;
    printBasicInfo();

    Timer allTimer;
    constexpr int bufSize=160;
    char buffer[bufSize];
    const int ndof=m_ElmtSystem.getMaxDofsPerNode();

    m_U.setToZeros();
    m_InitialVelocity.setToZeros();
    // 1) user-defined initial conditions write the solution and, separately,
    //    the t=0 velocity used to initialize central-difference history.
    m_ICSystem.applyInitialConditions(
        m_PDMesh,ndof,m_U,m_InitialVelocity);
    // 2) BCs win at constrained boundaries (Dirichlet hard-pin, mirror
    //    Neumann ghosts copy adjusted bulk values).
    m_BCSystem.presetSolution(m_PDMesh,ndof,m_U);
    m_Uold=m_U;

    if (m_ICSystem.hasVelocityIC()
        && (!m_ElmtSystem.isExplicit()
            || m_ElmtSystem.isQuasiStatic()
            || m_ElmtSystem.getTimeOrder()<2)) {
        MessagePrinter::printErrorTxt(
            "velocity initial conditions require a second-order explicit "
            "transient kernel (central difference)");
        MessagePrinter::exitPeriX();
    }

    allTimer.startTimer();

    // Explicit (forward-Euler) kernels are transient-only: there is no
    // steady "rate = 0" solve in this matrix-free path.
    if (m_ElmtSystem.isExplicit() && !m_JobSystem.isTransient()) {
        MessagePrinter::printErrorTxt("explicit (forward-Euler) element kernels require a transient "
                                      "analysis; set \"JobSystem\":{\"type\":\"transient\"} and provide a "
                                      "\"TimeStepping\" block.");
        MessagePrinter::exitPeriX();
    }

    if (!m_JobSystem.isTransient()) {
        // -------------------- static analysis --------------------
        LocalElmtInfo info;
        info.Dt=0.0;
        info.T=0.0;
        info.Timestep=0;

        const bool ok=m_NonlinearSolver.solve(m_PDMesh,m_Operators,m_ElmtSystem,m_BCSystem,
                                              m_PDSystem,m_LinearSolver,info,ndof,
                                              m_Matrix,m_U,m_Uold,m_dU,m_RHS);
        allTimer.endTimer();
        if (ok) {
            if (m_OutputSystem.wantsVTU()) {
                savePDResults(m_InputSystem.getInputFileName(),0,0.0);
                saveElementResults(m_InputSystem.getInputFileName(),0,0.0);
            }
            if (m_OutputSystem.wantsExodus())
                saveExodusResults(m_InputSystem.getInputFileName(),0,0.0);
            MessagePrinter::printStars();
            snprintf(buffer,bufSize,"Static solve is done in %d Newton iterations, |R|=%10.3e, elapsed time=%10.3es",
                     m_NonlinearSolver.getLastIters(),m_NonlinearSolver.getLastRNorm(),
                     allTimer.getDurationInSecond());
            MessagePrinter::printNormalTxt(buffer);
            MessagePrinter::printStars();
        }
        else {
            MessagePrinter::printErrorTxt("static solve did not converge, |R|="
                                          +std::to_string(m_NonlinearSolver.getLastRNorm()));
            MessagePrinter::exitPeriX();
        }
        return;
    }

    // -------------------- transient analysis --------------------
    // Wrap the per-step output into a single callback so TimeStepping
    // does not need to know about the input file path or the FEM/PD
    // split; PDProblem keeps owning the output side.
    const string &fname=m_InputSystem.getInputFileName();
    auto saveAll=[this,&fname](const int &step,const double &time) {
        if (m_OutputSystem.wantsVTU()) {
            savePDResults(fname,step,time);
            saveElementResults(fname,step,time);
        }
        if (m_OutputSystem.wantsExodus()) saveExodusResults(fname,step,time);
    };

    // Explicit kernels use a matrix-free integrator (no Jacobian / linear
    // solve); m_RHS doubles as the nodal-rate buffer. A quasi-static kernel
    // relaxes to static equilibrium with ADR; a 1st-order one (du/dt=R)
    // advances with forward Euler; a 2nd-order one (rho*u''=R, elastodynamics)
    // with central difference. Everything else uses the backward-Euler Newton
    // driver.
    bool ok;
    if (m_ElmtSystem.isExplicit()) {
        // The explicit/ADR drivers are matrix-free: they only call
        // BCSystem::presetSolution and never assemble K/RHS, so a strong
        // traction equation or source-form species flux would be ignored.
        // Refuse loudly instead of running an unloaded model.
        const auto matrixOnlyBCs=m_BCSystem.getLinearSystemOnlyBCs();
        if (!matrixOnlyBCs.empty()) {
            std::string list;
            for (const auto &n : matrixOnlyBCs) {
                if (!list.empty()) list+=", ";
                list+=n;
            }
            MessagePrinter::printErrorTxt("the matrix-free explicit/ADR drivers cannot apply "
                                          "linear-system-only BCs ["+list+"]: these act through the "
                                          "assembled K/RHS, which is never built on this path. "
                                          "Initialize an impact velocity through ICSystem, use an "
                                          "element load, or switch to an implicit kernel.");
            MessagePrinter::exitPeriX();
        }
        if (m_ElmtSystem.isQuasiStatic()) {
            ok=m_TimeStepping.solveADR(m_PDMesh,m_Operators,m_ElmtSystem,m_BCSystem,
                                       m_PDSystem,ndof,m_U,m_Uold,m_RHS,saveAll);
        }
        else if (m_ElmtSystem.getTimeOrder()>=2) {
            ok=m_TimeStepping.solveExplicitDynamics(m_PDMesh,m_Operators,m_ElmtSystem,m_BCSystem,
                                                    m_PDSystem,ndof,m_U,m_Uold,
                                                    m_InitialVelocity,m_RHS,saveAll);
        }
        else {
            ok=m_TimeStepping.solveExplicit(m_PDMesh,m_Operators,m_ElmtSystem,m_BCSystem,
                                            m_PDSystem,ndof,m_U,m_Uold,m_RHS,saveAll);
        }
    }
    else {
        ok=m_TimeStepping.solve(m_PDMesh,m_Operators,m_ElmtSystem,m_BCSystem,
                                m_PDSystem,m_LinearSolver,m_NonlinearSolver,
                                ndof,m_Matrix,m_U,m_Uold,m_dU,m_RHS,
                                saveAll);
    }
    allTimer.endTimer();

    MessagePrinter::printStars();
    if (ok) {
        snprintf(buffer,bufSize,
                 "Transient solve is done, %d steps, t=%12.5e, cutbacks=%d, elapsed time=%10.3es",
                 m_TimeStepping.getLastSteps(),m_TimeStepping.getLastTime(),
                 m_TimeStepping.getLastCutbacks(),allTimer.getDurationInSecond());
        MessagePrinter::printNormalTxt(buffer);
    }
    else {
        snprintf(buffer,bufSize,
                 "Transient solve aborted after %d steps, t=%12.5e, cutbacks=%d, elapsed time=%10.3es",
                 m_TimeStepping.getLastSteps(),m_TimeStepping.getLastTime(),
                 m_TimeStepping.getLastCutbacks(),allTimer.getDurationInSecond());
        MessagePrinter::printErrorTxt(buffer);
    }
    MessagePrinter::printStars();
    if (!ok) {
        MessagePrinter::exitPeriX();
    }
}
