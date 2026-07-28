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
//+++ Date    : 2026.05.30
//+++ Function: OpenMP-parallel residual / jacobian assembly. The outer
//+++           loop over source PD nodes is split across threads. Every
//+++           contribution of source node i is written to row i of
//+++           (K,RHS), so threads own disjoint rows of the assembled
//+++           system and need no atomics/locks. Each thread keeps a
//+++           private PDOperators copy and private local element buffers;
//+++           the element kernels' frozen per-node caches are filled once
//+++           by a serial preprocessIteration() and only read here. The
//+++           output is bitwise identical to the serial routine (each row
//+++           is summed by a single thread in the same bond order).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDSystem/PDSystem.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Utils/MessagePrinter.h"

namespace {
    // Opt-in assembly breakdown (PERIX_ASSEMBLE_TIMING=1): one line per call
    // splitting the preprocess sweep, the zeroing, and the bond loop. Zero
    // overhead when the environment variable is unset.
    inline bool assembleTimingOn() {
        static const bool on = [] {
            const char *e = std::getenv("PERIX_ASSEMBLE_TIMING");
            return e && e[0]=='1';
        }();
        return on;
    }
}

void PDSystem::formResidualAndJacobianParallel(const PDMesh &Mesh,
                                               PDOperators &t_PDOperators,
                                               const ElmtSystem &t_ElmtSystem,
                                               const LocalElmtInfo &Info,
                                               const VectorXd &U,
                                               const VectorXd &Uold,
                                               SparseMatrix &K,
                                               VectorXd &RHS) {
    if (t_ElmtSystem.getElementsNum()==0) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobianParallel: ElmtSystem has no element registered");
        MessagePrinter::exitPeriX();
    }

    const int ndof=t_ElmtSystem.getMaxDofsPerNode();
    const int NodesNum=Mesh.getNodesNum();

    if (RHS.getSize()!=NodesNum*ndof) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobianParallel: RHS size mismatch ("
                                      +std::to_string(RHS.getSize())+" vs "
                                      +std::to_string(NodesNum*ndof)+")");
        MessagePrinter::exitPeriX();
    }
    if (Uold.getSize()!=U.getSize()) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobianParallel: Uold size ("
                                      +std::to_string(Uold.getSize())+") does not match U ("
                                      +std::to_string(U.getSize())+")");
        MessagePrinter::exitPeriX();
    }

    // Frozen per-iteration caches are refreshed once before the threaded
    // region. Element kernels only read these caches during assembly.
    using clk=std::chrono::steady_clock;
    const bool timing=assembleTimingOn();
    const auto t0=clk::now();

    LocalElmtInfo preInfo=Info;
    preInfo.UseParallel=true;
    t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,preInfo,ndof,U,Uold);

    const auto t1=clk::now();

    K.setToZeros();
    RHS.setToZeros();

    const auto t2=clk::now();
    auto report=[&](const char *kind,const clk::time_point &tEnd) {
        if (!timing) return;
        std::printf("[assemble-timing] %s pre=%7.2fms zero=%6.2fms loop=%7.2fms\n",kind,
                    std::chrono::duration<double,std::milli>(t1-t0).count(),
                    std::chrono::duration<double,std::milli>(t2-t1).count(),
                    std::chrono::duration<double,std::milli>(tEnd-t2).count());
        std::fflush(stdout);
    };

    const auto &meshData=Mesh.getDataConstRef();

#pragma omp parallel
    {
        // --- per-thread private state ---
        // A copy of the PDDO operators so each thread mutates its own
        // A-matrix / per-bond operator scratch (calcAMatrix / calcOperators).
        // The copy inherits the (read-only after setup) multi-index tables.
        PDOperators ops=t_PDOperators;

        VectorXd localR(ndof), localR_node(ndof);
        MatrixXd localK_II(ndof,ndof), localK_IJ(ndof,ndof), localK_node(ndof,ndof);
        VectorXd U_I(ndof), U_J(ndof), Uold_I(ndof), Uold_J(ndof);

#pragma omp for schedule(guided)
        for (int i=1;i<=NodesNum;i++) {
            ops.calcAMatrix(i,meshData);
            const double volume=Mesh.getIthNodeVolume(i);

            const int rowBaseI=(i-1)*ndof;
            for (int a=1;a<=ndof;a++) {
                U_I(a)=U(rowBaseI+a);
                Uold_I(a)=Uold(rowBaseI+a);
            }

            // per-thread copy of the step context; per-bond fields are
            // overwritten below, so the nodal call sees them cleared.
            LocalElmtInfo bondInfo=Info;
            bondInfo.Xsi[0]=bondInfo.Xsi[1]=bondInfo.Xsi[2]=0.0;
            bondInfo.XsiNorm=0.0;

            // ---- nodal contribution (source / body force / time-derivative) ----
            t_ElmtSystem.computeNodalResidualAndJacobian(bondInfo,i,U_I,Uold_I,volume,
                                                        localR_node,localK_node);
            for (int a=1;a<=ndof;a++) {
                RHS.addValue(rowBaseI+a,localR_node(a));
                for (int b=1;b<=ndof;b++) {
                    K.addValue(rowBaseI+a,rowBaseI+b,-localK_node(a,b));
                }
            }

            // ---- bond contributions ----
            for (const auto &NodeJ : Mesh.getIthNodeNeighborNodeIDs(i)) {
                ops.calcOperators(i,NodeJ,meshData);
                const int rowBaseJ=(NodeJ-1)*ndof;
                for (int a=1;a<=ndof;a++) {
                    U_J(a)=U(rowBaseJ+a);
                    Uold_J(a)=Uold(rowBaseJ+a);
                }

                bondInfo.Xsi[0]=meshData.NodeCoords[(NodeJ-1)*3+0]-meshData.NodeCoords[(i-1)*3+0];
                bondInfo.Xsi[1]=meshData.NodeCoords[(NodeJ-1)*3+1]-meshData.NodeCoords[(i-1)*3+1];
                bondInfo.Xsi[2]=meshData.NodeCoords[(NodeJ-1)*3+2]-meshData.NodeCoords[(i-1)*3+2];
                bondInfo.XsiNorm=std::sqrt(bondInfo.Xsi[0]*bondInfo.Xsi[0]
                                          +bondInfo.Xsi[1]*bondInfo.Xsi[1]
                                          +bondInfo.Xsi[2]*bondInfo.Xsi[2]);

                t_ElmtSystem.computeBondResidualAndJacobian(
                    ops,bondInfo,i,NodeJ,U_I,U_J,Uold_I,Uold_J,volume,
                    localR,localK_II,localK_IJ);

                // Newton: K = -dF/dU, RHS = +F. All writes target row i, so
                // no two threads touch the same matrix/vector entry.
                for (int a=1;a<=ndof;a++) {
                    RHS.addValue(rowBaseI+a,localR(a));
                    for (int b=1;b<=ndof;b++) {
                        K.addValue(rowBaseI+a,rowBaseI+b,-localK_II(a,b));
                        K.addValue(rowBaseI+a,rowBaseJ+b,-localK_IJ(a,b));
                    }
                }
            }
        }
    }
    report("bond-loop",clk::now());
}
