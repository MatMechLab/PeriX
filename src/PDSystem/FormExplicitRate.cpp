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
//+++ Date    : 2026.06.03
//+++ Function: matrix-free explicit (forward-Euler) rate assembly,
//+++           serial or OpenMP, with a reference-geometry operator cache.
//+++           Walks every PD node and bond and accumulates the nodal rate
//+++           R_i of du_i/dt = R_i(U) -- the per-bond/per-node physics
//+++           supplied by the explicit ElmtSystem kernels via
//+++           computeBondResidual / computeNodalResidual. No Jacobian and
//+++           no sparse matrix are formed.
//+++
//+++           Two optimisations, both bitwise identical to the naive
//+++           serial sweep:
//+++           1) OpenMP: when Info.UseParallel (JobSystem.assemble==openmp)
//+++              the node loop is split across OMP_NUM_THREADS threads;
//+++              every source node i writes only its own rows of Rate, so
//+++              threads own disjoint rows (no atomics) and each keeps a
//+++              private PDOperators copy + buffers.
//+++           2) Operator cache: the PDDO per-bond operator vector depends
//+++              only on the (fixed) reference geometry, so it is the SAME
//+++              at every time step. It is solved ONCE (calcAMatrix +
//+++              calcOperators) into m_FEOpVals and thereafter replayed with
//+++              loadOperatorValues, removing the per-step per-node
//+++              moment-matrix solve -- the dominant cost of a PDDO
//+++              forward-Euler step -- in exchange for
//+++              O(total_bonds * opVecSize) doubles of RAM.
//+++
//+++           Index layout matches the implicit assembler:
//+++             g = (NodeID-1)*ndof + a   (1-based).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDSystem/PDSystem.h"

#include <cmath>

#include "Utils/MessagePrinter.h"

void PDSystem::formExplicitRate(const PDMesh &Mesh,
                                PDOperators &t_PDOperators,
                                const ElmtSystem &t_ElmtSystem,
                                const LocalElmtInfo &Info,
                                const VectorXd &U,
                                const VectorXd &Uold,
                                VectorXd &Rate) {
    if (t_ElmtSystem.getElementsNum()==0) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitRate: ElmtSystem has no element registered");
        MessagePrinter::exitPeriX();
    }

    const int ndof=t_ElmtSystem.getMaxDofsPerNode();
    const int NodesNum=Mesh.getNodesNum();

    if (Rate.getSize()!=NodesNum*ndof) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitRate: Rate size mismatch ("
                                      +std::to_string(Rate.getSize())+" vs "
                                      +std::to_string(NodesNum*ndof)+")");
        MessagePrinter::exitPeriX();
    }
    if (Uold.getSize()!=U.getSize()) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitRate: Uold size ("
                                      +std::to_string(Uold.getSize())+") does not match U ("
                                      +std::to_string(U.getSize())+")");
        MessagePrinter::exitPeriX();
    }

    // Freeze any per-node caches (e.g. |grad phi^n|) once before the sweep.
    // For forward Euler the frozen level IS the current field U=u^n, so the
    // explicit kernels cache from U; Uold is forwarded for completeness. The
    // kernels parallelise this per-node pass internally when Info.UseParallel.
    //
    // CONTRACT WARNING for kernel authors: Uold means DIFFERENT things per
    // explicit driver -- forward Euler passes Uold==U==u^n, central difference
    // passes u^{n-1}, and velocity-Verlet/ADR never advance it (it stays the
    // t=0 state for the whole run). No shipped explicit kernel reads Uold; a
    // new one that does must first unify these semantics driver-side.
    t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,Info,ndof,U,Uold);

    Rate.setToZeros();

    const auto &meshData=Mesh.getDataConstRef();
    const int opVecSize=t_PDOperators.getOperatorsVecSize();

    // ---- (re)build the reference-geometry operator cache once per run ----
    // The operators are a function of the fixed reference geometry only, so the
    // moment-matrix solve (calcAMatrix) and per-bond operator eval (calcOperators)
    // give the SAME result every step. Solve them once here and replay below.
    long long totalBonds=0;
    for (int i=1;i<=NodesNum;i++)
        totalBonds+=static_cast<long long>(Mesh.getIthNodeNeighborNodeIDs(i).size());
    const bool stale=!m_FEOpCacheValid || m_FEOpCacheNodes!=NodesNum
                     || m_FEOpVecSize!=opVecSize
                     || static_cast<long long>(m_FEOpVals.size())!=totalBonds*opVecSize;
    if (stale) {
        m_FEOpRowStart.assign(static_cast<std::size_t>(NodesNum)+1,0);
        for (int i=1;i<=NodesNum;i++)
            m_FEOpRowStart[static_cast<std::size_t>(i)]=
                m_FEOpRowStart[static_cast<std::size_t>(i-1)]
                +static_cast<long long>(Mesh.getIthNodeNeighborNodeIDs(i).size());
        m_FEOpVals.assign(static_cast<std::size_t>(totalBonds*opVecSize),0.0);

        const double mb=static_cast<double>(totalBonds)*opVecSize*sizeof(double)/(1024.0*1024.0);
        char buf[160];
        std::snprintf(buf,sizeof(buf),
            "explicit operator cache: %.1f MB for %lld bonds x %d ops "
            "(reference geometry, solved once and reused every step)",
            mb,totalBonds,opVecSize);
        MessagePrinter::printNormalTxt(buf);

#pragma omp parallel if(Info.UseParallel)
        {
            PDOperators ops=t_PDOperators;
#pragma omp for schedule(guided)
            for (int i=1;i<=NodesNum;i++) {
                ops.calcAMatrix(i,meshData);
                const long long base=m_FEOpRowStart[static_cast<std::size_t>(i-1)]*opVecSize;
                const auto &nb=Mesh.getIthNodeNeighborNodeIDs(i);
                for (std::size_t jdx=0;jdx<nb.size();jdx++) {
                    ops.calcOperators(i,nb[jdx],meshData);
                    double *dst=&m_FEOpVals[static_cast<std::size_t>(base
                                +static_cast<long long>(jdx)*opVecSize)];
                    for (int k=1;k<=opVecSize;k++) dst[k-1]=ops.getOperatorValue(k);
                }
            }
        }
        m_FEOpCacheValid=true; m_FEOpCacheNodes=NodesNum; m_FEOpVecSize=opVecSize;
    }

    // Single code path for serial and OpenMP: with Info.UseParallel==false the
    // region runs on one thread, so the sweep is the exact serial one and the
    // output is bitwise the same. Every write targets row i, so threads never
    // touch a shared entry.
#pragma omp parallel if(Info.UseParallel)
    {
        // per-thread private operator object (its m_OperatorsVec is overwritten
        // per bond from the cache) and per-thread local buffers.
        PDOperators ops=t_PDOperators;
        VectorXd localR(ndof), localR_node(ndof);
        VectorXd U_I(ndof), U_J(ndof), Uold_I(ndof), Uold_J(ndof);

#pragma omp for schedule(guided)
        for (int i=1;i<=NodesNum;i++) {
            const double volume=Mesh.getIthNodeVolume(i);
            const long long base=m_FEOpRowStart[static_cast<std::size_t>(i-1)]*opVecSize;

            const int rowBaseI=(i-1)*ndof;
            for (int a=1;a<=ndof;a++) {
                U_I(a)=U(rowBaseI+a);
                Uold_I(a)=Uold(rowBaseI+a);
            }

            // per-thread copy of the step context; clear stale per-bond fields
            // for the nodal call (mirrors the implicit assembler's convention).
            LocalElmtInfo bondInfo=Info;
            bondInfo.Xsi[0]=bondInfo.Xsi[1]=bondInfo.Xsi[2]=0.0;
            bondInfo.XsiNorm=0.0;

            // ---- nodal rate (source / reaction; NO time-derivative term) ----
            t_ElmtSystem.computeNodalResidual(bondInfo,i,U_I,Uold_I,volume,localR_node);
            for (int a=1;a<=ndof;a++) {
                Rate.addValue(rowBaseI+a,localR_node(a));
            }

            // ---- bond rates (spatial operator) ----
            const auto &nb=Mesh.getIthNodeNeighborNodeIDs(i);
            for (std::size_t jdx=0;jdx<nb.size();jdx++) {
                const int NodeJ=nb[jdx];
                // replay the cached operators for this bond (constant geometry);
                // no calcAMatrix / calcOperators on the hot path.
                ops.loadOperatorValues(&m_FEOpVals[static_cast<std::size_t>(base
                                      +static_cast<long long>(jdx)*opVecSize)]);
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

                t_ElmtSystem.computeBondResidual(
                    ops,bondInfo,i,NodeJ,U_I,U_J,Uold_I,Uold_J,volume,localR);

                // du_i/dt = R_i : the rate is accumulated directly (no Newton flip).
                for (int a=1;a<=ndof;a++) {
                    Rate.addValue(rowBaseI+a,localR(a));
                }
            }
        }
    }
}
