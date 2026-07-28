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
//+++ Date    : 2026.05.08
//+++ Function: NonlinearSolver implementation: the classic
//+++           Newton-Raphson iteration.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "NonlinearSolver/NonlinearSolver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>

#include "Utils/MessagePrinter.h"

void NonlinearSolver::formSystem(const PDMesh &Mesh,PDOperators &Operators,
                                 const ElmtSystem &ElmtSys,const BCSystem &BCSys,
                                 PDSystem &PDSys,const LocalElmtInfo &Info,
                                 const int &DofsPerNode,
                                 const VectorXd &U,const VectorXd &Uold,
                                 SparseMatrix &K,VectorXd &RHS) const {
    // residual + Jacobian with the selected back-end (OpenMP is bit-identical
    // to serial; CUDA matches to round-off), then BC row elimination / mirror.
    switch (m_AssembleType) {
    case AssembleType::CUDA:
#ifdef PERIX_CUDA_ASSEMBLE
        PDSys.formResidualAndJacobianCUDA(Mesh,Operators,ElmtSys,Info,U,Uold,K,RHS);
#else
        PDSys.formResidualAndJacobian(Mesh,Operators,ElmtSys,Info,U,Uold,K,RHS);
#endif
        break;
    case AssembleType::OPENMP:
        PDSys.formResidualAndJacobianParallel(Mesh,Operators,ElmtSys,Info,U,Uold,K,RHS);
        break;
    case AssembleType::SERIAL:
    default:
        PDSys.formResidualAndJacobian(Mesh,Operators,ElmtSys,Info,U,Uold,K,RHS);
        break;
    }
    // Opt-in BC-application timing, completing the PERIX_ASSEMBLE_TIMING
    // breakdown (the pre/zero/loop split is printed by the assembly driver).
    static const bool bcTiming = [] {
        const char *e=std::getenv("PERIX_ASSEMBLE_TIMING");
        return e && e[0]=='1';
    }();
    if (bcTiming) {
        const auto tb=std::chrono::steady_clock::now();
        BCSys.applyBCs(Mesh,Operators,DofsPerNode,U,Uold,Info,K,RHS);
        const double ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-tb).count();
        std::printf("[assemble-timing] bc=%6.2fms\n",ms);
        std::fflush(stdout);
        return;
    }
    BCSys.applyBCs(Mesh,Operators,DofsPerNode,U,Uold,Info,K,RHS);
}


bool NonlinearSolver::solve(const PDMesh &Mesh,
                            PDOperators &Operators,
                            const ElmtSystem &ElmtSys,
                            const BCSystem &BCSys,
                            PDSystem &PDSys,
                            LinearSolver &LinSolver,
                            const LocalElmtInfo &Info,
                            const int &DofsPerNode,
                            SparseMatrix &K,
                            VectorXd &U,
                            const VectorXd &Uold,
                            VectorXd &dU,
                            VectorXd &RHS) {
    constexpr int bufSize=160;
    char buffer[bufSize];

    int    Iters=0;
    bool   Converged=false;
    double RNorm=0.0,RNorm0=1.0,dUNorm=0.0;

    while (!Converged && Iters<m_MaxIters) {
        // ---- assemble residual + jacobian (selected back-end) + BCs ----
        m_Timer.startTimer();
        formSystem(Mesh,Operators,ElmtSys,BCSys,PDSys,Info,DofsPerNode,U,Uold,K,RHS);
        m_Timer.endTimer();
        const double assembleTime=m_Timer.getDurationInSecond();

        // ---- linear solve: K * dU = RHS  (full Newton direction) ----
        m_Timer.startTimer();
        const bool linOK=LinSolver.solve(K,RHS,dU);
        m_Timer.endTimer();
        const double solveTime=m_Timer.getDurationInSecond();

        // A failed or non-finite linear solve (e.g. cuDSS hitting a zero pivot on
        // an indefinite Jacobian and returning NaN) must NOT be applied: U += NaN
        // poisons U and every subsequent step, so the adaptive stepper could never
        // recover even after a cutback. Bail out with the LAST GOOD U and report
        // non-convergence, so the transient driver cuts dt back and retries.
        if (!linOK || !std::isfinite(dU.norm())) {
            MessagePrinter::printWarningTxt("NonlinearSolver: linear solve failed or produced "
                "a non-finite Newton update; aborting this Newton iteration so the timestep is "
                "cut back (was the last dt too large / the tangent singular?)");
            m_LastIters=Iters; m_LastRNorm=RNorm; m_LastRNorm0=RNorm0;
            m_LastDUNorm=dUNorm; m_LastConverged=false;
            return false;
        }

            // ---- Newton update: classic full step ----
            U+=dU;
            BCSys.presetSolution(Mesh,DofsPerNode,U,Info.T);
            RNorm=RHS.norm();
            dUNorm=dU.norm();

        if (Iters==0) RNorm0=RNorm;
        Iters++;

            if (m_Verbose) {
                std::snprintf(buffer,bufSize,
                    "  iters=%2d,|R|=%9.3e,|dU|=%9.3e [assemble=%9.3es,solve=%9.3es]",
                    Iters,RNorm,dUNorm,assembleTime,solveTime);
                MessagePrinter::printNormalTxt(buffer);
            }

        if (RNorm<m_AbsTol || (RNorm0>0.0 && RNorm<=m_RelTol*RNorm0)) {
            Converged=true;
            break;
        }
    }

    m_LastIters=Iters;
    m_LastRNorm=RNorm;
    m_LastRNorm0=RNorm0;
    m_LastDUNorm=dUNorm;
    m_LastConverged=Converged;
    return Converged;
}


void NonlinearSolver::printNonlinearSolverInfo() const {
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Nonlinear solver info (Newton-Raphson):");
    MessagePrinter::printNormalTxt("  max_iters="+std::to_string(m_MaxIters)
                                   +", abs_tol="+std::to_string(m_AbsTol)
                                   +", rel_tol="+std::to_string(m_RelTol));
    MessagePrinter::printNormalTxt("  assemble="+assembleTypeName(m_AssembleType));
    MessagePrinter::printStars();
}
