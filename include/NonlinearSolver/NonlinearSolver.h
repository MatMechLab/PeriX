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
//+++ Function: Newton-Raphson nonlinear solver family. Drives the
//+++           inner iteration of one (static or transient) load step:
//+++             1) PDSystem.formResidualAndJacobian()
//+++             2) BCSystem.applyBCs()
//+++             3) LinearSolver.solve(K, RHS, dU)
//+++             4) U += alpha*dU; BCSystem.presetSolution()
//+++           Convergence is declared when |R| < abs_tol or
//+++           |R| <= rel_tol * |R|_at_iter_0.
//+++
//+++           PDProblem owns a NonlinearSolver, configures it from
//+++           the "NonlinearSolver" input block, and calls solve()
//+++           once per static analysis or once per transient step.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <vector>

#include "BCSystem/BCSystem.h"
#include "ElmtSystem/ElmtSystem.h"
#include "ElmtSystem/LocalElmtInfo.h"
#include "JobSystem/AssembleType.h"
#include "LinearSolver/LinearSolver.h"
#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "PDSystem/PDSystem.h"
#include "Utils/Timer.h"

class NonlinearSolver {
public:
    NonlinearSolver()=default;


    void setMaxIters(const int &n)         { m_MaxIters=n; }
    [[nodiscard]] int getMaxIters() const  { return m_MaxIters; }

    void setAbsTol(const double &tol)         { m_AbsTol=tol; }
    [[nodiscard]] double getAbsTol() const    { return m_AbsTol; }

    void setRelTol(const double &tol)         { m_RelTol=tol; }
    [[nodiscard]] double getRelTol() const    { return m_RelTol; }

    /** show per-iter progress lines while solving (default true) */
    void setVerbose(const bool &v)         { m_Verbose=v; }
    [[nodiscard]] bool isVerbose() const   { return m_Verbose; }

    /**
     * which assembler the Newton loop dispatches to. Sourced from the
     * "JobSystem":{"assemble":...} input key (validated against the
     * compiled-in back-ends there); the serial CPU assembler is the
     * default and is always available.
     */
    void setAssembleType(const AssembleType &t)        { m_AssembleType=t; }
    [[nodiscard]] AssembleType getAssembleType() const { return m_AssembleType; }

    /**
     * run Newton-Raphson for the current step.
     * Returns true on convergence, false if MaxIters was reached.
     *
     * @param Mesh         the PD mesh
     * @param Operators    PD differential operators (mutated per-bond)
     * @param ElmtSys      element-kernel registry
     * @param BCSys        boundary-condition system (row elimination / mirror)
     * @param PDSys        residual/jacobian assembler
     * @param LinSolver    linear solver (Pardiso wrapper)
     * @param Info         per-step context (dt, t, step index)
     * @param DofsPerNode  total DoFs per pd node
     * @param K            workspace sparse matrix (will be overwritten)
     * @param U            initial guess in / converged solution out
     * @param Uold         previous-step solution (transient)
     * @param dU           workspace (Newton update; overwritten)
     * @param RHS          workspace (residual; overwritten)
     */
    bool solve(const PDMesh &Mesh,
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
               VectorXd &RHS);

    [[nodiscard]] int    getLastIters() const  { return m_LastIters; }
    [[nodiscard]] double getLastRNorm() const  { return m_LastRNorm; }
    [[nodiscard]] double getLastDUNorm() const { return m_LastDUNorm; }
    [[nodiscard]] double getLastRNorm0() const { return m_LastRNorm0; }
    [[nodiscard]] bool   getLastConverged() const { return m_LastConverged; }

    void printNonlinearSolverInfo() const;

private:
    /** assemble the residual+Jacobian with the selected back-end and apply the
     *  BCs into (K,RHS) -- the single step shared by the main Newton iteration
     *  and the line-search trial evaluations. */
    void formSystem(const PDMesh &Mesh,PDOperators &Operators,
                    const ElmtSystem &ElmtSys,const BCSystem &BCSys,PDSystem &PDSys,
                    const LocalElmtInfo &Info,const int &DofsPerNode,
                    const VectorXd &U,const VectorXd &Uold,
                    SparseMatrix &K,VectorXd &RHS) const;



    int    m_MaxIters=50;
    double m_AbsTol=1.0e-9;
    double m_RelTol=1.0e-12;
    bool   m_Verbose=true;
    // residual/Jacobian assembler picked at run time (default serial).
    AssembleType m_AssembleType=AssembleType::SERIAL;

    // last-call status
    int    m_LastIters=0;
    double m_LastRNorm=0.0;
    double m_LastRNorm0=0.0;
    double m_LastDUNorm=0.0;
    bool   m_LastConverged=false;

    Timer  m_Timer;
};
