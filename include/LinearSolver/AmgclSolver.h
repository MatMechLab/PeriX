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
//+++ Date    : 2026.07.30
//+++ Function: Minimal public AMGCL iterative linear-solver
//+++           backend. The public input controls only tolerance,
//+++           iteration limit, and progress reporting; the solver
//+++           algorithm is fixed to BiCGSTAB with smoothed-
//+++           aggregation AMG and ILU(0) relaxation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <memory>

#include "LinearSolver/LinearSolverBase.h"

/**
 * AMGCL iterative solver using the CPU/OpenMP builtin backend.
 *
 * The AMGCL template implementation is hidden behind a pimpl so including the
 * public LinearSolver dispatcher does not instantiate AMGCL in every source
 * file. The fixed PeriX CSR pattern is converted once during initSolver().
 */
class AmgclSolver : public LinearSolverBase {
public:
    AmgclSolver();
    ~AmgclSolver() override;

    /**
     * Cache the CSR pattern and configure the iterative solver.
     * @param K sparse matrix whose pattern is final
     * @param Parameters allow-listed JSON parameters: tol, maxiter, verbose
     */
    void initSolver(SparseMatrix &K,nlohmann::ordered_json &Parameters) final;

    /**
     * Solve the current numerical system and verify the true relative residual.
     * @param A current sparse matrix
     * @param b right-hand side
     * @param x solution
     * @return true only when ||b-Ax||_2/||b||_2 is at most tol
     */
    bool solveLinearSystem(SparseMatrix &A,VectorXd &b,VectorXd &x) final;

private:
    void releaseInternalMemory();

    /**
     * Detect a full node-block CSR layout. Public PeriX models use one to five
     * degrees of freedom per node; scalar storage is used if no block pattern
     * is present.
     */
    [[nodiscard]] int detectBlockSize(SparseMatrix &K) const;

    static constexpr int kMaxBlockSize=5;

    struct Impl;
    std::unique_ptr<Impl> m_Impl;

    bool m_IsInitialized=false;
    int m_N=0;
    int m_NNZ=0;
};
