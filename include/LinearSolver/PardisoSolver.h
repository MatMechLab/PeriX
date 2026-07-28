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
//+++ Date    : 2026.04.14
//+++ Function: The pardiso solver class, one must install oneAPI
//+++           to use this class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <vector>

#include "mkl_pardiso.h"
#include "LinearSolver/LinearSolverBase.h"

class PardisoSolver :public LinearSolverBase {
public:
    /**
     * default constructor
     */
    PardisoSolver();
    ~PardisoSolver() override;

    /**
     * init the linear solver based on the input sparse matrix
     * @param K sparse matrix
     * @param Parameters json parameters read from input file
     */
    void initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) final;

    /**
     * solve the linear equation system
     * @param A sparse matrix
     * @param b right hand side vector
     * @param x solution vector
     * @return true if everything is ok, otherwise false
     */
    bool solveLinearSystem(SparseMatrix &A,VectorXd &b,VectorXd &x) final;

private:
    void releaseInternalMemory();

    MKL_INT N;/**< size of current equations */
    MKL_INT mtype = 11;/**< Real unsymmetric matrix */
    MKL_INT nrhs = 1;/**< Number of right hand sides. */
    void *pt[64];/**< Internal solver memory pointer pt */
    MKL_INT iparm[64];/**< Pardiso control parameters. */
    MKL_INT maxfct, mnum, phase, error, msglvl;/**< Auxiliary variables */
    MKL_INT m_FactorPhase=23;/**< phase for a non-reuse solve: 23 (numeric factorize+solve,
                                  analysis reused from initSolver) or 13 (full re-analysis,
                                  needed only when value-dependent scaling/matching is on) */
    double ddum;/**< Double dummy */
    MKL_INT idum;/**< Integer dummy. */
    bool m_IsInitialized=false;

    // ---- factorization reuse ----
    // A snapshot of the matrix VALUES of the most recently factorized system.
    // When the next system to solve is bit-identical (same sparsity is
    // guaranteed by the fixed CSR pattern; values compared exactly), the
    // numerical factorization is skipped and only the (cheap) solve phase runs.
    // This is fully transparent and always correct: a kernel whose Jacobian
    // changes every iteration never matches, so its behaviour
    // is unchanged, while a constant-tangent kernel (e.g. dynamic_frac_mechanics,
    // whose matrix only changes when a bond breaks) reuses one factorization
    // across many Newton iterations and time steps.
    std::vector<double> m_LastValues;
    bool m_HaveFactor=false;
};
