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
//+++ Date    : 2026.05.25
//+++ Function: PeriX's default in-house direct sparse linear
//+++           solver. Profile (skyline) LDU factorization on a
//+++           unified column-wise profile with independent upper
//+++           (AU) and lower (AL) packed arrays so unsymmetric
//+++           Jacobians are factored without any pivoting. A
//+++           Reverse Cuthill-McKee (RCM) symmetric reordering is
//+++           applied by default to compress the profile for PD-
//+++           like banded patterns. No external library dependency.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <vector>

#include "LinearSolver/LinearSolverBase.h"

/**
 * PeriX's default in-house direct sparse linear solver.
 *
 * The solver stores the matrix in a unified column-wise profile
 * (skyline) form:
 *   m_JP[j+1]-m_JP[j]   number of off-diagonal entries in column j
 *                       AND in row j (same envelope, so the matrix
 *                       can be unsymmetric in value but the profile
 *                       is taken as the union of upper- and lower-
 *                       triangular nonzero patterns).
 *   m_AU[m_JP[j]..m_JP[j+1]-1] stores A(j-K_j..j-1, j) (column j,
 *                       upper triangle) packed top-to-bottom.
 *   m_AL[m_JP[j]..m_JP[j+1]-1] stores A(j, j-K_j..j-1) (row j,
 *                       lower triangle) packed left-to-right.
 *   m_AD[j]            stores A(j,j); replaced by 1/D(j) after
 *                       factorization.
 *
 * Factorization: A = L * D * U where L is unit lower triangular,
 * D is diagonal and U is unit upper triangular.  No pivoting.
 *
 * After factorization, m_AU stores U (final), m_AL stores L
 * (final), m_AD stores 1/D.
 */
class DefaultSolver : public LinearSolverBase {
public:
    DefaultSolver();
    ~DefaultSolver() override = default;

    /**
     * Build the skyline profile structure from a CSR sparse matrix
     * and optionally print storage statistics. The actual
     * factorization is deferred to solveLinearSystem (a single
     * symbolic-init then per-solve numerical-factor flow is what the
     * PeriX framework drives).
     * @param K sparse matrix (CSR, 0-based)
     * @param Parameters json parameters read from input file
     */
    void initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) final;

    /**
     * Numerically factor A (re-using the cached skyline structure
     * from initSolver) and solve A x = b. Falls back to a fresh
     * structural setup if the sparsity pattern changed.
     * @param A coefficient matrix in CSR form
     * @param b right-hand side
     * @param x solution
     * @return true on success, false if a zero pivot is encountered
     */
    bool solveLinearSystem(SparseMatrix &A, VectorXd &b, VectorXd &x) final;

private:
    /**
     * Compute a symmetric reordering of equations (RCM by default,
     * identity if disabled). Populates m_Perm and m_InvPerm so that
     * downstream profile work uses the new ordering.
     * @param A coefficient matrix (read-only)
     */
    void computeOrdering(SparseMatrix &A);

    /**
     * Build the skyline pointer array m_JP from the CSR pattern of A
     * using the current m_InvPerm, and allocate AU, AL, AD storage.
     * Aborts with a clear message (suggesting PARDISO/cuDSS) if the
     * projected allocation exceeds m_MaxProfileGB. Returns the total
     * profile size (m_JP[n]).
     * @param A coefficient matrix
     */
    long long buildProfile(SparseMatrix &A);

    /**
     * Zero AU/AL/AD then copy CSR numerical values into the skyline
     * arrays. Applies m_InvPerm so each entry A(i,j) is placed at
     * skyline position corresponding to (m_InvPerm[i], m_InvPerm[j]).
     * @param A coefficient matrix
     */
    void copyValuesFromCSR(SparseMatrix &A);

    /**
     * Compute A = L*D*U in place (using m_AU, m_AL, m_AD). On exit
     * m_AU stores U, m_AL stores L, m_AD stores 1/D.
     * @return true on success, false on zero pivot
     */
    bool factorize();

    /**
     * Solve A x = b given the factored L*D*U. Applies the
     * permutation in the right places: b -> b_perm via m_Perm,
     * solve in permuted space, then permute solution back.
     * @param b right-hand side (read)
     * @param x solution (written)
     */
    void solveFactored(const VectorXd &b, VectorXd &x) const;

    /**
     * Fast check: does the cached (rows,nnz) fingerprint still
     * match the input matrix? If not the profile must be rebuilt.
     * @param A current matrix
     * @return true if pattern likely unchanged
     */
    bool patternMatches(SparseMatrix &A) const;

private:
    int                 m_N = 0;            /**< number of equations */
    std::vector<long long> m_JP;            /**< profile pointers, size N+1, m_JP[0]=0 */
    std::vector<double> m_AD;               /**< diagonal (size N) -- reciprocal after factor */
    std::vector<double> m_AU;               /**< upper triangle packed by column */
    std::vector<double> m_AL;               /**< lower triangle packed by row */

    // Symmetric reordering. m_Perm[new] = old (the equation that
    // ends up at slot `new`). m_InvPerm[old] = new (where the
    // original equation `old` lives in the factored arrays).
    std::vector<int>    m_Perm;
    std::vector<int>    m_InvPerm;
    bool                m_UseOrdering = true;

    // Working vector used by solveFactored: holds b permuted into
    // the factor's ordering on entry and the solution in that
    // ordering on exit. Sized N once the matrix is initialized.
    mutable std::vector<double> m_BPerm;

    bool                m_IsInitialized = false;
    bool                m_Verbose = false;  /**< print profile statistics */

    // Storage guard: abort (with a hint to switch to PARDISO/cuDSS) if the
    // projected skyline allocation (AU+AL+AD) exceeds this many GiB. The
    // profile is O(n * bandwidth) -- super-linear in the mesh -- so a large
    // 3D system would otherwise OOM here before the first solve. <= 0
    // disables the guard ("params":{"max_profile_gb":...} in the input).
    double              m_MaxProfileGB = 16.0;

    // Lightweight structural fingerprint to detect pattern changes
    // between solve() calls (rows + nnz only -- O(1) check).
    int                 m_CachedNNZ = -1;
    int                 m_CachedRows = -1;
};
