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
//+++ Function: NVIDIA cuDSS GPU direct-solver backend. Available only
//+++           when PeriX is built with -DUSE_CUDSS=on (gated by the
//+++           PERIX_HAS_CUDSS macro). The CSR sparsity pattern (row
//+++           offsets and column indices) and the symbolic analysis
//+++           are set up once in initSolver and kept resident on the
//+++           device; each solveLinearSystem then uploads only the
//+++           numeric values and the right-hand side, runs the GPU
//+++           factorization + triangular solves, and copies the
//+++           solution back.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <cuda_runtime.h>
#include <cudss.h>

#include <vector>

#include "LinearSolver/LinearSolverBase.h"

class CudssSolver : public LinearSolverBase {
public:
    CudssSolver();
    ~CudssSolver() override;

    /**
     * initialise the GPU solver from the matrix sparsity pattern:
     * allocate the device buffers, upload the (fixed) CSR structure
     * once, build the cuDSS objects and run the symbolic ANALYSIS phase.
     * @param K          sparse matrix (provides size, nnz, CSR pattern)
     * @param Parameters json parameters read from input file
     */
    void initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) final;

    /**
     * solve A x = b on the GPU. Only the numeric values of A and the
     * right-hand side b are transferred to the device (the structure is
     * already resident); the FACTORIZATION + SOLVE phases run on the
     * device and x is copied back to the host.
     * @param A sparse matrix (same pattern as the one passed to initSolver)
     * @param b right hand side vector
     * @param x solution vector
     * @return true on success
     */
    bool solveLinearSystem(SparseMatrix &A, VectorXd &b, VectorXd &x) final;

private:
    void releaseInternalMemory();

    cudaStream_t  m_Stream = nullptr;
    cudssHandle_t m_Handle = nullptr;
    cudssConfig_t m_Config = nullptr;
    cudssData_t   m_Data   = nullptr;
    cudssMatrix_t m_A = nullptr;   /**< CSR system matrix (device) */
    cudssMatrix_t m_X = nullptr;   /**< dense solution (device)    */
    cudssMatrix_t m_B = nullptr;   /**< dense rhs (device)         */

    int m_N   = 0;                 /**< number of equations */
    int m_NNZ = 0;                 /**< number of stored nonzeros */

    // device-side CSR + dense vectors; the structure (offsets, columns)
    // is uploaded once, only m_dValues and m_dB change per solve.
    int    *m_dRowOffsets = nullptr;  /**< (N+1) CSR row pointers */
    int    *m_dColIndices = nullptr;  /**< (NNZ)  CSR column ids  */
    double *m_dValues     = nullptr;  /**< (NNZ)  CSR values      */
    double *m_dB          = nullptr;  /**< (N)    right-hand side */
    double *m_dX          = nullptr;  /**< (N)    solution        */

    bool m_IsInitialized = false;
    bool m_HybridMemory  = false;  /**< spill factor to host memory (small GPUs) */

    // ---- factorization reuse / refactorization ----
    // Mirrors the PARDISO backend: a snapshot of the most recently factorized
    // CSR values. Bit-identical values -> the device factor is still valid, so
    // skip the factorization entirely and run only the triangular SOLVE phase
    // (constant-tangent kernels + every line-search re-solve). Otherwise, once a
    // first full FACTORIZATION exists, later value changes use the cheaper
    // CUDSS_PHASE_REFACTORIZATION (reuses the pivot order/structure); a failed
    // or non-finite refactorized solve falls back to a full FACTORIZATION once.
    std::vector<double> m_LastValues;
    bool m_HaveFactor   = false;
    bool m_ReuseFactor  = true;   /**< params.reuse_factor    (default on) */
    bool m_UseRefactor  = true;   /**< params.refactorization (default on) */
    // The reuse snapshot costs one nnz-sized copy per factorized solve. A
    // constant-tangent workload hits the cache by its second solve; a Jacobian
    // that changes every Newton iteration never hits it, so after this many
    // consecutive misses the cache is switched off (and its buffer freed)
    // instead of taxing every remaining solve. Reset on any hit.
    int m_ReuseMisses   = 0;
    static constexpr int kReuseMissLimit = 8;

    // ---- pinned (page-locked) host staging ----
    // cudaHostRegister the stable host arrays (CSR values, b, x) once so the
    // per-iteration H2D/D2H transfers run at pinned bandwidth. Registered
    // pointers are tracked so a reallocation re-registers and release() cleans up.
    bool  m_PinHost      = true;  /**< params.pin_host (default on) */
    void *m_RegVals = nullptr; size_t m_RegValsBytes = 0;
    void *m_RegB    = nullptr; size_t m_RegBBytes    = 0;
    void *m_RegX    = nullptr; size_t m_RegXBytes    = 0;

    bool m_Timing = false;        /**< PERIX_CUDSS_TIMING=1 or params.timing: per-solve breakdown */

    /** register ptr as pinned memory (tracked; re-registers if the pointer moved) */
    void pinHostBuffer(void *ptr, size_t bytes, void *&reg, size_t &regBytes);
};
