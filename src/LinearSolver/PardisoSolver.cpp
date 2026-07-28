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

#include "LinearSolver/PardisoSolver.h"

#include <cstring>

#include "Utils/MessagePrinter.h"

namespace {
    MKL_INT getIntParameter(const nlohmann::ordered_json &params,
                            const char *key,
                            const MKL_INT defaultValue) {
        if (!params.contains(key)) {
            return defaultValue;
        }
        return static_cast<MKL_INT>(params.at(key).get<int>());
    }

    bool getBoolParameter(const nlohmann::ordered_json &params,
                          const char *key,
                          const bool defaultValue) {
        if (!params.contains(key)) {
            return defaultValue;
        }
        return params.at(key).get<bool>();
    }
} // namespace

PardisoSolver::PardisoSolver() {
    mtype=11;             /* Real and nonsymmetric matrix */
    nrhs=1;               /* Number of right hand sides. */
    for(int i=0;i<64;i++){
        iparm[i]=0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = 2;         /* Fill-in reordering from METIS */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Conjugate transposed/transpose solve */
    iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    iparm[34] = 1;        /* Zero-based indexing: columns and rows indexing in arrays ia, ja, and perm starts from 0 (C-style indexing). */
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;             /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information  */
    error = 0;            /* Initialize error flag */
    ddum = 0.0;
    idum = 0;

    /* -------------------------------------------------------------------- */
    /* .. Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    /* -------------------------------------------------------------------- */
    for (int i = 0; i < 64; i++ ){
        pt[i] = 0;
    }
}

PardisoSolver::~PardisoSolver() {
    releaseInternalMemory();
}

void PardisoSolver::releaseInternalMemory() {
    if (!m_IsInitialized) {
        return;
    }

    phase = -1;
    error = 0;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
            &N,
            &ddum,
            &idum,
            &idum,
            &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    for (int i = 0; i < 64; ++i) {
        pt[i] = nullptr;
    }
    m_IsInitialized=false;
    m_HaveFactor=false;
    m_LastValues.clear();
}

void PardisoSolver::initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) {
    if (Parameters.max_size()){}
    releaseInternalMemory();

    mtype=11;             /* Real and nonsymmetric matrix */
    nrhs=1;               /* Number of right hand sides. */
    for(int i=0;i<64;i++){
        iparm[i]=0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = getIntParameter(Parameters, "iparm1", 2);/* Fill-in reordering from METIS */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = getIntParameter(Parameters, "iparm9", 13);/* Perturb the pivot elements with 1E-13 */
    iparm[10] = getBoolParameter(Parameters, "iparm10", false) ? 1 : 0;/* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Conjugate transposed/transpose solve */
    iparm[12] = getBoolParameter(Parameters, "iparm12", false) ? 1 : 0;/* Maximum weighted matching algorithm */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = getBoolParameter(Parameters, "iparm17", false) ? -1 : 0;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = getBoolParameter(Parameters, "iparm18", false) ? -1 : 0;;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    iparm[23] = getIntParameter(Parameters, "iparm23", 0);/* two-level factorization algorithm */
    iparm[34] = 1;        /* Zero-based indexing: columns and rows indexing in arrays ia, ja, and perm starts from 0 (C-style indexing). */
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;             /* Which factorization to use. */
    msglvl = getIntParameter(Parameters, "msglvl", 0);           /* Print statistical information  */
    error = 0;            /* Initialize error flag */
    ddum = 0.0;
    idum = 0;

    if (iparm[23] != 0 && (iparm[10] != 0 || iparm[12] != 0)) {
        MessagePrinter::printWarningTxt("      iparm[23] is disabled because it is not effective together with iparm[10] or iparm[12]");
        iparm[23] = 0;
    }

    // Factorization phase for the per-iteration solves. The symbolic analysis
    // (reordering + symbolic factorization, phase 11) is done ONCE below on the
    // fixed CSR pattern, so the canonical MKL workflow re-runs only the NUMERIC
    // factorization afterwards: phase 23 (factorize+solve). Phase 13 would
    // repeat the whole METIS reordering + symbolic pass on every Newton
    // iteration -- pure overhead, since with the default iparm[10]=iparm[12]=0
    // the analysis is value-independent and its result never changes. When the
    // user DOES enable value-dependent scaling/matching (iparm10/iparm12), the
    // permutation depends on the numeric values, so re-analysis per solve is
    // kept for them (and available explicitly via "full_reanalysis":true).
    m_FactorPhase = 23;
    if (iparm[10] != 0 || iparm[12] != 0 || getBoolParameter(Parameters,"full_reanalysis",false)) {
        m_FactorPhase = 13;
    }

    /* -------------------------------------------------------------------- */
    /* .. Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    /* -------------------------------------------------------------------- */
    for (int i = 0; i < 64; i++ ){
        pt[i] = nullptr;
    }

    /* -------------------------------------------------------------------- */
    /* .. Reordering and Symbolic Factorization. This step also allocates */
    /* all memory that is necessary for the factorization. */
    /* -------------------------------------------------------------------- */
    phase = 11;
    N=K.getSize();
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &N,
             K.getCSRValuesPtr(),
             K.getCSRRowsIndexPtr(),
             K.getCSRColsIndexPtr(),
             nullptr, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ){
        MessagePrinter::printErrorTxt("error="+to_string(error)+" occurs during symbolic factorization of your Pardiso solver");
        MessagePrinter::exitPeriX();
    }
    m_IsInitialized=true;
    MessagePrinter::printNormalTxt("      Pardiso analysis and symbolic factorization completed");
    MessagePrinter::printNormalTxt("      Number of nonzeros in factors = "+to_string(iparm[17]));
    MessagePrinter::printNormalTxt("      Number of factorization MFLOPS = "+to_string(iparm[18]));
}

bool PardisoSolver::solveLinearSystem(SparseMatrix &A, VectorXd &b, VectorXd &x) {
    if (!m_IsInitialized) {
        MessagePrinter::printErrorTxt("pardiso solver is not initialized before solve");
        return false;
    }

    // ---- transparent factorization reuse ----
    // If the matrix values are bit-identical to the last factorized system, the
    // existing numerical factorization is still valid, so skip straight to the
    // solve phase (33). Otherwise re-factorize and solve (13) and snapshot the
    // new values. The CSR sparsity pattern is fixed for the whole run (built
    // once in SparseMatrix::initFromPDMesh; broken bonds only zero entries, BCs
    // touch existing slots), so an exact value match is a sufficient and
    // conservative reuse test. This collapses the per-step Newton iterations of
    // a constant-tangent kernel to one factorization + cheap back-substitutions,
    // and leaves every changing-Jacobian kernel at its original behaviour.
    const int nnz=A.getNNZNum();
    const double *vals=A.getCSRValuesPtr();
    const bool reuse = m_HaveFactor
                    && static_cast<int>(m_LastValues.size())==nnz
                    && std::memcmp(m_LastValues.data(),vals,
                                   static_cast<std::size_t>(nnz)*sizeof(double))==0;

    phase = reuse ? 33 : m_FactorPhase;   // 33: solve only; 23: refactorize + solve
                                          // (13 = full re-analysis, only when scaling/matching is on)
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &N,
             A.getCSRValuesPtr(),
             A.getCSRRowsIndexPtr(),
             A.getCSRColsIndexPtr(),
             &idum, &nrhs, iparm, &msglvl,
             b.getDataPtr(),
             x.getDataPtr(),
             &error);
    if ( error != 0 ){
        // A failed (re)factorization has already overwritten the numeric factor
        // held in pt[] (PARDISO reuses the same internal handle), so the cached
        // snapshot no longer corresponds to any valid factor. Invalidate the
        // reuse cache, or a later BIT-IDENTICAL assembly -- e.g. the same
        // quasi-static tangent re-assembled after a timestep cutback restores U
        // -- would take the reuse path and back-substitute (phase 33) on the
        // dead factor, returning garbage instead of triggering a fresh
        // factorization of the recovered system.
        m_HaveFactor=false;
        m_LastValues.clear();
        MessagePrinter::printErrorTxt("error="+to_string(error)+" during solution phase of Pardiso solver");
        return false;
    }
    if (!reuse) {
        m_LastValues.assign(vals,vals+nnz);
        m_HaveFactor=true;
    }
    return true;
}
