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
//+++ Function: NVIDIA cuDSS GPU direct-solver implementation. The
//+++           fixed CSR structure and the symbolic analysis are set up
//+++           once in initSolver; each solveLinearSystem uploads only
//+++           the values and the right-hand side, runs the device
//+++           factorization + triangular solves, and copies x back.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "LinearSolver/CudssSolver.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "Utils/MessagePrinter.h"

namespace {
    // Fatal CUDA/cuDSS checks: correctness comes first, so any failure
    // aborts rather than silently returning a wrong solution.
    inline void checkCuda(cudaError_t e, const char *what) {
        if (e!=cudaSuccess) {
            MessagePrinter::printErrorTxt(std::string("CudssSolver: CUDA error in ")+what
                                          +": "+cudaGetErrorString(e));
            MessagePrinter::exitPeriX();
        }
    }
    inline void checkCudss(cudssStatus_t s, const char *what) {
        if (s!=CUDSS_STATUS_SUCCESS) {
            MessagePrinter::printErrorTxt(std::string("CudssSolver: cuDSS error in ")+what
                                          +": status="+std::to_string(static_cast<int>(s)));
            MessagePrinter::exitPeriX();
        }
    }
    inline bool getBoolParam(const nlohmann::ordered_json &p,const char *key,bool def) {
        return (p.contains(key)) ? p.at(key).get<bool>() : def;
    }
    inline std::string getStrParam(const nlohmann::ordered_json &p,const char *key,
                                   const std::string &def) {
        return (p.contains(key)) ? p.at(key).get<std::string>() : def;
    }
    inline int getIntParam(const nlohmann::ordered_json &p,const char *key,int def) {
        return (p.contains(key)) ? p.at(key).get<int>() : def;
    }
    inline double getDoubleParam(const nlohmann::ordered_json &p,const char *key,double def) {
        return (p.contains(key)) ? p.at(key).get<double>() : def;
    }
    // map a user string to a cuDSS reordering algorithm; -1 => leave at default
    inline int reorderingFromStr(const std::string &s) {
        if (s=="default") return CUDSS_REORDERING_ALG_DEFAULT;
        if (s=="amd")     return CUDSS_REORDERING_ALG_AMD;
        if (s=="nd"||s=="nested_dissection") return CUDSS_REORDERING_ALG_NESTED_DISSECTION;
        if (s=="colamd")  return CUDSS_REORDERING_ALG_COLAMD;
        if (s=="btf_colamd") return CUDSS_REORDERING_ALG_BTF_COLAMD;
        if (s=="none")    return CUDSS_REORDERING_ALG_NONE;
        return -1;
    }
}

CudssSolver::CudssSolver() = default;

CudssSolver::~CudssSolver() {
    releaseInternalMemory();
}

void CudssSolver::pinHostBuffer(void *ptr, size_t bytes, void *&reg, size_t &regBytes) {
    // Page-lock a stable host staging array so the per-iteration cudaMemcpyAsync
    // runs at pinned bandwidth instead of staging through a pageable bounce
    // buffer. Registration is tracked: same pointer+size is a no-op, a moved or
    // resized buffer is re-registered, and a registration failure only warns
    // (the copy still works, just at pageable speed).
    if (!m_PinHost || ptr==nullptr || bytes==0) return;
    if (reg==ptr && regBytes==bytes) return;
    if (reg) { cudaHostUnregister(reg); reg=nullptr; regBytes=0; }
    const cudaError_t e=cudaHostRegister(ptr,bytes,cudaHostRegisterDefault);
    if (e==cudaSuccess) {
        reg=ptr; regBytes=bytes;
    }
    else {
        cudaGetLastError(); // clear the sticky error; pinning is best-effort
        MessagePrinter::printWarningTxt("CudssSolver: cudaHostRegister failed ("
            +std::string(cudaGetErrorString(e))+"); continuing with pageable transfers");
        m_PinHost=false;    // don't retry every iteration
    }
}

void CudssSolver::releaseInternalMemory() {
    if (m_RegVals) { cudaHostUnregister(m_RegVals); m_RegVals=nullptr; m_RegValsBytes=0; }
    if (m_RegB)    { cudaHostUnregister(m_RegB);    m_RegB=nullptr;    m_RegBBytes=0; }
    if (m_RegX)    { cudaHostUnregister(m_RegX);    m_RegX=nullptr;    m_RegXBytes=0; }
    m_LastValues.clear();
    m_HaveFactor=false;

    if (m_A) { cudssMatrixDestroy(m_A); m_A=nullptr; }
    if (m_B) { cudssMatrixDestroy(m_B); m_B=nullptr; }
    if (m_X) { cudssMatrixDestroy(m_X); m_X=nullptr; }
    if (m_Data   && m_Handle) { cudssDataDestroy(m_Handle,m_Data); m_Data=nullptr; }
    if (m_Config)             { cudssConfigDestroy(m_Config);      m_Config=nullptr; }
    if (m_Handle)             { cudssDestroy(m_Handle);            m_Handle=nullptr; }

    if (m_dRowOffsets) { cudaFree(m_dRowOffsets); m_dRowOffsets=nullptr; }
    if (m_dColIndices) { cudaFree(m_dColIndices); m_dColIndices=nullptr; }
    if (m_dValues)     { cudaFree(m_dValues);     m_dValues=nullptr; }
    if (m_dB)          { cudaFree(m_dB);          m_dB=nullptr; }
    if (m_dX)          { cudaFree(m_dX);          m_dX=nullptr; }
    if (m_Stream)      { cudaStreamDestroy(m_Stream); m_Stream=nullptr; }

    m_N=0; m_NNZ=0;
    m_IsInitialized=false;
}

void CudssSolver::initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) {
    releaseInternalMemory();

    m_N  =K.getSize();
    m_NNZ=K.getNNZNum();
    if (m_N<=0) {
        return; // nothing to set up for an empty system
    }

    // Optional cross-GPU control: hybrid memory mode lets cuDSS spill the
    // factor to host memory, so a large factorisation that would not fit in a
    // smaller GPU's device memory (e.g. an 8 GB RTX 4060) still runs -- at some
    // bandwidth cost. Off by default (best on a large GPU such as the 24 GB
    // RTX 4090); enable with "params":{"hybrid_memory":true} on a small GPU.
    m_HybridMemory = getBoolParam(Parameters,"hybrid_memory",false);

    // Factor-reuse / refactorization / pinned-staging switches (all default ON;
    // see the member docs in CudssSolver.h). Timing is an opt-in per-solve
    // breakdown for performance analysis: params {"timing":true} or the
    // environment variable PERIX_CUDSS_TIMING=1.
    m_ReuseFactor = getBoolParam(Parameters,"reuse_factor",true);
    m_UseRefactor = getBoolParam(Parameters,"refactorization",true);
    m_PinHost     = getBoolParam(Parameters,"pin_host",true);
    {
        const char *e=std::getenv("PERIX_CUDSS_TIMING");
        m_Timing = getBoolParam(Parameters,"timing",false) || (e && e[0]=='1');
    }
    m_LastValues.clear();
    m_HaveFactor=false;
    m_ReuseMisses=0;

    checkCuda(cudaStreamCreate(&m_Stream),"cudaStreamCreate");
    checkCudss(cudssCreate(&m_Handle),"cudssCreate");
    checkCudss(cudssSetStream(m_Handle,m_Stream),"cudssSetStream");
    checkCudss(cudssConfigCreate(&m_Config),"cudssConfigCreate");
    checkCudss(cudssDataCreate(m_Handle,&m_Data),"cudssDataCreate");

    if (m_HybridMemory) {
        const int on=1;
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_HYBRID_MEMORY_MODE,
                                  const_cast<int*>(&on),sizeof(int)),
                   "cudssConfigSet(HYBRID_MEMORY_MODE)");
        MessagePrinter::printNormalTxt("CudssSolver: hybrid memory mode enabled "
                                       "(factor may spill to host memory)");
    }

    // Optional reordering override. The fill-in (hence factorisation cost) of a
    // sparse direct solve is dominated by the fill-reducing ordering; the best
    // choice is matrix-dependent, so it is exposed for tuning. Default leaves
    // cuDSS to pick (nested dissection).
    const std::string reo = getStrParam(Parameters,"reordering","default");
    const int reoAlg = reorderingFromStr(reo);
    if (reoAlg<0) {
        MessagePrinter::printErrorTxt("CudssSolver: unknown reordering '"+reo+"' "
            "(use default|amd|nd|colamd|btf_colamd|none)");
        MessagePrinter::exitPeriX();
    }
    if (reo!="default") {
        cudssReorderingAlg_t a = static_cast<cudssReorderingAlg_t>(reoAlg);
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_REORDERING_ALG,&a,sizeof(a)),
                   "cudssConfigSet(REORDERING_ALG)");
        MessagePrinter::printNormalTxt("CudssSolver: reordering = "+reo);
    }

    // Optional iterative refinement (default OFF -> bare LU factorise+solve).
    // The bare cuDSS solve already matches PARDISO to ~1e-10 on the Cahn-Hilliard
    // c/mu system at every size tested (200x200 .. 768x768, and the full coarsening
    // run), so refinement is NOT needed by default. It is exposed only as a safety
    // lever for a genuinely ill-conditioned system on another GPU/driver:
    // "params":{"ir_steps":2} adds residual (Richardson) refinement that can only
    // sharpen the solve, never harm it.
    //
    // NOTE: cuDSS MATCHING/scaling (CUDSS_CONFIG_MATCHING_ALG) was evaluated as a
    // robustness aid and DELIBERATELY NOT enabled: turning it on permutes/scales the
    // indefinite c/mu matrix into a factorisation that returns a WRONG solve here
    // (the Newton residual freezes at its initial value and the step never
    // converges). The bare ordering is correct, so matching stays off.
    const int irSteps=getIntParam(Parameters,"ir_steps",0);
    if (irSteps>0) {
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_IR_N_STEPS,&irSteps,sizeof(irSteps)),
                   "cudssConfigSet(IR_N_STEPS)");
        MessagePrinter::printNormalTxt("CudssSolver: iterative refinement steps = "
                                       +std::to_string(irSteps));
    }

    // STATIC-PIVOTING PERTURBATION (the direct analogue of PARDISO iparm[10],
    // which is ON by default in PARDISO -- the reason PARDISO does not NaN here).
    // cuDSS's default pivoting for a GENERAL matrix with the default
    // nested-dissection reordering is CUDSS_PIVOT_LOCAL_BLOCK: it pivots only
    // WITHIN a supernode, never globally. On an indefinite saddle-point Jacobian
    // -- e.g. the coupled Cahn-Hilliard + mechanics + fracture system
    // (frac_stress_cahnhilliard, strong Omega coupling, damage, a crack) -- a
    // pivot that needs a GLOBAL exchange goes to (near-)zero, and the bare
    // factorization divides by it and returns NaN while STILL reporting
    // CUDSS_STATUS_SUCCESS (a silent failure). Perturbing any pivot whose
    // magnitude falls below pivot_epsilon (scaled by the matrix, like PARDISO)
    // removes the division-by-zero: the factorization stays finite and the
    // subsequent Newton step is at worst inexact (inexact Newton still converges;
    // the adaptive stepper cuts back if it does not). For a well-conditioned
    // matrix no pivot is that small, so this is a no-op -- the Cahn-Hilliard c/mu
    // solves validated to ~1e-10 vs PARDISO are unchanged. Tunable / disable with
    // "params":{"pivot_epsilon":0}. NOTE: distinct from cuDSS MATCHING (kept off
    // above), which permutes/scales the matrix and gave a WRONG solve here.
    // Optional bitwise run-to-run reproducibility. The default cuDSS numeric
    // factorization is NOT deterministic across identical runs (GPU reduction
    // ordering varies at round-off level); on a well-conditioned system this is
    // invisible, but an ill-conditioned system amplifies
    // it to observable field differences between two identical runs. cuDSS
    // provides a deterministic mode at some performance cost; expose it for
    // reproducibility-sensitive studies: "params":{"deterministic":true}.
    if (getBoolParam(Parameters,"deterministic",false)) {
        const int on=1;
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_DETERMINISTIC_MODE,
                                  const_cast<int*>(&on),sizeof(int)),
                   "cudssConfigSet(DETERMINISTIC_MODE)");
        MessagePrinter::printNormalTxt("CudssSolver: deterministic mode enabled "
                                       "(bitwise-reproducible factorization)");
    }

    const double pivotEps=getDoubleParam(Parameters,"pivot_epsilon",1.0e-13);
    if (pivotEps>0.0) {
        cudssPivotEpsilonAlg_t epsAlg=CUDSS_PIVOT_EPSILON_ALG_SCALED;  // eps relative to matrix scale
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_PIVOT_EPSILON_ALG,&epsAlg,sizeof(epsAlg)),
                   "cudssConfigSet(PIVOT_EPSILON_ALG)");
        checkCudss(cudssConfigSet(m_Config,CUDSS_CONFIG_PIVOT_EPSILON,
                                  const_cast<double*>(&pivotEps),sizeof(pivotEps)),
                   "cudssConfigSet(PIVOT_EPSILON)");
    }

    // device buffers
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_dRowOffsets),
                         static_cast<size_t>(m_N+1)*sizeof(int)),"cudaMalloc(rowOffsets)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_dColIndices),
                         static_cast<size_t>(m_NNZ)*sizeof(int)),"cudaMalloc(colIndices)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_dValues),
                         static_cast<size_t>(m_NNZ)*sizeof(double)),"cudaMalloc(values)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_dB),
                         static_cast<size_t>(m_N)*sizeof(double)),"cudaMalloc(b)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_dX),
                         static_cast<size_t>(m_N)*sizeof(double)),"cudaMalloc(x)");

    // Upload the CSR structure ONCE -- it never changes across Newton
    // iterations. The values are uploaded too so the matrix object is
    // valid for the (symbolic) analysis phase; the real per-iteration
    // values arrive in solveLinearSystem.
    checkCuda(cudaMemcpyAsync(m_dRowOffsets,K.getCSRRowsIndexPtr(),
                              static_cast<size_t>(m_N+1)*sizeof(int),
                              cudaMemcpyHostToDevice,m_Stream),"H2D(rowOffsets)");
    checkCuda(cudaMemcpyAsync(m_dColIndices,K.getCSRColsIndexPtr(),
                              static_cast<size_t>(m_NNZ)*sizeof(int),
                              cudaMemcpyHostToDevice,m_Stream),"H2D(colIndices)");
    checkCuda(cudaMemcpyAsync(m_dValues,K.getCSRValuesPtr(),
                              static_cast<size_t>(m_NNZ)*sizeof(double),
                              cudaMemcpyHostToDevice,m_Stream),"H2D(values)");

    // PeriX stores a general (unsymmetric), zero-based CSR matrix. The cuDSS
    // (>=0.5) CSR constructor takes the row-offset, column-index and value
    // types separately as cudssDataType_t (CUDSS_R_32I for the int32 offsets
    // and column ids, CUDSS_R_64F for the double values).
    checkCudss(cudssMatrixCreateCsr(&m_A,m_N,m_N,m_NNZ,
                                    m_dRowOffsets,nullptr,m_dColIndices,m_dValues,
                                    CUDSS_R_32I,CUDSS_R_32I,CUDSS_R_64F,
                                    CUDSS_MTYPE_GENERAL,CUDSS_MVIEW_FULL,CUDSS_BASE_ZERO),
               "cudssMatrixCreateCsr");
    checkCudss(cudssMatrixCreateDn(&m_B,m_N,1,m_N,m_dB,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(b)");
    checkCudss(cudssMatrixCreateDn(&m_X,m_N,1,m_N,m_dX,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR),
               "cudssMatrixCreateDn(x)");

    // Symbolic analysis (reordering + symbolic factorization): depends on
    // the structure only, so it is done once here and reused every solve.
    checkCudss(cudssExecute(m_Handle,CUDSS_PHASE_ANALYSIS,m_Config,m_Data,m_A,m_X,m_B),
               "cudssExecute(ANALYSIS)");
    checkCuda(cudaStreamSynchronize(m_Stream),"cudaStreamSynchronize(analysis)");

    // Pin the host CSR values array now (its allocation is fixed for the run;
    // the assembler only rewrites it in place). The per-iteration nnz-sized
    // H2D upload -- and the CUDA assembler's D2H into this same buffer -- then
    // run at page-locked bandwidth. b/x are pinned lazily at the first solve.
    pinHostBuffer(K.getCSRValuesPtr(),
                  static_cast<size_t>(m_NNZ)*sizeof(double),m_RegVals,m_RegValsBytes);

    m_IsInitialized=true;
}

bool CudssSolver::solveLinearSystem(SparseMatrix &A, VectorXd &b, VectorXd &x) {
    if (!m_IsInitialized) {
        MessagePrinter::printErrorTxt("CudssSolver::solveLinearSystem: solver is not initialised");
        MessagePrinter::exitPeriX();
    }
    if (A.getSize()!=m_N || A.getNNZNum()!=m_NNZ) {
        MessagePrinter::printErrorTxt("CudssSolver::solveLinearSystem: matrix size/nnz changed after init "
                                      "("+std::to_string(A.getSize())+"/"+std::to_string(A.getNNZNum())
                                      +" vs "+std::to_string(m_N)+"/"+std::to_string(m_NNZ)+")");
        MessagePrinter::exitPeriX();
    }
    if (b.getSize()!=m_N) {
        MessagePrinter::printErrorTxt("CudssSolver::solveLinearSystem: rhs size "
                                      +std::to_string(b.getSize())+" != "+std::to_string(m_N));
        MessagePrinter::exitPeriX();
    }
    if (x.getSize()!=m_N) {
        x.resize(m_N);
    }

    // Pin the (stable) host b/x arrays lazily -- after the possible x.resize
    // above so the registered pointer is the one the copies actually use.
    pinHostBuffer(b.getDataPtr(),static_cast<size_t>(m_N)*sizeof(double),m_RegB,m_RegBBytes);
    pinHostBuffer(x.getDataPtr(),static_cast<size_t>(m_N)*sizeof(double),m_RegX,m_RegXBytes);

    // ---- optional per-solve timing breakdown (PERIX_CUDSS_TIMING=1) ----
    // tick() synchronizes the stream first so each interval attributes the GPU
    // work to the right bucket; without timing it is a plain (cheap) clock read
    // and the stream stays fully asynchronous as before.
    using clk=std::chrono::steady_clock;
    auto tick=[&]()->clk::time_point {
        if (m_Timing) cudaStreamSynchronize(m_Stream);
        return clk::now();
    };
    auto ms=[](const clk::time_point &a,const clk::time_point &b)->double {
        return std::chrono::duration<double,std::milli>(b-a).count();
    };
    const auto t0=tick();

    // ---- transparent factorization reuse (same contract as the PARDISO
    // backend): bit-identical values -> the resident device factor is still
    // valid, skip factorization AND the nnz-sized H2D upload, run only the
    // triangular solves. ----
    double *vals=A.getCSRValuesPtr();
    const bool reuse = m_ReuseFactor && m_HaveFactor
                    && static_cast<int>(m_LastValues.size())==m_NNZ
                    && std::memcmp(m_LastValues.data(),vals,
                                   static_cast<std::size_t>(m_NNZ)*sizeof(double))==0;
    const auto t_cmp=tick();

    if (!reuse) {
        checkCuda(cudaMemcpyAsync(m_dValues,vals,
                                  static_cast<size_t>(m_NNZ)*sizeof(double),
                                  cudaMemcpyHostToDevice,m_Stream),"H2D(values)");
    }
    checkCuda(cudaMemcpyAsync(m_dB,b.getDataPtr(),
                              static_cast<size_t>(m_N)*sizeof(double),
                              cudaMemcpyHostToDevice,m_Stream),"H2D(b)");
    const auto t_h2d=tick();

    // One factorization-kind phase on the GPU. Unlike the one-off setup in
    // initSolver, a FAILURE here is RECOVERABLE (return false -> the Newton
    // driver cuts dt back), so a non-success status warns instead of aborting.
    auto execPhase=[&](const cudssPhase_t ph,const char *what)->bool {
        const cudssStatus_t s=cudssExecute(m_Handle,ph,m_Config,m_Data,m_A,m_X,m_B);
        if (s!=CUDSS_STATUS_SUCCESS) {
            MessagePrinter::printWarningTxt(std::string("CudssSolver: ")+what
                +" failed (status="+std::to_string(static_cast<int>(s))+")");
            return false;
        }
        return true;
    };
    // Triangular solves + copy-back + NaN/Inf GUARD. cuDSS can report SUCCESS yet
    // produce a non-finite solution when a zero pivot slips through (see the
    // pivot-epsilon note in initSolver); detect that here so the caller can
    // retry with a full factorization or cut the timestep back -- the plain
    // Newton path does not otherwise inspect the update.
    auto solveAndCheck=[&]()->bool {
        if (!execPhase(CUDSS_PHASE_SOLVE,"triangular solve")) return false;
        checkCuda(cudaMemcpyAsync(x.getDataPtr(),m_dX,
                                  static_cast<size_t>(m_N)*sizeof(double),
                                  cudaMemcpyDeviceToHost,m_Stream),"D2H(x)");
        checkCuda(cudaStreamSynchronize(m_Stream),"cudaStreamSynchronize(solve)");
        const double *xp=x.getDataPtr();
        for (int i=0;i<m_N;++i) {
            if (!std::isfinite(xp[i])) return false;
        }
        return true;
    };

    bool ok=false;
    const char *mode="reuse";
    auto t_fact=t_h2d;
    if (reuse) {
        ok=solveAndCheck();
        t_fact=t_h2d;
        if (!ok) {
            // A factor built from bit-identical values should never fail its
            // back-substitution; treat this as a stale/corrupt factor and
            // rebuild once from scratch (the values are already resident).
            MessagePrinter::printWarningTxt("CudssSolver: reused-factor solve failed; "
                "rebuilding the factorization");
            mode="reuse->full";
            ok=execPhase(CUDSS_PHASE_FACTORIZATION,"factorization") && solveAndCheck();
        }
    }
    else {
        const bool tryRefactor=(m_HaveFactor && m_UseRefactor);
        mode=tryRefactor?"refactor":"full";
        ok=execPhase(tryRefactor?CUDSS_PHASE_REFACTORIZATION:CUDSS_PHASE_FACTORIZATION,
                     tryRefactor?"refactorization":"factorization");
        t_fact=tick();
        if (ok) ok=solveAndCheck();
        if (!ok && tryRefactor) {
            // REFACTORIZATION reuses the pivot order of the first factorization;
            // when the values have drifted far enough that this order breaks
            // down (failed phase or non-finite x), redo ONE full factorization
            // (fresh pivoting) before reporting failure to the Newton driver.
            MessagePrinter::printWarningTxt("CudssSolver: refactorized solve failed or was "
                "non-finite; retrying with a full factorization");
            mode="refactor->full";
            ok=execPhase(CUDSS_PHASE_FACTORIZATION,"factorization") && solveAndCheck();
        }
    }
    const auto t_end=tick();

    if (!ok) {
        // No valid factor is guaranteed to be resident after a failed attempt;
        // drop the reuse cache so the next call rebuilds from scratch.
        m_HaveFactor=false;
        m_LastValues.clear();
        int info=0; size_t written=0;
        cudssDataGet(m_Handle,m_Data,CUDSS_DATA_INFO,&info,sizeof(info),&written);
        MessagePrinter::printWarningTxt("CudssSolver: solve failed or produced non-finite "
            "entries (cuDSS INFO="+std::to_string(info)+", likely a singular/zero pivot the "
            "static-pivoting perturbation could not save); signalling a linear-solver "
            "failure so the nonlinear driver cuts back. If this persists, try a smaller "
            "dt, or \"params\":{\"pivot_epsilon\":1e-10} / \"reordering\":\"colamd\".");
        return false;
    }

    if (m_ReuseFactor) {
        if (reuse) {
            m_ReuseMisses=0;
        }
        else if (m_HaveFactor && ++m_ReuseMisses>=kReuseMissLimit) {
            // The Jacobian has changed on every one of the last kReuseMissLimit
            // solves: this is a changing-tangent workload that will never hit
            // the cache, so stop paying the nnz-sized snapshot copy for it.
            m_ReuseFactor=false;
            std::vector<double>().swap(m_LastValues);
            MessagePrinter::printNormalTxt("      CudssSolver: factor-reuse cache retired "
                "(Jacobian changed on "+std::to_string(kReuseMissLimit)+" consecutive solves)");
        }
        if (m_ReuseFactor && !reuse) {
            m_LastValues.assign(vals,vals+m_NNZ);
        }
    }
    m_HaveFactor=true;

    if (m_Timing) {
        std::printf("[cudss-timing] mode=%-13s memcmp=%7.2fms h2d=%7.2fms fact=%7.2fms "
                    "solve+d2h=%7.2fms total=%7.2fms n=%d nnz=%d\n",
                    mode,ms(t0,t_cmp),ms(t_cmp,t_h2d),ms(t_h2d,t_fact),ms(t_fact,t_end),
                    ms(t0,t_end),m_N,m_NNZ);
        std::fflush(stdout);
    }
    return true;
}
