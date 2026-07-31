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
//+++ Function: Minimal public AMGCL backend. Uses BiCGSTAB and
//+++           escalates the preconditioner automatically
//+++           (smoothed-aggregation AMG/ILU(0) -> ILU(0) -> ILU(k)),
//+++           detects node blocks, reuses the preconditioner, and
//+++           accepts a solve only after checking the true residual.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "LinearSolver/AmgclSolver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>   // omp_set_num_threads for the scoped AMGCL solver team
#endif

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/solver/bicgstab.hpp>

#include "Utils/MessagePrinter.h"

namespace {

/** Preconditioner rungs of the automatic fallback ladder, from cheapest to
 *  strongest. ILU(0) is provably too weak for the dense, non-symmetric,
 *  non-M-matrix PDDO stencil (its incomplete factor can even be exactly
 *  singular there), so a level-of-fill ILU is kept in reserve. */
enum class PrecondKind { AmgIlu0, Ilu0, Iluk };

struct Options {
    double Tol=1.0e-10;
    int MaxIters=200;
    bool Verbose=false;
    int IluLevel=2;
    int Threads=0;
};

/** AMGCL's builtin backend splits every sparse kernel into a fresh OpenMP
 *  region, so a Krylov solve issues thousands of barriers over comparatively
 *  little work per row. On a many-core node the barrier latency then dwarfs the
 *  arithmetic: on a 96-core machine the 200x200 Poisson deck takes 2.0 s on 4
 *  threads but 172 s on the default 96, and forcing OMP_WAIT_POLICY=passive
 *  only trades the spin for wake-up latency (115 s). Sizing the team by the
 *  work actually available keeps the backend on the flat part of that curve
 *  (643k rows: 40.6 s on 8 threads vs 104.6 s serial). */
constexpr int kMinRowsPerSolverThread=20000;
constexpr int kMaxSolverThreads=8;

/** Applies the AMGCL thread team for one solve and puts the ambient OpenMP
 *  width back afterwards. PeriX otherwise never calls omp_set_num_threads --
 *  the assembly loops must keep running on the user's OMP_NUM_THREADS team --
 *  so the override has to be strictly scoped to the linear solve. */
class SolverThreadTeam {
public:
    SolverThreadTeam(const std::ptrdiff_t rows,const int requested) {
#ifdef _OPENMP
        m_Ambient=omp_get_max_threads();
        if (m_Ambient<=1) return;

        int team=requested;
        if (team<=0) {
            const std::ptrdiff_t byWork=
                rows/static_cast<std::ptrdiff_t>(kMinRowsPerSolverThread);
            team=static_cast<int>(
                std::min<std::ptrdiff_t>(byWork,kMaxSolverThreads));
            if (team<1) team=1;
        }
        team=std::min(team,m_Ambient);
        if (team==m_Ambient) return;

        omp_set_num_threads(team);
        m_Restore=true;
#else
        (void)rows;
        (void)requested;
#endif
    }

    ~SolverThreadTeam() {
#ifdef _OPENMP
        if (m_Restore) omp_set_num_threads(m_Ambient);
#endif
    }

    SolverThreadTeam(const SolverThreadTeam &)=delete;
    SolverThreadTeam &operator=(const SolverThreadTeam &)=delete;

private:
    int m_Ambient=1;
    bool m_Restore=false;
};

template <int B>
struct BlockTraits {
    using MatrixValue=amgcl::static_matrix<double,B,B>;
    using Backend=amgcl::backend::builtin<MatrixValue>;
};

template <>
struct BlockTraits<1> {
    using MatrixValue=double;
    using Backend=amgcl::backend::builtin<double>;
};

class RunnerBase {
public:
    virtual ~RunnerBase()=default;

    virtual void setup(
        const std::ptrdiff_t &n,
        std::vector<std::ptrdiff_t> &rows,
        std::vector<std::ptrdiff_t> &cols,
        std::vector<double> &values,
        const Options &options)=0;

    virtual void solve(
        const std::ptrdiff_t &n,
        std::vector<std::ptrdiff_t> &rows,
        std::vector<std::ptrdiff_t> &cols,
        std::vector<double> &values,
        const std::vector<double> &rhs,
        std::vector<double> &solution,
        std::size_t &iterations,
        double &reportedResidual)=0;
};

template <int B,PrecondKind P>
class Runner final : public RunnerBase {
public:
    using MatrixValue=typename BlockTraits<B>::MatrixValue;
    using Backend=typename BlockTraits<B>::Backend;
    using Preconditioner=std::conditional_t<
        P==PrecondKind::AmgIlu0,
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0>,
        std::conditional_t<
            P==PrecondKind::Ilu0,
            amgcl::relaxation::as_preconditioner<
                Backend,
                amgcl::relaxation::ilu0>,
            amgcl::relaxation::as_preconditioner<
                Backend,
                amgcl::relaxation::iluk>>>;
    using IterativeSolver=amgcl::solver::bicgstab<Backend>;
    using Solver=amgcl::make_solver<Preconditioner,IterativeSolver>;

    void setup(
        const std::ptrdiff_t &n,
        std::vector<std::ptrdiff_t> &rows,
        std::vector<std::ptrdiff_t> &cols,
        std::vector<double> &values,
        const Options &options) final {
        typename Solver::params parameters;
        parameters.solver.tol=options.Tol;
        parameters.solver.maxiter=static_cast<std::size_t>(options.MaxIters);
        if constexpr (P==PrecondKind::Iluk) {
            parameters.precond.k=options.IluLevel;
        }

        auto matrix=std::tie(n,rows,cols,values);
        if constexpr (B==1) {
            m_Solver=std::make_unique<Solver>(matrix,parameters);
        }
        else {
            m_Solver=std::make_unique<Solver>(
                amgcl::adapter::block_matrix<MatrixValue>(matrix),parameters);
        }
    }

    void solve(
        const std::ptrdiff_t &n,
        std::vector<std::ptrdiff_t> &rows,
        std::vector<std::ptrdiff_t> &cols,
        std::vector<double> &values,
        const std::vector<double> &rhs,
        std::vector<double> &solution,
        std::size_t &iterations,
        double &reportedResidual) final {
        auto matrix=std::tie(n,rows,cols,values);
        if constexpr (B==1) {
            std::tie(iterations,reportedResidual)=
                (*m_Solver)(matrix,rhs,solution);
        }
        else {
            auto blockMatrix=amgcl::adapter::block_matrix<MatrixValue>(matrix);
            auto blockRhs=amgcl::backend::reinterpret_as_rhs<MatrixValue>(rhs);
            auto blockSolution=
                amgcl::backend::reinterpret_as_rhs<MatrixValue>(solution);
            std::tie(iterations,reportedResidual)=
                (*m_Solver)(blockMatrix,blockRhs,blockSolution);
        }
    }

private:
    std::unique_ptr<Solver> m_Solver;
};

template <int B>
std::unique_ptr<RunnerBase> makeRunnerForBlock(const PrecondKind precond) {
    switch (precond) {
        case PrecondKind::AmgIlu0:
            return std::make_unique<Runner<B,PrecondKind::AmgIlu0>>();
        case PrecondKind::Ilu0:
            return std::make_unique<Runner<B,PrecondKind::Ilu0>>();
        case PrecondKind::Iluk:
            return std::make_unique<Runner<B,PrecondKind::Iluk>>();
    }
    return nullptr;
}

std::unique_ptr<RunnerBase> makeRunner(
    const int blockSize,const PrecondKind precond) {
    switch (blockSize) {
        case 1: return makeRunnerForBlock<1>(precond);
        case 2: return makeRunnerForBlock<2>(precond);
        case 3: return makeRunnerForBlock<3>(precond);
        case 4: return makeRunnerForBlock<4>(precond);
        case 5: return makeRunnerForBlock<5>(precond);
        default: return nullptr;
    }
}

} // namespace

struct AmgclSolver::Impl {
    Options SolverOptions;
    std::unique_ptr<RunnerBase> ActiveRunner;

    std::ptrdiff_t N=0;
    std::vector<std::ptrdiff_t> Rows;
    std::vector<std::ptrdiff_t> Cols;
    std::vector<double> Values;
    std::vector<double> Rhs;
    std::vector<double> Solution;

    int BlockSize=1;
    bool HavePreconditioner=false;
    PrecondKind Precond=PrecondKind::AmgIlu0;
};

AmgclSolver::AmgclSolver()=default;

AmgclSolver::~AmgclSolver() {
    releaseInternalMemory();
}

void AmgclSolver::releaseInternalMemory() {
    m_Impl.reset();
    m_IsInitialized=false;
    m_N=0;
    m_NNZ=0;
}

int AmgclSolver::detectBlockSize(SparseMatrix &K) const {
    const int n=K.getSize();
    const int *rows=K.getCSRRowsIndexPtr();
    const int *cols=K.getCSRColsIndexPtr();
    if (n<=0 || rows==nullptr || cols==nullptr) return 1;

    for (int blockSize=kMaxBlockSize;blockSize>=2;--blockSize) {
        if (n%blockSize!=0) continue;

        bool matches=true;
        for (int firstRow=0;firstRow<n && matches;firstRow+=blockSize) {
            const int rowLength=rows[firstRow+1]-rows[firstRow];
            if (rowLength<=0 || rowLength%blockSize!=0) {
                matches=false;
                break;
            }

            for (int offset=0;offset<rowLength;offset+=blockSize) {
                const int firstColumn=cols[rows[firstRow]+offset];
                if (firstColumn%blockSize!=0) {
                    matches=false;
                    break;
                }
                for (int component=1;component<blockSize;++component) {
                    if (cols[rows[firstRow]+offset+component]
                        !=firstColumn+component) {
                        matches=false;
                        break;
                    }
                }
            }

            for (int component=1;component<blockSize && matches;++component) {
                const int row=firstRow+component;
                if (rows[row+1]-rows[row]!=rowLength) {
                    matches=false;
                    break;
                }
                for (int offset=0;offset<rowLength;++offset) {
                    if (cols[rows[row]+offset]
                        !=cols[rows[firstRow]+offset]) {
                        matches=false;
                        break;
                    }
                }
            }
        }

        if (matches) return blockSize;
    }
    return 1;
}

void AmgclSolver::initSolver(
    SparseMatrix &K,nlohmann::ordered_json &Parameters) {
    releaseInternalMemory();

    m_N=K.getSize();
    m_NNZ=K.getNNZNum();
    if (m_N<=0) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver cannot initialize an empty linear system");
        MessagePrinter::exitPeriX();
    }

    m_Impl=std::make_unique<Impl>();
    auto &implementation=*m_Impl;
    implementation.SolverOptions.Tol=
        Parameters.value("tol",1.0e-10);
    implementation.SolverOptions.MaxIters=
        Parameters.value("maxiter",200);
    implementation.SolverOptions.Verbose=
        Parameters.value("verbose",false);
    implementation.SolverOptions.IluLevel=
        Parameters.value("iluk_level",2);
    implementation.SolverOptions.Threads=
        Parameters.value("solver_threads",0);

    if (!(implementation.SolverOptions.Tol>0.0)
        || !std::isfinite(implementation.SolverOptions.Tol)) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver parameter 'tol' must be a positive finite number");
        MessagePrinter::exitPeriX();
    }
    if (implementation.SolverOptions.MaxIters<1) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver parameter 'maxiter' must be at least one");
        MessagePrinter::exitPeriX();
    }
    if (implementation.SolverOptions.IluLevel<1
        || implementation.SolverOptions.IluLevel>6) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver parameter 'iluk_level' must be between 1 and 6");
        MessagePrinter::exitPeriX();
    }
    if (implementation.SolverOptions.Threads<0) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver parameter 'solver_threads' must be zero (auto) or positive");
        MessagePrinter::exitPeriX();
    }

    implementation.N=m_N;
    const int *rows=K.getCSRRowsIndexPtr();
    const int *cols=K.getCSRColsIndexPtr();
    implementation.Rows.assign(rows,rows+m_N+1);
    implementation.Cols.assign(cols,cols+m_NNZ);
    implementation.Values.resize(static_cast<std::size_t>(m_NNZ));
    implementation.Rhs.resize(static_cast<std::size_t>(m_N));
    implementation.Solution.resize(static_cast<std::size_t>(m_N));

    implementation.BlockSize=detectBlockSize(K);
    implementation.Precond=PrecondKind::AmgIlu0;
    implementation.ActiveRunner=
        makeRunner(implementation.BlockSize,implementation.Precond);
    if (!implementation.ActiveRunner) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver failed to create its fixed public solver configuration");
        MessagePrinter::exitPeriX();
    }

    m_IsInitialized=true;
    MessagePrinter::printNormalTxt(
        "      AMGCL: BiCGSTAB + smoothed-aggregation AMG/ILU(0), block size "
        +std::to_string(implementation.BlockSize));
}

bool AmgclSolver::solveLinearSystem(
    SparseMatrix &A,VectorXd &b,VectorXd &x) {
    if (!m_IsInitialized || !m_Impl) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver::solveLinearSystem called before initSolver");
        MessagePrinter::exitPeriX();
    }
    if (A.getSize()!=m_N || A.getNNZNum()!=m_NNZ) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver requires the CSR pattern to remain fixed after initialization");
        MessagePrinter::exitPeriX();
    }
    if (b.getSize()!=m_N) {
        MessagePrinter::printErrorTxt(
            "AmgclSolver right-hand-side size does not match the matrix");
        MessagePrinter::exitPeriX();
    }
    if (x.getSize()!=m_N) x.resize(m_N);

    auto &implementation=*m_Impl;
    const SolverThreadTeam solverThreads(
        m_N,implementation.SolverOptions.Threads);
    const double *matrixValues=A.getCSRValuesPtr();
    const double *rhsValues=b.getDataPtr();
    double *solutionValues=x.getDataPtr();

    std::copy(
        matrixValues,matrixValues+m_NNZ,implementation.Values.begin());
    std::copy(
        rhsValues,rhsValues+m_N,implementation.Rhs.begin());

    double rhsNormSquared=0.0;
#pragma omp parallel for reduction(+:rhsNormSquared)
    for (int row=0;row<m_N;++row) {
        rhsNormSquared+=rhsValues[row]*rhsValues[row];
    }
    const double rhsNorm=std::sqrt(rhsNormSquared);
    if (rhsNorm==0.0) {
        std::fill(solutionValues,solutionValues+m_N,0.0);
        return true;
    }

    auto trueRelativeResidual=[&]() {
        double residualNormSquared=0.0;
#pragma omp parallel for reduction(+:residualNormSquared)
        for (int row=0;row<m_N;++row) {
            double product=0.0;
            for (std::ptrdiff_t entry=implementation.Rows[row];
                 entry<implementation.Rows[row+1];++entry) {
                product+=matrixValues[entry]
                    *solutionValues[implementation.Cols[entry]];
            }
            const double residual=rhsValues[row]-product;
            residualNormSquared+=residual*residual;
        }
        return std::sqrt(residualNormSquared)/rhsNorm;
    };

    auto runOnce=[&](const bool rebuild) {
        try {
            if (rebuild || !implementation.HavePreconditioner) {
                implementation.ActiveRunner->setup(
                    implementation.N,
                    implementation.Rows,
                    implementation.Cols,
                    implementation.Values,
                    implementation.SolverOptions);
                implementation.HavePreconditioner=true;
            }

            std::fill(
                implementation.Solution.begin(),
                implementation.Solution.end(),0.0);
            std::size_t iterations=0;
            double reportedResidual=0.0;
            implementation.ActiveRunner->solve(
                implementation.N,
                implementation.Rows,
                implementation.Cols,
                implementation.Values,
                implementation.Rhs,
                implementation.Solution,
                iterations,
                reportedResidual);

            std::copy(
                implementation.Solution.begin(),
                implementation.Solution.end(),solutionValues);
            for (int row=0;row<m_N;++row) {
                if (!std::isfinite(solutionValues[row])) {
                    return std::numeric_limits<double>::infinity();
                }
            }

            const double trueResidual=trueRelativeResidual();
            if (implementation.SolverOptions.Verbose) {
                char message[192];
                std::snprintf(
                    message,sizeof(message),
                    "      AMGCL: iterations=%zu, reported residual=%9.3e, "
                    "true residual=%9.3e%s",
                    iterations,reportedResidual,trueResidual,
                    rebuild ? " (preconditioner rebuilt)" : "");
                MessagePrinter::printNormalTxt(message);
            }
            return trueResidual;
        }
        catch (const std::exception &exception) {
            MessagePrinter::printWarningTxt(
                std::string("AMGCL iteration failed: ")+exception.what());
            implementation.HavePreconditioner=false;
            return std::numeric_limits<double>::infinity();
        }
    };

    double trueResidual=runOnce(false);
    if (trueResidual>implementation.SolverOptions.Tol) {
        trueResidual=runOnce(true);
    }

    // Escalate the preconditioner rather than give up: the PDDO stencil is
    // dense and non-symmetric, and the aggregation hierarchy (or a zero-fill
    // ILU) can simply be too weak for it. Each rung is only ever built once
    // the cheaper one has been shown to fail on this very matrix.
    while (trueResidual>implementation.SolverOptions.Tol
           && implementation.Precond!=PrecondKind::Iluk) {
        std::string nextName;
        if (implementation.Precond==PrecondKind::AmgIlu0) {
            implementation.Precond=PrecondKind::Ilu0;
            nextName="single-level ILU(0)";
        }
        else {
            implementation.Precond=PrecondKind::Iluk;
            nextName="ILU("
                +std::to_string(implementation.SolverOptions.IluLevel)+")";
        }
        char notice[256];
        std::snprintf(
            notice,sizeof(notice),
            "AMGCL did not reach the requested true residual (%9.3e); "
            "retrying with the %s preconditioner",
            trueResidual,nextName.c_str());
        MessagePrinter::printWarningTxt(notice);

        auto next=makeRunner(implementation.BlockSize,implementation.Precond);
        if (!next) break;
        implementation.ActiveRunner=std::move(next);
        implementation.HavePreconditioner=false;
        trueResidual=runOnce(true);
    }

    if (trueResidual>implementation.SolverOptions.Tol) {
        char message[320];
        std::snprintf(
            message,sizeof(message),
            "AMGCL did not reach the requested true relative residual "
            "(%9.3e > %9.3e after at most %d iterations, preconditioner "
            "ladder exhausted); raise 'maxiter'/'iluk_level' or check that "
            "every boundary ghost row carries a boundary condition",
            trueResidual,implementation.SolverOptions.Tol,
            implementation.SolverOptions.MaxIters);
        MessagePrinter::printWarningTxt(message);
        return false;
    }
    return true;
}
