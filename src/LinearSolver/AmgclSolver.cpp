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
//+++ Function: Minimal public AMGCL backend. Uses BiCGSTAB with
//+++           smoothed-aggregation AMG and ILU(0), automatically
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

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/solver/bicgstab.hpp>

#include "Utils/MessagePrinter.h"

namespace {

struct Options {
    double Tol=1.0e-10;
    int MaxIters=200;
    bool Verbose=false;
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

template <int B,bool UseAmg>
class Runner final : public RunnerBase {
public:
    using MatrixValue=typename BlockTraits<B>::MatrixValue;
    using Backend=typename BlockTraits<B>::Backend;
    using Preconditioner=std::conditional_t<
        UseAmg,
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0>,
        amgcl::relaxation::as_preconditioner<
            Backend,
            amgcl::relaxation::ilu0>>;
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
std::unique_ptr<RunnerBase> makeRunnerForBlock(const bool useAmg) {
    if (useAmg) return std::make_unique<Runner<B,true>>();
    return std::make_unique<Runner<B,false>>();
}

std::unique_ptr<RunnerBase> makeRunner(
    const int blockSize,const bool useAmg) {
    switch (blockSize) {
        case 1: return makeRunnerForBlock<1>(useAmg);
        case 2: return makeRunnerForBlock<2>(useAmg);
        case 3: return makeRunnerForBlock<3>(useAmg);
        case 4: return makeRunnerForBlock<4>(useAmg);
        case 5: return makeRunnerForBlock<5>(useAmg);
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
    bool UseAmg=true;
    bool FellBackToIlu0=false;
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

    implementation.N=m_N;
    const int *rows=K.getCSRRowsIndexPtr();
    const int *cols=K.getCSRColsIndexPtr();
    implementation.Rows.assign(rows,rows+m_N+1);
    implementation.Cols.assign(cols,cols+m_NNZ);
    implementation.Values.resize(static_cast<std::size_t>(m_NNZ));
    implementation.Rhs.resize(static_cast<std::size_t>(m_N));
    implementation.Solution.resize(static_cast<std::size_t>(m_N));

    implementation.BlockSize=detectBlockSize(K);
    implementation.ActiveRunner=
        makeRunner(implementation.BlockSize,true);
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

    if (trueResidual>implementation.SolverOptions.Tol
        && implementation.UseAmg
        && !implementation.FellBackToIlu0) {
        MessagePrinter::printWarningTxt(
            "AMGCL multigrid did not reach the requested true residual; "
            "retrying with the fixed single-level ILU(0) fallback");
        implementation.UseAmg=false;
        implementation.FellBackToIlu0=true;
        implementation.ActiveRunner=
            makeRunner(implementation.BlockSize,false);
        implementation.HavePreconditioner=false;
        trueResidual=runOnce(true);
    }

    if (trueResidual>implementation.SolverOptions.Tol) {
        char message[224];
        std::snprintf(
            message,sizeof(message),
            "AMGCL did not reach the requested true relative residual "
            "(%9.3e > %9.3e after at most %d iterations)",
            trueResidual,implementation.SolverOptions.Tol,
            implementation.SolverOptions.MaxIters);
        MessagePrinter::printWarningTxt(message);
        return false;
    }
    return true;
}
