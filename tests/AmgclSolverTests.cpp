//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "LinearSolver/AmgclSolver.h"

namespace {

double relativeResidual(
    SparseMatrix &matrix,const VectorXd &rhs,const VectorXd &solution) {
    const int n=matrix.getSize();
    const int *rows=matrix.getCSRRowsIndexPtr();
    const int *cols=matrix.getCSRColsIndexPtr();
    const double *values=matrix.getCSRValuesPtr();

    double residualSquared=0.0;
    double rhsSquared=0.0;
    for (int row=0;row<n;++row) {
        double product=0.0;
        for (int entry=rows[row];entry<rows[row+1];++entry) {
            product+=values[entry]*solution(cols[entry]+1);
        }
        const double residual=rhs(row+1)-product;
        residualSquared+=residual*residual;
        rhsSquared+=rhs(row+1)*rhs(row+1);
    }
    return std::sqrt(residualSquared/rhsSquared);
}

VectorXd multiply(SparseMatrix &matrix,const VectorXd &solution) {
    VectorXd rhs(matrix.getSize(),0.0);
    const int *rows=matrix.getCSRRowsIndexPtr();
    const int *cols=matrix.getCSRColsIndexPtr();
    const double *values=matrix.getCSRValuesPtr();
    for (int row=0;row<matrix.getSize();++row) {
        double product=0.0;
        for (int entry=rows[row];entry<rows[row+1];++entry) {
            product+=values[entry]*solution(cols[entry]+1);
        }
        rhs(row+1)=product;
    }
    return rhs;
}

SparseMatrix scalarPoissonMatrix(const int n) {
    SparseMatrix matrix(n);
    for (int row=1;row<=n;++row) {
        if (row>1) matrix.insert(row,row-1);
        matrix.insert(row,row);
        if (row<n) matrix.insert(row,row+1);
    }
    matrix.setup();
    for (int row=1;row<=n;++row) {
        if (row>1) matrix.insertValue(row,row-1,-1.0);
        matrix.insertValue(row,row,2.5);
        if (row<n) matrix.insertValue(row,row+1,-1.0);
    }
    return matrix;
}

SparseMatrix twoDofBlockMatrix() {
    constexpr int nodes=3;
    constexpr int dofs=2;
    SparseMatrix matrix(nodes*dofs);

    for (int node=0;node<nodes;++node) {
        for (int rowComponent=0;rowComponent<dofs;++rowComponent) {
            const int row=node*dofs+rowComponent+1;
            for (int columnNode=std::max(0,node-1);
                 columnNode<=std::min(nodes-1,node+1);++columnNode) {
                for (int columnComponent=0;
                     columnComponent<dofs;++columnComponent) {
                    matrix.insert(
                        row,columnNode*dofs+columnComponent+1);
                }
            }
        }
    }
    matrix.setup();

    for (int node=0;node<nodes;++node) {
        for (int rowComponent=0;rowComponent<dofs;++rowComponent) {
            const int row=node*dofs+rowComponent+1;
            for (int columnNode=std::max(0,node-1);
                 columnNode<=std::min(nodes-1,node+1);++columnNode) {
                for (int columnComponent=0;
                     columnComponent<dofs;++columnComponent) {
                    double value=0.0;
                    if (columnNode==node) {
                        value=(columnComponent==rowComponent) ? 4.0 : 0.25;
                    }
                    else {
                        value=(columnComponent==rowComponent) ? -1.0 : -0.05;
                    }
                    matrix.insertValue(
                        row,columnNode*dofs+columnComponent+1,value);
                }
            }
        }
    }
    return matrix;
}

bool solveKnownSystem(
    const char *name,SparseMatrix &matrix,AmgclSolver &solver,
    const VectorXd &expected,const bool initialize,
    nlohmann::ordered_json &parameters) {
    if (initialize) solver.initSolver(matrix,parameters);

    VectorXd rhs=multiply(matrix,expected);
    VectorXd solution(matrix.getSize(),0.0);
    if (!solver.solveLinearSystem(matrix,rhs,solution)) {
        std::printf("FAIL %s: solver reported failure\n",name);
        return false;
    }

    double maximumError=0.0;
    for (int row=1;row<=matrix.getSize();++row) {
        maximumError=std::max(
            maximumError,std::fabs(solution(row)-expected(row)));
    }
    const double residual=relativeResidual(matrix,rhs,solution);
    if (maximumError>1.0e-8 || residual>1.0e-11) {
        std::printf(
            "FAIL %s: max error=%9.3e residual=%9.3e\n",
            name,maximumError,residual);
        return false;
    }
    return true;
}

bool solveKnownSystem(
    const char *name,SparseMatrix &matrix,AmgclSolver &solver,
    const VectorXd &expected,const bool initialize) {
    nlohmann::ordered_json parameters={
        {"tol",1.0e-11},
        {"maxiter",300},
        {"verbose",false}
    };
    return solveKnownSystem(
        name,matrix,solver,expected,initialize,parameters);
}

} // namespace

int main() {
    int failures=0;

    // Snapshot the ambient OpenMP width before any solve runs. The solver sizes
    // its own team for the AMGCL kernels, but PeriX's assembly loops must keep
    // running on the user's OMP_NUM_THREADS team, so every solve has to put the
    // width back. Sampling this after a solve would compare a clobbered value
    // against itself and pass even when the restore is missing.
#ifdef _OPENMP
    const int ambientThreads=omp_get_max_threads();
#endif

    SparseMatrix scalar=scalarPoissonMatrix(23);
    VectorXd scalarExpected(23,0.0);
    for (int row=1;row<=scalarExpected.getSize();++row) {
        scalarExpected(row)=std::sin(0.2*static_cast<double>(row));
    }
    AmgclSolver scalarSolver;
    if (!solveKnownSystem(
            "scalar AMG",scalar,scalarSolver,scalarExpected,true)) {
        ++failures;
    }

    SparseMatrix block=twoDofBlockMatrix();
    VectorXd blockExpected(6,0.0);
    for (int row=1;row<=blockExpected.getSize();++row) {
        blockExpected(row)=0.5*static_cast<double>(row);
    }
    AmgclSolver blockSolver;
    if (!solveKnownSystem(
            "two-DoF block AMG",block,blockSolver,blockExpected,true)) {
        ++failures;
    }

    for (int row=1;row<=block.getSize();++row) {
        block.addValue(row,row,0.5);
    }
    if (!solveKnownSystem(
            "reused preconditioner",block,blockSolver,blockExpected,false)) {
        ++failures;
    }

    // An explicit team must be honoured without changing the answer.
    SparseMatrix pinned=scalarPoissonMatrix(23);
    VectorXd pinnedExpected(23,0.0);
    for (int row=1;row<=pinnedExpected.getSize();++row) {
        pinnedExpected(row)=std::sin(0.2*static_cast<double>(row));
    }
    nlohmann::ordered_json pinnedParameters={
        {"tol",1.0e-11},
        {"maxiter",300},
        {"verbose",false},
        {"solver_threads",2}
    };
    AmgclSolver pinnedSolver;
    if (!solveKnownSystem(
            "explicit solver_threads",pinned,pinnedSolver,pinnedExpected,true,
            pinnedParameters)) {
        ++failures;
    }

    // The solver sizes its own OpenMP team, but PeriX's assembly loops must keep
    // running on the user's OMP_NUM_THREADS width, so the override has to be put
    // back once the solve returns.
#ifdef _OPENMP
    if (omp_get_max_threads()!=ambientThreads) {
        std::printf(
            "FAIL thread-team restore: ambient threads %d became %d\n",
            ambientThreads,omp_get_max_threads());
        ++failures;
    }
#endif

    if (failures==0) {
        std::printf("AMGCL solver tests passed\n");
    }
    return failures==0 ? 0 : 1;
}
