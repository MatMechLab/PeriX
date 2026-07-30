//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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
    const VectorXd &expected,const bool initialize) {
    nlohmann::ordered_json parameters={
        {"tol",1.0e-11},
        {"maxiter",300},
        {"verbose",false}
    };
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

} // namespace

int main() {
    int failures=0;

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

    if (failures==0) {
        std::printf("AMGCL solver tests passed\n");
    }
    return failures==0 ? 0 : 1;
}
