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
//+++ Function: Define the general Matrix  in PeriX
//+++           we mainly use this for the calculation of jacobian
//+++           If one wants to use Eigen's MatrixXd, please use
//+++           Eigen::MatrixXd, which is different with ours !!!
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MathUtils/MatrixXd.h"

namespace {

using RowMajorEigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}

bool MatrixXd::solveMatrixChecked(const MatrixXd &B, MatrixXd &X, const double &relTol) const{
    checkSquare("solveMatrixChecked");
    if (B.getM()!=m_M) {
        MessagePrinter::printErrorTxt("solveMatrixChecked: B must have the same number of rows as A");
        MessagePrinter::exitPeriX();
    }
    if (X.getM()!=m_N || X.getN()!=B.getN()) X.resize(m_N, B.getN());

    const Eigen::Map<const RowMajorEigenMatrix> A(m_Vals.data(), m_M, m_N);
    const Eigen::Map<const RowMajorEigenMatrix> Bm(B.getDataPtr(), B.getM(), B.getN());
    Eigen::Map<RowMajorEigenMatrix> Xm(X.getDataPtr(), X.getM(), X.getN());

    const Eigen::FullPivLU<Eigen::MatrixXd> lu(A);   // one factorization for all columns
    Xm = lu.solve(Bm);

    // Consistency check: for a rank-deficient A, lu.solve() returns an exact
    // solution only when B is in range(A); otherwise the result is garbage
    // and the residual is O(||B||).
    const double resid = (A*Xm - Bm).cwiseAbs().maxCoeff();
    const double scale = std::max(Bm.cwiseAbs().maxCoeff(),
                                  A.cwiseAbs().maxCoeff()*Xm.cwiseAbs().maxCoeff());
    return resid <= relTol*std::max(scale, std::numeric_limits<double>::min());
}

void MatrixXd::solve(const VectorXd &b,VectorXd &x) const{
    if(b.getSize()!=getM()){
        MessagePrinter::printErrorTxt("size of rhs vector b is not equal to the row number of your matrix, can't execute the solve function");
        MessagePrinter::exitPeriX();
    }
    if(x.getSize()!=getN()){
        MessagePrinter::printErrorTxt("size of solution vector x is not equal to the column number of your matrix, can't execute the solve function");
        MessagePrinter::exitPeriX();
    }

    const Eigen::Map<const RowMajorEigenMatrix> A(m_Vals.data(), m_M, m_N);
    const Eigen::Map<const Eigen::VectorXd> B(b.getDataPtr(), b.getSize());
    Eigen::Map<Eigen::VectorXd> X(x.getDataPtr(), x.getSize());
    X = A.fullPivLu().solve(B);
}

VectorXd MatrixXd::solve(const VectorXd &b) const{
    if(b.getSize()!=getM()){
        MessagePrinter::printErrorTxt("size of rhs vector b is not equal to the row number of your matrix, can't execute the solve function");
        MessagePrinter::exitPeriX();
    }

    VectorXd x(getN(),0.0);
    const Eigen::Map<const RowMajorEigenMatrix> A(m_Vals.data(), m_M, m_N);
    const Eigen::Map<const Eigen::VectorXd> B(b.getDataPtr(), b.getSize());
    Eigen::Map<Eigen::VectorXd> X(x.getDataPtr(), x.getSize());
    X = A.fullPivLu().solve(B);
    return x;
}
