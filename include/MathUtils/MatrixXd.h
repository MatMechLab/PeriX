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
//+++ Function: Define the general Matrix  in AsFem
//+++           we mainly use this for the calculation of jacobian
//+++           If one wants to use Eigen's MatrixXd, please use
//+++           Eigen::MatrixXd, which is different with ours !!!
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Eigen"

#include "MathUtils/VectorXd.h"
#include "Utils/MessagePrinter.h"

class MatrixXd{
public:
    MatrixXd() = default;
    MatrixXd(const MatrixXd&) = default;
    MatrixXd(MatrixXd&&) noexcept = default;
    MatrixXd(const int &m,const int &n) { resize(m, n); }
    MatrixXd(const int &m,const int &n,const double &val) { resize(m, n, val); }
    MatrixXd& operator=(const MatrixXd&) = default;
    MatrixXd& operator=(MatrixXd&&) noexcept = default;
    ~MatrixXd() = default;

    /**
     * Resize the matrix, the memory is reallocated, the matrix is set to zero by default
     * @param m integer, the size of the 1st dimention
     * @param n integra, the size of the 2nd dimention
     */
    void resize(const int &m,const int &n){
        const std::size_t count = checkedSizeProduct(m, n);
        m_M = m;
        m_N = n;
        m_MN = m * n;
        m_Vals.assign(count, 0.0);
    }

    /**
     * Resize the matrix with a initial value, the memory is reallocated
     * @param m integer, the size of the 1st dimention
     * @param n integra, the size of the 2nd dimention
     * @param val the initial value for the resized matrix
     */
    void resize(const int &m,const int &n,const double &val){
        const std::size_t count = checkedSizeProduct(m, n);
        m_M = m;
        m_N = n;
        m_MN = m * n;
        m_Vals.assign(count, val);
    }

    /**
     * Return the pointer of the matrix's data (its a vector<double> type)
     */
    [[nodiscard]] double* getDataPtr() noexcept {
        return m_Vals.data();
    }

    /**
     * Return the pointer of the matrix's data (its a vector<double> type)
     */
    [[nodiscard]] const double* getDataPtr() const noexcept {
        return m_Vals.data();
    }

    /**
     * Return the size of the 1st dimension
     */
    [[nodiscard]] inline int getM()const noexcept { return m_M; }
    /**
     * Return the size of the 2nd dimension
     */
    [[nodiscard]] inline int getN()const noexcept { return m_N; }
    /**
     * Clean the whole matrix data
     */
    void clean() noexcept {
        m_Vals.clear();
        m_M = 0;
        m_N = 0;
        m_MN = 0;
    }
    //*****************************************
    //*** Operator overload
    //*****************************************
    /**
     * The () operator for the data access
     * @param i the index of the 1st dimension, it should start from 1, not 0!!!
     * @param j the index of the 2nd dimension, it should start from 1, not 0!!!
     */
    [[nodiscard]] inline double& operator()(const int &i,const int &j){
        return m_Vals[checkedOffset(i, j)];
    }
    /**
     * The () operator for the data access with constant reference(not editable!)
     * @param i the index of the 1st dimension, it should start from 1, not 0!!!
     * @param j the index of the 2nd dimension, it should start from 1, not 0!!!
     */
    [[nodiscard]] inline const double& operator()(const int &i,const int &j)const{
        return m_Vals[checkedOffset(i, j)];
    }
    /**
     * The [] operator, the data of our matrix is just simple vector in 1D
     * @param i the index of the data vector element, it should start from 1, not 0!!!
     */
    [[nodiscard]] inline double& operator[](const int &i){
        return m_Vals[checkedLinearOffset(i)];
    }
    /**
     * The [] operator with constant reference, the data of our matrix is just simple vector in 1D
     * @param i the index of the data vector element, it should start from 1, not 0!!!
     */
    [[nodiscard]] inline const double& operator[](const int &i)const{
        return m_Vals[checkedLinearOffset(i)];
    }
    //*****************************************
    //*** For basic mathematic operator
    //*****************************************
    //*** for =
    /**
     * The '=' for equal operator
     * @param val the double type value to set up the whole matrix
     */
    inline MatrixXd& operator=(const double &val) noexcept {
        std::fill(m_Vals.begin(), m_Vals.end(), val);
        return *this;
    }
    //****************************
    //*** for +
    /**
     * The '+' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    [[nodiscard]] inline MatrixXd operator+(const double &val)const{
        MatrixXd temp(*this);
        for(double &x: temp.m_Vals) x += val;
        return temp;
    }
    /**
     * The '+' operator between matrix and matrix
     * @param a the right-hand side matrix (the dimensions should be the same)
     */
    [[nodiscard]] inline MatrixXd operator+(const MatrixXd &a)const{
        checkSameShape(a, "a+b");
        MatrixXd temp(m_M,m_N);
        const double *rhs = a.m_Vals.data();
        double *out = temp.m_Vals.data();
        for(int i=0;i<m_MN;++i) out[i] = m_Vals[i] + rhs[i];
        return temp;
    }
    //*** for +=
    /**
     * The '+=' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    inline MatrixXd& operator+=(const double &val) noexcept {
        for(double &x: m_Vals) x += val;
        return *this;
    }
    /**
     * The '+=' operator between matrix and matrix
     * @param a the right-hand side matrix (the dimensions should be the same)
     */
    inline MatrixXd& operator+=(const MatrixXd &a){
        checkSameShape(a, "a+=b");
        const double *rhs = a.m_Vals.data();
        double *lhs = m_Vals.data();
        for(int i=0;i<m_MN;++i) lhs[i] += rhs[i];
        return *this;
    }
    //****************************
    //*** for -
    /**
     * The '-' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    [[nodiscard]] inline MatrixXd operator-(const double &val)const{
        MatrixXd temp(*this);
        for(double &x: temp.m_Vals) x -= val;
        return temp;
    }
    /**
     * The '-' operator between matrix and matrix
     * @param a the right-hand side matrix (the dimensions should be the same)
     */
    [[nodiscard]] inline MatrixXd operator-(const MatrixXd &a)const{
        checkSameShape(a, "a-b");
        MatrixXd temp(m_M,m_N);
        const double *rhs = a.m_Vals.data();
        double *out = temp.m_Vals.data();
        for(int i=0;i<m_MN;++i) out[i] = m_Vals[i] - rhs[i];
        return temp;
    }
    //*** for -=
    /**
     * The '-=' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    inline MatrixXd& operator-=(const double &val) noexcept {
        for(double &x: m_Vals) x -= val;
        return *this;
    }
    /**
     * The '-=' operator between matrix and matrix
     * @param a the right-hand side matrix (the dimensions should be the same)
     */
    inline MatrixXd& operator-=(const MatrixXd &a){
        checkSameShape(a, "a-=b");
        const double *rhs = a.m_Vals.data();
        double *lhs = m_Vals.data();
        for(int i=0;i<m_MN;++i) lhs[i] -= rhs[i];
        return *this;
    }
    //****************************
    //*** for *
    /**
     * The '*' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    [[nodiscard]] inline MatrixXd operator*(const double &val)const{
        MatrixXd temp(*this);
        for(double &x: temp.m_Vals) x *= val;
        return temp;
    }
    /**
     * The '*' operator between matrix and vector
     * @param a the right-hand side vector (the dimensions should be the same)
     */
    [[nodiscard]] inline VectorXd operator*(const VectorXd &a)const{
        if(m_N != a.getSize()){
            MessagePrinter::printErrorTxt("A*b should be applied to A matrix with the same cols as b vector!");
            MessagePrinter::exitPeriX();
        }
        VectorXd temp(m_M,0.0);
        const double *mat = m_Vals.data();
        const double *vec = a.getDataPtr();
        double *out = temp.getDataPtr();
        for(int i=0;i<m_M;++i){
            const double *row = mat + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_N);
            double sum = 0.0;
            for(int j=0;j<m_N;++j) sum += row[j] * vec[j];
            out[i] = sum;
        }
        return temp;
    }
    /**
     * The '*' operator between matrix and matrix
     * @param a the right-hand side matrix (the dimensions should be the same)
     */
    [[nodiscard]] inline MatrixXd operator*(const MatrixXd &a)const{
        if(m_N != a.getM()){
            MessagePrinter::printErrorTxt("A*B should be applied to A matrix with the same cols as the rows of B matrix!");
            MessagePrinter::exitPeriX();
        }
        MatrixXd temp(m_M, a.getN(), 0.0);
        const MatrixXd bt = a.transpose();
        const double *lhs = m_Vals.data();
        const double *rhs = bt.m_Vals.data();
        double *out = temp.m_Vals.data();
        for(int i=0;i<m_M;++i){
            const double *lhsRow = lhs + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_N);
            double *outRow = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(a.m_N);
            for(int j=0;j<a.m_N;++j){
                const double *rhsRow = rhs + static_cast<std::size_t>(j) * static_cast<std::size_t>(m_N);
                double sum = 0.0;
                for(int k=0;k<m_N;++k) sum += lhsRow[k] * rhsRow[k];
                outRow[j] = sum;
            }
        }
        return temp;
    }
    //*** for *=
    /**
     * The '*=' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    inline MatrixXd& operator*=(const double &val) noexcept {
        for(double &x: m_Vals) x *= val;
        return *this;
    }
    //****************************
    //*** for /
    /**
     * The '/' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    [[nodiscard]] inline MatrixXd operator/(const double &val)const{
        checkDivisor(val, "/");
        MatrixXd temp(*this);
        const double inv = 1.0 / val;
        for(double &x: temp.m_Vals) x *= inv;
        return temp;
    }
    //*** for /=
    /**
     * The '/=' operator between matrix and scalar
     * @param val the right-hand side scalar (double type)
     */
    inline MatrixXd& operator/=(const double &val){
        checkDivisor(val, "/=");
        const double inv = 1.0 / val;
        for(double &x: m_Vals) x *= inv;
        return *this;
    }
    /**
     * This function will set the whole matrix to zero
     */
    void setToZeros() noexcept {
        std::fill(m_Vals.begin(),m_Vals.end(),0.0);
    }
    /**
     * This function will set each element of the matrix to be random value
     */
    void setToRandom(){
        static thread_local std::mt19937_64 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        for(double &x: m_Vals) x = dist(rng);
    }
    /**
     * This function return the inverse matrix of current one, it should be noted
     * this function will not change the value of current matrix
     */
    [[nodiscard]] inline MatrixXd inverse()const{
        checkSquare("inverse");
        using RowMajorEigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const Eigen::Map<const RowMajorEigenMatrix> mat(m_Vals.data(), m_M, m_N);
        MatrixXd temp(m_M,m_N);
        Eigen::Map<RowMajorEigenMatrix>(temp.m_Vals.data(), m_M, m_N) = mat.inverse();
        return temp;
    }
    /**
     * This function return the determinant of the current matrix
     */
    [[nodiscard]] inline double det()const{
        checkSquare("determinant");
        using RowMajorEigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const Eigen::Map<const RowMajorEigenMatrix> mat(m_Vals.data(), m_M, m_N);
        return mat.determinant();
    }
    /**
     * This function return the transporse matrix of current one,
     * the current matrix will not be changed
     */
    [[nodiscard]] inline MatrixXd transpose() const{
        MatrixXd temp(m_N,m_M);
        const double *src = m_Vals.data();
        double *dst = temp.m_Vals.data();
        for(int i=0;i<m_M;++i){
            const std::size_t srcRow = static_cast<std::size_t>(i) * static_cast<std::size_t>(m_N);
            for(int j=0;j<m_N;++j){
                dst[static_cast<std::size_t>(j) * static_cast<std::size_t>(m_M) + static_cast<std::size_t>(i)] =
                    src[srcRow + static_cast<std::size_t>(j)];
            }
        }
        return temp;
    }
    /**
     * This function return the transposed matrix, important, the current matrix will
     * be transposed, if you don't want to use this one, then please call transpose()
     */
    inline void transposed(){
        std::vector<double> transposedVals(static_cast<std::size_t>(m_MN));
        const double *src = m_Vals.data();
        double *dst = transposedVals.data();
        for(int i=0;i<m_M;++i){
            const std::size_t srcRow = static_cast<std::size_t>(i) * static_cast<std::size_t>(m_N);
            for(int j=0;j<m_N;++j){
                dst[static_cast<std::size_t>(j) * static_cast<std::size_t>(m_M) + static_cast<std::size_t>(i)] =
                    src[srcRow + static_cast<std::size_t>(j)];
            }
        }
        m_Vals.swap(transposedVals);
        std::swap(m_M, m_N);
    }

    /**
     * Solve A * X = B for the matrix right-hand side B with ONE full-pivot
     * LU factorization shared by all columns (solve() re-factors A on every
     * call), and verify the result is CONSISTENT:
     *   ||A X - B||_max <= relTol * max(||B||_max, ||A||_max ||X||_max).
     * A rank-DEFICIENT A is fine as long as every column of B lies in
     * range(A) -- e.g. the PDDO moment system of a symmetric family at high
     * order, where the orthogonality constraints are still met exactly and
     * the null-space component of the solution is irrelevant. Returns false
     * only when the system is numerically INCONSISTENT (B not reachable --
     * a truly degenerate family); the caller owns the error message, since
     * it knows the context (e.g. which PD node is degenerate).
     * @param B      right-hand side matrix (same rows as A)
     * @param X      output solution matrix (resized as needed)
     * @param relTol relative consistency tolerance (e.g. 1e-8)
     */
    [[nodiscard]] bool solveMatrixChecked(const MatrixXd &B, MatrixXd &X, const double &relTol) const;

    /**
     * This function solve the system equation Ax=b
     * @param x input vector which stores the solution
     * @param b input vector which serves as the right hand side term
    */
    void solve(const VectorXd &b,VectorXd &x) const;
    /**
     * This function solve the system equation Ax=b, with a given b vector and return the solution x
     * @param b input vector which serves as the right hand side term
    */
    [[nodiscard]] VectorXd solve(const VectorXd &b) const;

private:
    [[nodiscard]] static std::size_t checkedSizeProduct(int m, int n) {
        if (m < 0 || n < 0) {
            MessagePrinter::printErrorTxt("matrix dimensions must be non-negative");
            MessagePrinter::exitPeriX();
        }
        const long long mn = static_cast<long long>(m) * static_cast<long long>(n);
        if (mn > static_cast<long long>(std::numeric_limits<int>::max())) {
            MessagePrinter::printErrorTxt("matrix size is too large for MatrixXd");
            MessagePrinter::exitPeriX();
        }
        return static_cast<std::size_t>(mn);
    }

    [[nodiscard]] std::size_t checkedOffset(int i, int j) const {
        if(i<1||i>m_M){
            MessagePrinter::printErrorTxt("i="+std::to_string(i)+" is out of range(m="+std::to_string(m_M)+") in MatrixXd.h");
            MessagePrinter::exitPeriX();
        }
        if(j<1||j>m_N){
            MessagePrinter::printErrorTxt("j="+std::to_string(j)+" is out of range(n="+std::to_string(m_N)+") in MatrixXd.h");
            MessagePrinter::exitPeriX();
        }
        return static_cast<std::size_t>(i - 1) * static_cast<std::size_t>(m_N) + static_cast<std::size_t>(j - 1);
    }

    [[nodiscard]] std::size_t checkedLinearOffset(int i) const {
        if(i<1||i>m_MN){
            MessagePrinter::printErrorTxt("i="+std::to_string(i)+" is out of range(mn="+std::to_string(m_MN)+") in MatrixXd.h");
            MessagePrinter::exitPeriX();
        }
        return static_cast<std::size_t>(i - 1);
    }

    void checkSameShape(const MatrixXd &a, const char *op) const {
        if(m_M!=a.m_M || m_N!=a.m_N){
            MessagePrinter::printErrorTxt(std::string(op)+" can't be applied to two matrix with different size");
            MessagePrinter::exitPeriX();
        }
    }

    void checkSquare(const char *op) const {
        if(m_M != m_N){
            MessagePrinter::printErrorTxt(std::string(op)+" only works for square matrix");
            MessagePrinter::exitPeriX();
        }
    }

    static void checkDivisor(double val, const char *op) {
        if(std::abs(val) < 1.0e-15){
            MessagePrinter::printErrorTxt("val="+std::to_string(val)+" is singular for "+std::string(op)+" operator in MatrixXd");
            MessagePrinter::exitPeriX();
        }
    }

private:
    std::vector<double> m_Vals;/**< double type vector to store the matrix element*/
    int m_M = 0; /**< the integer variable for the 1st dimension of the matrix*/
    int m_N = 0; /**< the integer variable for the 2nd dimension of the matrix*/
    int m_MN = 0;/**< the integer variable for the total length of the matrix*/
};
