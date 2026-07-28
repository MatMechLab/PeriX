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
//+++ Function: the vectorxd class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>


/**
 * This class implements the vector with dynamic size, which is different from the vector3d(fixed size, mainly used in shape function calculation)
 */
class VectorXd{
public:
    /**
     * construtor for different purpose
     */
    VectorXd() = default;
    VectorXd(const VectorXd&) = default;
    VectorXd(VectorXd&&) noexcept = default;
    explicit VectorXd(int m) { resize(m); }
    VectorXd(int m, double val) { resize(m, val); }
    VectorXd& operator=(const VectorXd&) = default;
    VectorXd& operator=(VectorXd&&) noexcept = default;

    ~VectorXd() = default;

    /**
     * resize the vector
     * @param m the size of the vector
     */
    void resize(const int &m){
        checkNonNegativeSize(m);
        m_Vals.resize(static_cast<std::size_t>(m), 0.0);
    }
    /**
     * resize the vector with initial value
     * @param m the size of the vector
     * @param val the intial value of the resized vector
     */
    void resize(const int &m,const double &val){
        checkNonNegativeSize(m);
        m_Vals.assign(static_cast<std::size_t>(m), val);
    }
    /**
     * return the size of the vector
     */
    [[nodiscard]] inline int getSize() const noexcept {
        return static_cast<int>(m_Vals.size());
    }

    /**
     * clean the whole vector
     */
    void clean() noexcept {
        m_Vals.clear();
    }

    /**
     * check the status of current vector
     * @return true if current vector is empty
     */
    [[nodiscard]] inline bool empty() const noexcept {
        return m_Vals.empty();
    }

    /**
     * get the copy of vector
     * @return copy of vector
     */
    [[nodiscard]] std::vector<double> getDataCopy() const {
        return m_Vals;
    }

    /**
     * get the reference of the vector data
     * @return reference of the data
     */
    std::vector<double>& getDataRef() noexcept {
        return m_Vals;
    }

    /**
     * get the const reference of the vector data
     * @return const reference of the data
     */
    [[nodiscard]] const std::vector<double>& getDataRef() const noexcept {
        return m_Vals;
    }

    /**
     * get the pointer of current vector's data
     */
    [[nodiscard]] double* getDataPtr() noexcept {
        return m_Vals.data();
    }
    /**
     * get the const pointer of current vector's data
     */
    [[nodiscard]] const double* getDataPtr() const noexcept {
        return m_Vals.data();
    }
    //*****************************************
    //*** Operator overload
    //*****************************************
    /**
     * () operator for the element access of the vector
     * @param i the index of the single element
     */
    [[nodiscard]] inline double& operator()(const int &i){
        const std::size_t idx = checkedOffset(i);
        return m_Vals[idx];
    }
    /**
     * const () operator for the elemenet access of the vector
     * @param i the index of the single element
     */
    [[nodiscard]] inline const double& operator()(const int &i)const{
        const std::size_t idx = checkedOffset(i);
        return m_Vals[idx];
    }

    /**
     * add value to the current VectorXd
     * @param i index, start from 1
     * @param val value to be added
     */
    void addValue(const int &i,const double &val) {
        const std::size_t idx = checkedOffset(i);
        m_Vals[idx] += val;
    }

    /**
     * insert value to the specific position
     * @param i index, start from 1
     * @param val value to be inserted
     */
    void insertValue(const int &i,const double &val) {
        const std::size_t idx = checkedOffset(i);
        m_Vals[idx] = val;
    }
    //*****************************************
    //*** For basic mathematic operator
    //*****************************************
    //*** for =
    /**
     * '=' operator for scalar
     * @param val right hand side scalar
     */
    inline VectorXd& operator=(double val) noexcept {
        std::fill(m_Vals.begin(), m_Vals.end(), val);
        return *this;
    }

    //*** for +
    /**
     * '+' operator for scalar
     * @param val the right hand side scalar
     */
    inline VectorXd operator+(double val) const {
        VectorXd temp(*this);
        for (double& x : temp.m_Vals) x += val;
        return temp;
    }
    /**
     * '+' for vector
     * @param a the right hand side vector
     */
    inline VectorXd operator+(const VectorXd& a) const {
        checkSameSize(a, "a+b");
        VectorXd temp(getSize());
        const std::size_t n = m_Vals.size();
        const double* rhs = a.m_Vals.data();
        double* out = temp.m_Vals.data();
        for (std::size_t i = 0; i < n; ++i) out[i] = m_Vals[i] + rhs[i];
        return temp;
    }
    //*** for +=
    /**
     * '+=' operator for scalar
     * @param val the right hand side scalar
     */
    inline VectorXd& operator+=(double val) noexcept {
        for (double& x : m_Vals) x += val;
        return *this;
    }
    /**
     * '+=' for vector
     * @param a the right hand side vector
     */
    inline VectorXd& operator+=(const VectorXd& a) {
        checkSameSize(a, "a+=b");
        const std::size_t n = m_Vals.size();
        const double* rhs = a.m_Vals.data();
        double* lhs = m_Vals.data();
        for (std::size_t i = 0; i < n; ++i) lhs[i] += rhs[i];
        return *this;
    }
    //************************
    //*** for -
    /**
     * '-' operator for scalar
     * @param val right hand side scalar value
     */
    inline VectorXd operator-(double val) const {
        VectorXd temp(*this);
        for (double& x : temp.m_Vals) x -= val;
        return temp;
    }
    /**
     * '-' operator for vector
     * @param a right hand side vector
     */
    inline VectorXd operator-(const VectorXd& a) const {
        checkSameSize(a, "a-b");
        VectorXd temp(getSize());
        const std::size_t n = m_Vals.size();
        const double* rhs = a.m_Vals.data();
        double* out = temp.m_Vals.data();
        for (std::size_t i = 0; i < n; ++i) out[i] = m_Vals[i] - rhs[i];
        return temp;
    }
    //*** for -=
    /**
     * '-' operator for scalar
     * @param val right hand side scalar
     */
    inline VectorXd& operator-=(double val) noexcept {
        for (double& x : m_Vals) x -= val;
        return *this;
    }
    /**
     * '-=' operator for vector
     * @param a right hand side vector
     */
    inline VectorXd& operator-=(const VectorXd& a) {
        checkSameSize(a, "a-=b");
        const std::size_t n = m_Vals.size();
        const double* rhs = a.m_Vals.data();
        double* lhs = m_Vals.data();
        for (std::size_t i = 0; i < n; ++i) lhs[i] -= rhs[i];
        return *this;
    }
    //***********************************************
    //*** for *
    /**
     * '*' operator for scalar
     * @param val right hand side scalar
     */
    inline VectorXd operator*(double val) const {
        VectorXd temp(*this);
        for (double& x : temp.m_Vals) x *= val;
        return temp;
    }
    //*** for *=
    /**
     * '*=' operator for scalar
     * @param val the right hand side scalar
     */
    inline VectorXd& operator*=(double val) noexcept {
        for (double& x : m_Vals) x *= val;
        return *this;
    }
    //**********************************************
    //*** for /
    /**
     * '/' operator for scalar
     * @param val right hand side scalar
     */
    inline VectorXd operator/(double val) const {
        checkDivisor(val, "/");
        VectorXd temp(*this);
        const double inv = 1.0 / val;
        for (double& x : temp.m_Vals) x *= inv;
        return temp;
    }
    //*** for /=
    /**
     * '/=' operator for scalar
     * @param val right hand side scalar
     */
    inline VectorXd& operator/=(double val) {
        checkDivisor(val, "/=");
        const double inv = 1.0 / val;
        for (double& x : m_Vals) x *= inv;
        return *this;
    }
    //***********************************************
    /**
     * set vector's value to be zero
     */
    void setToZeros(){
        std::fill(m_Vals.begin(),m_Vals.end(),0.0);
    }
    /**
     * set vector's components to be random values
     */
    void setToRandom(){
        static thread_local std::mt19937_64 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (double& x : m_Vals) x = dist(rng);
    }
    /**
     * get the L2 norm of current vector
     */
    [[nodiscard]] inline double norm() const noexcept {
        return std::sqrt(normsq());
    }
    /**
     * get the squared L2 norm of current vector
     */
    [[nodiscard]] inline double normsq() const noexcept {
        double sum = 0.0;
        const double* vals = m_Vals.data();
        const std::size_t n = m_Vals.size();
        for (std::size_t i = 0; i < n; ++i) sum += vals[i] * vals[i];
        return sum;
    }

    void print()const;

private:
    /**
     * check the validation of vector's zie
     * @param m input size
     */
    static void checkNonNegativeSize(int m) {
        if (m < 0) {
            std::cerr << "*** Error: vector size m=" << m << " is negative" << std::endl;
            std::abort();
        }
    }

    /**
     * check the validation of input index
     * @param i index, start from 1
     */
    [[nodiscard]] std::size_t checkedOffset(int i) const {
        if (i < 1 || static_cast<std::size_t>(i) > m_Vals.size()) {
            std::cerr << "*** Error: i=" << i << " is out of range(size=" << m_Vals.size() << ")" << std::endl;
            std::abort();
        }
        return static_cast<std::size_t>(i - 1);
    }

    /**
     * check if two vectors have the same size
     * @param a input vector
     * @param op error message
     */
    void checkSameSize(const VectorXd& a, const char* op) const {
        if (m_Vals.size() != a.m_Vals.size()) {
            std::cerr << "*** Error: " << op << " can't be applied to two vectors with different size" << std::endl;
            std::abort();
        }
    }

    /**
     * check if val is nonsingular for / operator
     * @param val divided value
     * @param op error message
     */
    static void checkDivisor(double val, const char* op) {
        if (std::abs(val) < 1.0e-16) {
            std::cerr << "*** Error: val=" << val << " is singular for " << op << " operator in VectorXd" << std::endl;
            std::abort();
        }
    }
private:
    std::vector<double> m_Vals;/**< the double array for vector's components*/
};