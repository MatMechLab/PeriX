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
//+++ Function: Implement rank-2 tensor class for the common
//+++           tensor manipulation in AsFem
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "MathUtils/Vector3d.h"
#include "MathUtils/Rank4Tensor.h"

#include "Utils/MessagePrinter.h"

class Rank4Tensor;

using std::abs;
using std::sqrt;
using std::to_string;
using std::vector;

/**
 * This class implement the general manipulation for rank-2 tensor.
 */
class Rank2Tensor{
public:
    /**
     * different initial method for rank-2 tensor
     */
    enum InitMethod{
        ZERO,
        IDENTITY,
        RANDOM
    };
public:
    /**
     * constructor
     */
    Rank2Tensor();
    /**
     * constructor with default value
     * @param val the right hand side scalar value
     */
    Rank2Tensor(const double &val);
    /**
     * constructor with a rank-2 tensor
     * @param a the right hand side rank-2 tensor
     */
    Rank2Tensor(const Rank2Tensor &a);
    /**
     * constructor with specific init method
     * @param initmethod the initial method
     */
    Rank2Tensor(const InitMethod &initmethod);
    /**
     * the deconstructor
     */
    ~Rank2Tensor();
    //**********************************************************************
    //*** for row, col and other elements access
    //**********************************************************************
    /**
     * get the ith row of the rank-2 tensor
     * @param i i-th row number, start from 1 to 3
     */
    [[nodiscard]] inline Vector3d getIthRow(const int &i)const{
        checkIndex(i, 1);
        Vector3d temp(0.0);
        const int row=i-1;
        temp(1)=m_Vals[row][0];
        temp(2)=m_Vals[row][1];
        temp(3)=m_Vals[row][2];
        return temp;
    }
    /**
     * get the ith column of the rank-2 tensor
     * @param i i-th col number, start from 1 to 3
     */
    [[nodiscard]] inline Vector3d getIthCol(const int &i)const{
        checkIndex(1, i);
        Vector3d temp(0.0);
        const int col=i-1;
        temp(1)=m_Vals[0][col];
        temp(2)=m_Vals[1][col];
        temp(3)=m_Vals[2][col];
        return temp;
    }
    //**********************************************************************
    //*** for operator override
    //**********************************************************************
    //*******************************
    //*** for ()  operator
    //*******************************
    /** for index based access(start from 1, instead of zero !!!)
     * @param i i index of the rank-2 tensor, start from 1
     * @param j j index of the rank-2 tensor, start from 1
     */
    [[nodiscard]] inline const double& operator()(const int &i,const int &j) const{
        checkIndex(i, j);
        return m_Vals[i-1][j-1];
    }
    /** for index based access(start from 1, instead of zero !!!)
     * @param i i index of the rank-2 tensor, start from 1
     * @param j j index of the rank-2 tensor, start from 1
     */
    [[nodiscard]] inline double& operator()(const int &i,const int &j){
        checkIndex(i, j);
        return m_Vals[i-1][j-1];
    }
    //*******************************
    //*** for =  operator
    //*******************************
    /**
     * '=' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor& operator=(const double &a){
        for(auto & m_val : m_Vals){
            m_val[0]=a;m_val[1]=a;m_val[2]=a;
        }
        return *this;
    }
    /**
     * '=' operator for rank-2 tensor
     * @param a the right hand side rank-2 tensor
     */
    inline Rank2Tensor& operator=(const Rank2Tensor &a){
        if(this==&a) return *this;
        for(int i=0;i<3;i++){
            m_Vals[i][0]=a.m_Vals[i][0];
            m_Vals[i][1]=a.m_Vals[i][1];
            m_Vals[i][2]=a.m_Vals[i][2];
        }
        return *this;
    }
    //*******************************
    //*** for +  operator
    //*******************************
    /**
     * '+' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor operator+(const double &a) const{
        Rank2Tensor temp(0.0);
        for(int i=1;i<=3;i++){
            temp.m_Vals[i-1][0]=m_Vals[i-1][0]+a;
            temp.m_Vals[i-1][1]=m_Vals[i-1][1]+a;
            temp.m_Vals[i-1][2]=m_Vals[i-1][2]+a;
        }
        return temp;
    }
    /**
     * '+' operator for rank-2 tensor
     * @param a right hand side rank-2 tensor
     */
    inline Rank2Tensor operator+(const Rank2Tensor &a) const{
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=m_Vals[i][0]+a.m_Vals[i][0];
            temp.m_Vals[i][1]=m_Vals[i][1]+a.m_Vals[i][1];
            temp.m_Vals[i][2]=m_Vals[i][2]+a.m_Vals[i][2];
        }
        return temp;
    }
    //*******************************
    //*** for +=  operator
    //*******************************
    /**
     * '+=' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor& operator+=(const double &a) {
        for(auto & m_val : m_Vals){
            m_val[0]=m_val[0]+a;
            m_val[1]=m_val[1]+a;
            m_val[2]=m_val[2]+a;
        }
        return *this;
    }
    /**
     * '+=' for rank-2 tensor
     * @param a right hand side tensor
     */
    inline Rank2Tensor& operator+=(const Rank2Tensor &a){
        for(int i=0;i<3;i++){
            m_Vals[i][0]=m_Vals[i][0]+a.m_Vals[i][0];
            m_Vals[i][1]=m_Vals[i][1]+a.m_Vals[i][1];
            m_Vals[i][2]=m_Vals[i][2]+a.m_Vals[i][2];
        }
        return *this;
    }
    /**
     * add scalar value to diagnal element, the new_tensor=old_tensor+I*a, where I is the identity tensor
     * @param val the scalar value to be added
     */
    inline void addIa(const double &val){
        m_Vals[0][0]+=val;
        m_Vals[1][1]+=val;
        m_Vals[2][2]+=val;
    }
    //*******************************
    //*** for -  operator
    //*******************************
    /**
     * '-' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor operator-(const double &a) const{
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=m_Vals[i][0]-a;
            temp.m_Vals[i][1]=m_Vals[i][1]-a;
            temp.m_Vals[i][2]=m_Vals[i][2]-a;
        }
        return temp;
    }
    /**
     * '-' operator for rank-2 tensor
     * @param a right hand side rank-2 tensor
     */
    inline Rank2Tensor operator-(const Rank2Tensor &a) const{
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=m_Vals[i][0]-a.m_Vals[i][0];
            temp.m_Vals[i][1]=m_Vals[i][1]-a.m_Vals[i][1];
            temp.m_Vals[i][2]=m_Vals[i][2]-a.m_Vals[i][2];
        }
        return temp;
    }
    //*******************************
    //*** for -=  operator
    //*******************************
    /**
     * '-=' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor& operator-=(const double &a) {
        for(auto & m_val : m_Vals){
            m_val[0]=m_val[0]-a;
            m_val[1]=m_val[1]-a;
            m_val[2]=m_val[2]-a;
        }
        return *this;
    }
    /**
     * '-=' for rank-2 tensor
     * @param a right hand side tensor
     */
    inline Rank2Tensor& operator-=(const Rank2Tensor &a){
        for(int i=0;i<3;i++){
            m_Vals[i][0]-=a.m_Vals[i][0];
            m_Vals[i][1]-=a.m_Vals[i][1];
            m_Vals[i][2]-=a.m_Vals[i][2];
        }
        return *this;
    }
    //*******************************
    //*** for *  operator
    //*******************************
    /**
     * '*' for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor operator*(const double &a) const{
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=m_Vals[i][0]*a;
            temp.m_Vals[i][1]=m_Vals[i][1]*a;
            temp.m_Vals[i][2]=m_Vals[i][2]*a;
        }
        return temp;
    }
    /**
     * '*' operator for vector3d
     * @param a right hand side vector3d
     */
    inline Vector3d operator*(const Vector3d &a) const{
        Vector3d temp(0.0);
        const double a1=a(1), a2=a(2), a3=a(3);
        temp(1)=m_Vals[0][0]*a1 + m_Vals[0][1]*a2 + m_Vals[0][2]*a3;
        temp(2)=m_Vals[1][0]*a1 + m_Vals[1][1]*a2 + m_Vals[1][2]*a3;
        temp(3)=m_Vals[2][0]*a1 + m_Vals[2][1]*a2 + m_Vals[2][2]*a3;
        return temp;
    }
    /**
     * '*' operator for rank-2 tensor, return \f$a_{ij}=b_{ik}c_{kj}\f$
     * @param a the right hand side rank-2 tensor
     */
    inline Rank2Tensor operator*(const Rank2Tensor &a) const{
        // return A*B(still rank-2 tensor)
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            const double ai0=m_Vals[i][0];
            const double ai1=m_Vals[i][1];
            const double ai2=m_Vals[i][2];
            temp.m_Vals[i][0]=ai0*a.m_Vals[0][0] + ai1*a.m_Vals[1][0] + ai2*a.m_Vals[2][0];
            temp.m_Vals[i][1]=ai0*a.m_Vals[0][1] + ai1*a.m_Vals[1][1] + ai2*a.m_Vals[2][1];
            temp.m_Vals[i][2]=ai0*a.m_Vals[0][2] + ai1*a.m_Vals[1][2] + ai2*a.m_Vals[2][2];
        }
        return temp;
    }
    /**
     * double dot(:, the double contruction) between two rank-2 tensor, the result is \f$\sum a_{ij}b_{ij}\f$, which is a scalar
     * @param a the right hand side rank-2 tensor
     */
    [[nodiscard]] inline double doubledot(const Rank2Tensor &a) const{
        // return A:B calculation
        return m_Vals[0][0]*a.m_Vals[0][0] + m_Vals[0][1]*a.m_Vals[0][1] + m_Vals[0][2]*a.m_Vals[0][2]
             + m_Vals[1][0]*a.m_Vals[1][0] + m_Vals[1][1]*a.m_Vals[1][1] + m_Vals[1][2]*a.m_Vals[1][2]
             + m_Vals[2][0]*a.m_Vals[2][0] + m_Vals[2][1]*a.m_Vals[2][1] + m_Vals[2][2]*a.m_Vals[2][2];
    }
    /**
     * double dot(:, the double contruction) between rank-2 and rank-4 tensor,
     * the result is \f$\sum a_{ij}c_{ijkl}=b_{kl}\f$, which is a rank-2 tensor
     * @param a the right hand side rank-2 tensor
     */
    [[nodiscard]] Rank2Tensor doubledot(const Rank4Tensor &a) const;
    //*******************************
    //*** for *=  operator
    //*******************************
    /**
     * '*=' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor& operator*=(const double &a) {
        for(auto & m_val : m_Vals){
            m_val[0]*=a;m_val[1]*=a;m_val[2]*=a;
        }
        return *this;
    }
    /**
     * '*=' for rank-2 tensor
     * @param a right hand side rank-2 tensor
     */
    inline Rank2Tensor& operator*=(const Rank2Tensor &a){
        Rank2Tensor temp((*this)*a);
        *this=temp;
        return *this;
    }
    //*******************************
    //*** for /  operator
    //*******************************
    /**
     * '/' for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor operator/(const double &a) const{
        if(abs(a)<1.0e-16){
            MessagePrinter::printErrorTxt("a="+to_string(a)+" is singular for A/a operator in rank-2 tensor");
            MessagePrinter::exitPeriX();
        }
        const double inva=1.0/a;
        Rank2Tensor temp(0.0);
        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=m_Vals[i][0]*inva;
            temp.m_Vals[i][1]=m_Vals[i][1]*inva;
            temp.m_Vals[i][2]=m_Vals[i][2]*inva;
        }
        return temp;
    }
    //*******************************
    //*** for /=  operator
    //*******************************
    /**
     * '/=' operator for scalar
     * @param a right hand side scalar
     */
    inline Rank2Tensor& operator/=(const double &a){
        if(abs(a)<1.0e-16){
            MessagePrinter::printErrorTxt("a="+to_string(a)+" is singular for /= operator in rank-2 tensor");
            MessagePrinter::exitPeriX();
        }
        const double inva=1.0/a;
        for(auto &row : m_Vals){
            row[0]*=inva; row[1]*=inva; row[2]*=inva;
        }
        return *this;
    }
    //*******************************************************************
    //*** for left hand side manipulation
    //*******************************************************************
    /**
     * '*' for left hand side scalar
     * @param lhs left hand side scalar value
     * @param a right hand side rank-2 tensor
     */
    friend Rank2Tensor operator*(const double &lhs,const Rank2Tensor &a);
    //*** for left hand vector times rank-2 tensor
    /**
     * '*' operator for left hand side vector3d
     * @param lhs the left hand side vector3d
     * @param a the right hand side rank-2 tensor
     */
    friend Vector3d operator*(const Vector3d &lhs,const Rank2Tensor &a);
    //*******************************************************************
    //*** for advanced math operators
    //*******************************************************************
    /**
     * get the exponetial formula of current rank-2 tensor
    */
    [[nodiscard]] Rank2Tensor exp()const{
        Rank2Tensor I;
        I.setToIdentity();
        const Rank2Tensor a2=(*this)*(*this);
        const Rank2Tensor a3=a2*(*this);
        const Rank2Tensor a4=a3*(*this);
        const Rank2Tensor a5=a4*(*this);
        const Rank2Tensor a6=a5*(*this);
        return I
              +(*this)
              +a2*(1.0/2.0)
              +a3*(1.0/6.0)
              +a4*(1.0/24.0)
              +a5*(1.0/120.0)
              +a6*(1.0/720.0);
    }
    /**
     * get the exponential of input rank-2 tensor
     * @param a the given rank-2 tensor
    */
    friend Rank2Tensor exp(const Rank2Tensor &a);
    /**
     * get the input rank-2 tensor's (a*Rank-2) exponential's first order derivative w.r.t. scalar a
     * @param a the scalar factor
     * @param b the rank-2 tensor
    */
    friend Rank2Tensor dexp(const double &a,const Rank2Tensor &b);
    //*******************************************************************
    //*** for higher order tensor calculation
    //*******************************************************************
    /**
     * Otime(\f$\otimes\f$) between two rank-2 tensor, which will return \f$c_{ijkl}=a_{ij}b_{kl}\f$
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor otimes(const Rank2Tensor &a) const;
    /**
     * Odot (\f$\odot\f$) between two rank-2 tensor, which will return \f$c_{ijkl}=\frac{1}{2}(a_{ik}b_{jl}+a_{il}b_{jk})\f$
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor odot(const Rank2Tensor &a) const;

    /**
     * IJLK times (\f$IJ\otimes lk\f$) operator
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor IJxLK(const Rank2Tensor &a) const;

    /**
     * IKJL times (\f$Ik\otimes Jl\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor IKxJL(const Rank2Tensor &a) const;
    /**
     * IKLJ times (\f$Ik\otimes lJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor IKxLJ(const Rank2Tensor &a) const;

    /**
     * ILJK times (\f$Il\otimes Jk\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor ILxJK(const Rank2Tensor &a) const;
    /**
     * ILKJ times (\f$Il\otimes kJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor ILxKJ(const Rank2Tensor &a) const;

    /**
     * JIKL times (\f$Ji\otimes Kl\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JIxKL(const Rank2Tensor &a) const;
    /**
     * JILK times (\f$Ji\otimes lK\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JIxLK(const Rank2Tensor &a) const;

    /**
     * JKIL times (\f$Jk\otimes Il\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JKxIL(const Rank2Tensor &a) const;
    /**
     * JKLI times (\f$Jk\otimes lI\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JKxLI(const Rank2Tensor &a) const;

    /**
     * JLIK times (\f$Jl\otimes Ik\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JLxIK(const Rank2Tensor &a) const;
    /**
     * JLKI times (\f$Jl\otimes Ki\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor JLxKI(const Rank2Tensor &a) const;

    /**
     * KIJL times (\f$ki\otimes JL\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KIxJL(const Rank2Tensor &a) const;
    /**
     * KILJ times (\f$ki\otimes LJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KIxLJ(const Rank2Tensor &a) const;

    /**
     * KJIL times (\f$kj\otimes IL\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KJxIL(const Rank2Tensor &a) const;
    /**
     * KJLI times (\f$kj\otimes LI\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KJxLI(const Rank2Tensor &a) const;

    /**
     * KLIJ times (\f$kl\otimes IJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KLxIJ(const Rank2Tensor &a) const;
    /**
     * KLJI times (\f$kl\otimes JI\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor KLxJI(const Rank2Tensor &a) const;

    /**
     * LIKJ times (\f$li\otimes KJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LIxKJ(const Rank2Tensor &a) const;
    /**
     * LIJK times (\f$li\otimes JK\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LIxJK(const Rank2Tensor &a) const;

    /**
     * LJIK times (\f$lj\otimes IK\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LJxIK(const Rank2Tensor &a) const;
    /**
     * LJKI times (\f$lj\otimes KI\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LJxKI(const Rank2Tensor &a) const;

    /**
     * LKIJ times (\f$lk\otimes IJ\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LKxIJ(const Rank2Tensor &a) const;
    /**
     * LKJI times (\f$lk\otimes JI\f$)
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor LKxJI(const Rank2Tensor &a) const;
    //**********************************************************************
    //*** for general settings
    //**********************************************************************
    /**
     * set the elements of current rank-2 tensor to be zero
     */
    inline void setToZeros(){
        std::fill_n(data(), 9, 0.0);
    }
    /**
     * set current rannk-2 tensor to be an identitiy tensor, where \f$a_{ij}=\delta_{ij}\f$.
     */
    inline void setToIdentity(){
        m_Vals[0][0]=1.0; m_Vals[0][1]=0.0; m_Vals[0][2]=0.0;
        m_Vals[1][0]=0.0; m_Vals[1][1]=1.0; m_Vals[1][2]=0.0;
        m_Vals[2][0]=0.0; m_Vals[2][1]=0.0; m_Vals[2][2]=1.0;
    }
    /**
     * set current rank-2 tensor to be a random one
     */
    inline void setToRandom(){
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0,1.0);
        for(int n=0;n<9;++n) data()[n]=dist(gen);
    }
    /**
     * cross dot \f$\otimes\f$ for two vector(from STL), set current one to \f$c_{ij}=a_{i}\times b_{j}\f$.
     * @param a vector<double> for 1st dimension
     * @param b vector<double> for 2nd dimension
     */
    inline void setFromVectorDyad(const vector<double> &a,const vector<double> &b){
        if(a.size()<3 || b.size()<3){
            MessagePrinter::printErrorTxt("vector size is smaller than 3 in Rank2Tensor::setFromVectorDyad");
            MessagePrinter::exitPeriX();
        }
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                m_Vals[i][j]=a[i]*b[j];
            }
        }
    }
    /**
     * fill up current rank-2 tensor from the dyad of two vector
     * @param a the first vector
     * @param b the second vector
     */
    inline void setFromVectorDyad(const Vector3d &a,const Vector3d &b){
        const double a1=a(1), a2=a(2), a3=a(3);
        const double b1=b(1), b2=b(2), b3=b(3);
        m_Vals[0][0]=a1*b1; m_Vals[0][1]=a1*b2; m_Vals[0][2]=a1*b3;
        m_Vals[1][0]=a2*b1; m_Vals[1][1]=a2*b2; m_Vals[1][2]=a2*b3;
        m_Vals[2][0]=a3*b1; m_Vals[2][1]=a3*b2; m_Vals[2][2]=a3*b3;
    }
    /**
     * fill up current rank-2 tensor from the displacement gradient
     * @param gradUx the gradient of Ux, namely, \f$\nabla u_{x}\f$
     */
    void setFromGradU1D(const Vector3d &gradUx);
    /**
     * fill up current rank-2 tensor from the gradient of ux and uy
     * @param gradUx the gradient of Ux, namely, \f$\nabla u_{x}\f$
     * @param gradUy the gradient of Uy, namely, \f$\nabla u_{y}\f$
     */
    void setFromGradU2D(const Vector3d &gradUx,const Vector3d &gradUy);
    /**
     * fill up current rank-2 tensor from the gradient of ux, uy, and uz
     * @param gradUx the gradient of Ux, namely, \f$\nabla u_{x}\f$
     * @param gradUy the gradient of Uy, namely, \f$\nabla u_{y}\f$
     * @param gradUz the gradient of Uz, namely, \f$\nabla u_{z}\f$
     */
    void setFromGradU3D(const Vector3d &gradUx,const Vector3d &gradUy,const Vector3d &gradUz);
    /**
     * fill up current rank-2 tensor with eular angle, which should be used for the rotation tensor
     * @param theta1 the first eular angle, the angle must be degree
     * @param theta2 the second eular angle, the angle must be degree
     * @param theta3 the third eular angle, the angle must be degree
     */
    void setRotationTensorFromEulerAngle(const double &theta1,const double &theta2,const double &theta3);

    //**********************************************************************
    //*** for some common mathematic manipulations
    //**********************************************************************
    /**
     * return the trace of a rank-2 tensor, result is \f$\sum a_{ii}\f$
     */
    [[nodiscard]] inline double trace() const{
        return m_Vals[0][0]+m_Vals[1][1]+m_Vals[2][2];
    }
    /**
     * return the determinant of the current rank-2 tensor
     */
    [[nodiscard]] inline double det() const{
        // taken from http://mathworld.wolfram.com/Determinant.html
        // Eq.10
        return m_Vals[0][0]*m_Vals[1][1]*m_Vals[2][2]
              -m_Vals[0][0]*m_Vals[1][2]*m_Vals[2][1]
              -m_Vals[0][1]*m_Vals[1][0]*m_Vals[2][2]
              +m_Vals[0][1]*m_Vals[1][2]*m_Vals[2][0]
              +m_Vals[0][2]*m_Vals[1][0]*m_Vals[2][1]
              -m_Vals[0][2]*m_Vals[1][1]*m_Vals[2][0];
    }
    /**
     * return the \f$L_{2}\f$ norm of current rank-2 tensor, result is \f$\sqrt{\sum a_{ij}^{2}}\f$
     */
    [[nodiscard]] inline double norm() const{
        return sqrt(normsq());
    }

    /**
     * return the \f$L_{2}\f$ norm^2 of current rank-2 tensor, result is \f$\sqrt{\sum a_{ij}^{2}}\f$
     */
    [[nodiscard]] inline double normsq() const{
        return m_Vals[0][0]*m_Vals[0][0] + m_Vals[0][1]*m_Vals[0][1] + m_Vals[0][2]*m_Vals[0][2]
             + m_Vals[1][0]*m_Vals[1][0] + m_Vals[1][1]*m_Vals[1][1] + m_Vals[1][2]*m_Vals[1][2]
             + m_Vals[2][0]*m_Vals[2][0] + m_Vals[2][1]*m_Vals[2][1] + m_Vals[2][2]*m_Vals[2][2];
    }
    //*** for the different invariants of stress(strain)
    /**
     * return the first invariant of current rank-2 tensor, namely, \f$I_{1}\f$.
     */
    [[nodiscard]] inline double firstInvariant() const{
        return trace();
    }
    /**
     * return the second invariant of current tensor, namely, \f$I_{2}\f$.
     */
    [[nodiscard]] inline double secondInvariant() const{
        double trAA=((*this)*(*this)).trace();
        double trA=trace();
        return 0.5*(trA*trA-trAA);
    }
    /**
     * return the third invariant of current tensor, namely, \f$I_{3}\f$.
     */
    [[nodiscard]] inline double thirdInvariant() const{
        return det();
    }
    //*** for inverse
    /**
     * return the inverse tensor of current one, namely, \f$\mathbf{A}^{-1}\f$.
     */
    [[nodiscard]] inline Rank2Tensor inverse() const{
        const double a00=m_Vals[0][0], a01=m_Vals[0][1], a02=m_Vals[0][2];
        const double a10=m_Vals[1][0], a11=m_Vals[1][1], a12=m_Vals[1][2];
        const double a20=m_Vals[2][0], a21=m_Vals[2][1], a22=m_Vals[2][2];
        const double J=det();
        if(abs(J)<1.0e-16){
            MessagePrinter::printErrorTxt("inverse operation failed for a singular rank-2 tensor !");
            MessagePrinter::exitPeriX();
        }
        const double invJ=1.0/J;
        Rank2Tensor inv(0.0);
        const double A= a11*a22-a12*a21;
        const double D=-a01*a22+a02*a21;
        const double G= a01*a12-a02*a11;
        inv.m_Vals[0][0]=A*invJ; inv.m_Vals[0][1]=D*invJ; inv.m_Vals[0][2]=G*invJ;

        const double B=-a10*a22+a12*a20;
        const double E= a00*a22-a02*a20;
        const double H=-a00*a12+a02*a10;
        inv.m_Vals[1][0]=B*invJ; inv.m_Vals[1][1]=E*invJ; inv.m_Vals[1][2]=H*invJ;

        const double C= a10*a21-a11*a20;
        const double F=-a00*a21+a01*a20;
        const double I= a00*a11-a01*a10;
        inv.m_Vals[2][0]=C*invJ; inv.m_Vals[2][1]=F*invJ; inv.m_Vals[2][2]=I*invJ;
        return inv;
    }
    /**
     * return the transpose tensor of current one, namely, \f$\mathbf{A}^{T}\f$.
     */
    [[nodiscard]] inline Rank2Tensor transpose() const{
        Rank2Tensor temp(0.0);
        temp.m_Vals[0][0]=m_Vals[0][0]; temp.m_Vals[0][1]=m_Vals[1][0]; temp.m_Vals[0][2]=m_Vals[2][0];
        temp.m_Vals[1][0]=m_Vals[0][1]; temp.m_Vals[1][1]=m_Vals[1][1]; temp.m_Vals[1][2]=m_Vals[2][1];
        temp.m_Vals[2][0]=m_Vals[0][2]; temp.m_Vals[2][1]=m_Vals[1][2]; temp.m_Vals[2][2]=m_Vals[2][2];
        return temp;
    }
    /**
     * transpose current tensor, and overwrite the original one
     */
    inline void transposed(){
        std::swap(m_Vals[0][1],m_Vals[1][0]);
        std::swap(m_Vals[0][2],m_Vals[2][0]);
        std::swap(m_Vals[1][2],m_Vals[2][1]);
    }
    /**
     * rotate the current tensor by rotation tensor r, important: the values of current tensor will be modified!!!
     * @param r the rotation tensor
     */
    void rotated(const Rank2Tensor &r) {
        Rank2Tensor temp(0.0);
        double RA[3][3];

        for(int i=0;i<3;i++){
            RA[i][0]=r.m_Vals[i][0]*m_Vals[0][0] + r.m_Vals[i][1]*m_Vals[1][0] + r.m_Vals[i][2]*m_Vals[2][0];
            RA[i][1]=r.m_Vals[i][0]*m_Vals[0][1] + r.m_Vals[i][1]*m_Vals[1][1] + r.m_Vals[i][2]*m_Vals[2][1];
            RA[i][2]=r.m_Vals[i][0]*m_Vals[0][2] + r.m_Vals[i][1]*m_Vals[1][2] + r.m_Vals[i][2]*m_Vals[2][2];
        }

        for(int i=0;i<3;i++){
            temp.m_Vals[i][0]=RA[i][0]*r.m_Vals[0][0] + RA[i][1]*r.m_Vals[0][1] + RA[i][2]*r.m_Vals[0][2];
            temp.m_Vals[i][1]=RA[i][0]*r.m_Vals[1][0] + RA[i][1]*r.m_Vals[1][1] + RA[i][2]*r.m_Vals[1][2];
            temp.m_Vals[i][2]=RA[i][0]*r.m_Vals[2][0] + RA[i][1]*r.m_Vals[2][1] + RA[i][2]*r.m_Vals[2][2];
        }

        *this=temp;
    }
    /**
     * return the deviatoric part of current rank-2 tensor
     */
    [[nodiscard]] inline Rank2Tensor dev()const{
        Rank2Tensor temp(*this);
        const double one_third_trace=trace()/3.0;
        temp.m_Vals[0][0]-=one_third_trace;
        temp.m_Vals[1][1]-=one_third_trace;
        temp.m_Vals[2][2]-=one_third_trace;
        return temp;
    }

    /**
     * calculate the max principal value of current stress/strain tensor
     * @return get the max principal value
     */
    double getMaxPrincipalValue(bool require_symmetric=true) const;

    /**
     * calculate the min principal value of current stress/strain tensor
     * @return get the min principal value
     */
    double getMinPricipalValue(bool require_symmetric=true) const;
    //**********************************************************************
    //*** for decomposition
    //**********************************************************************
    /**
     * calculate the eigen value and eigen vector for current rank-2 tensor
     * @param eigval the double array, which stores the eigen value
     * @param eigvec the rank-2 tensor, whoses column stores the related eigen vector
     */
    void calcEigenValueAndEigenVectors(double (&eigval)[3],Rank2Tensor &eigvec,bool require_symmetric=true) const;
    /**
     * calculate the eigen value and eigen vector for current rank-2 tensor
     * @param eigval the double array, which stores the eigen value
     * @param eigvec the rank-2 tensor, whoses column stores the related eigen vector
     */
    Rank4Tensor calcPositiveProjTensor(double (&eigval)[3],Rank2Tensor &eigvec,bool require_symmetric=true) const;
    /**
     * calculate the positive projection tensor (rank-4 tensor) based on current rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor getPositiveProjectionTensor(bool require_symmetric=true) const;
    //**********************************************************************
    //*** for decomposition
    //**********************************************************************
    /**
     * print all the elements to the terminal
     */
    inline void print() const{
        printf("*** %14.6e ,%14.6e ,%14.6e ***\n",m_Vals[0][0],m_Vals[0][1],m_Vals[0][2]);
        printf("*** %14.6e ,%14.6e ,%14.6e ***\n",m_Vals[1][0],m_Vals[1][1],m_Vals[1][2]);
        printf("*** %14.6e ,%14.6e ,%14.6e ***\n",m_Vals[2][0],m_Vals[2][1],m_Vals[2][2]);
    }



private:
    inline static void checkIndex(int i,int j){
        if(i<1||i>3 || j<1||j>3 ){
            MessagePrinter::printErrorTxt("i="+to_string(i)+" or j="+to_string(j)+" is out of range when you call a rank-2 tensor");
            MessagePrinter::exitPeriX();
        }
    }
    [[nodiscard]] inline double* data() noexcept { return &m_Vals[0][0]; }
    [[nodiscard]] inline const double* data() const noexcept { return &m_Vals[0][0]; }
    double m_Vals[3][3];/**< matrix for the rank-2 tensor */
};
