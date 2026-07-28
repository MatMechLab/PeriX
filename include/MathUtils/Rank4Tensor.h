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
//+++ Function: Implement rank-4 tensor class for the common
//+++           tensor manipulation in AsFem
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "MathUtils/Vector3d.h"
#include "MathUtils/Rank2Tensor.h"

#include "Utils/MessagePrinter.h"

class Rank2Tensor;

using std::fill;
using std::sqrt;
using std::abs;

/**
 * This class implement the general manipulation for rank-2 tensor.
 */
class Rank4Tensor{
public:
    /**
     * enum for different initializing methods
     */
    enum InitMethod{
        ZERO,
        IDENTITY,
        IDENTITY4,
        IDENTITY4TRANS,
        IDENTITY4SYMMETRIC,
        RANDOM
    };
public:
    /**
     * constructor
     */
    Rank4Tensor();
    /**
     * constructor with default value
     * @param val the default value
     */
    explicit Rank4Tensor(double val);
    /**
     * constructor with another rank-4 tensor
     * @param a the right hand side rank-4 tensor
     */
    Rank4Tensor(const Rank4Tensor &a);
    /**
     * constructor with initial method
     * @param method the specific initial method
     */
    Rank4Tensor(const InitMethod &method);
    /**
     * deconstructor
     */
    ~Rank4Tensor() = default;

    //**********************************************************************
    //*** for operator override
    //**********************************************************************
    //*******************************
    //*** for ()  operator
    //*******************************
    /**
     * () operator for rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param i i index, start from 1 instead of 0 !!!
     * @param j j index, start from 1 instead of 0 !!!
     * @param k k index, start from 1 instead of 0 !!!
     * @param l l index, start from 1 instead of 0 !!!
     */
    [[nodiscard]] inline const double& operator()(const int &i,const int &j,const int &k,const int &l) const{
        checkIndex(i,j,k,l);
        return m_vals[i-1][j-1][k-1][l-1];
    }
    /**
     * () operator for rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param i i index, start from 1 instead of 0 !!!
     * @param j j index, start from 1 instead of 0 !!!
     * @param k k index, start from 1 instead of 0 !!!
     * @param l l index, start from 1 instead of 0 !!!
     */
    [[nodiscard]] inline double& operator()(const int &i,const int &j,const int &k,const int &l){
        checkIndex(i,j,k,l);
        return m_vals[i-1][j-1][k-1][l-1];
    }
    /**
     * get the voigt component of current rank-4 tensor
     * @param i i-index for 1st dimension
     * @param j j-index for 2nd dimension
     */
    [[nodiscard]] double getVoigtComponent(const int &i,const int &j)const;

    /**
     * return the reference via the voigt component of current rank-4 tensor
     * @param i i-index for 1st dimension
     * @param j j-index for 2nd dimension
     */
    double& voigtComponent(const int &i,const int &j);
    /**
     * return the C_iJkL*N,J*N,L value for jacobian matrix calculation
     * @param i the 1st dimension index
     * @param k the 2nd dimension index
     * @param grad_test the test shape function's gradient
     * @param grad_trial the trial shape function's gradient
     */
    [[nodiscard]] inline double getIKComponent(const int &i,const int &k,const Vector3d &grad_test,const Vector3d &grad_trial)const{
        checkIKIndex(i, "i");
        checkIKIndex(k, "k");
        const int ii=i-1;
        const int kk=k-1;
        const double gt1=grad_test(1), gt2=grad_test(2), gt3=grad_test(3);
        const double gr1=grad_trial(1), gr2=grad_trial(2), gr3=grad_trial(3);

        return (m_vals[ii][0][kk][0]*gt1 + m_vals[ii][1][kk][0]*gt2 + m_vals[ii][2][kk][0]*gt3)*gr1
             + (m_vals[ii][0][kk][1]*gt1 + m_vals[ii][1][kk][1]*gt2 + m_vals[ii][2][kk][1]*gt3)*gr2
             + (m_vals[ii][0][kk][2]*gt1 + m_vals[ii][1][kk][2]*gt2 + m_vals[ii][2][kk][2]*gt3)*gr3;
    }
    //*******************************
    //*** for =  operator
    //*******************************
    /**
     * = operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor& operator=(const double &a){
        std::fill_n(data(), 81, a);
        return *this;
    }
    /**
     * = operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand rank-4 tensor
     */
    inline Rank4Tensor& operator=(const Rank4Tensor &a){
        if(this != &a){
            std::copy_n(a.data(), 81, data());
        }
        return *this;
    }
    //*******************************
    //*** for + operator
    //*******************************
    /**
     * + operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor operator+(const double &a) const{
        Rank4Tensor temp;
        const double *src=data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=src[n]+a;
        return temp;
    }
    /**
     * + operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand rank-4 tensor
     */
    inline Rank4Tensor operator+(const Rank4Tensor &a) const{
        Rank4Tensor temp;
        const double *lhs=data();
        const double *rhs=a.data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=lhs[n]+rhs[n];
        return temp;
    }
    //*******************************
    //*** for += operator
    //*******************************
    /**
     * += operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor& operator+=(const double &a){
        double *vals=data();
        for(int n=0;n<81;++n) vals[n]+=a;
        return *this;
    }
    /**
     * += operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side rank-4 tensor
     */
    inline Rank4Tensor& operator+=(const Rank4Tensor &a){
        double *lhs=data();
        const double *rhs=a.data();
        for(int n=0;n<81;++n) lhs[n]+=rhs[n];
        return *this;
    }
    //*******************************
    //*** for - operator
    //*******************************
    /**
     * - operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor operator-(const double &a) const{
        Rank4Tensor temp;
        const double *src=data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=src[n]-a;
        return temp;
    }
    /**
     * - operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand rank-4 tensor
     */
    inline Rank4Tensor operator-(const Rank4Tensor &a) const{
        Rank4Tensor temp;
        const double *lhs=data();
        const double *rhs=a.data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=lhs[n]-rhs[n];
        return temp;
    }
    //*******************************
    //*** for -= operator
    //*******************************
    /**
     * -= operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor& operator-=(const double &a){
        double *vals=data();
        for(int n=0;n<81;++n) vals[n]-=a;
        return *this;
    }
    /**
     * -= operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side rank-4 tensor
     */
    inline Rank4Tensor& operator-=(const Rank4Tensor &a){
        double *lhs=data();
        const double *rhs=a.data();
        for(int n=0;n<81;++n) lhs[n]-=rhs[n];
        return *this;
    }
    //*******************************
    //*** for * operator
    //*******************************
    /**
     * * operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor operator*(const double &a) const{
        Rank4Tensor temp;
        const double *src=data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=src[n]*a;
        return temp;
    }
    /**
     * * operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}=\mathbb{C}_{ijkm}\mathbf{A}_{ml}\f$
     * @param a right hand side ran-2 tensor
     */
    Rank4Tensor operator*(const Rank2Tensor &a) const;
    /**
     * double dot : between a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$ and rank-2 tensor
     * @param a right hand side rank-2 tensor
     */
    [[nodiscard]] Rank2Tensor doubledot(const Rank2Tensor &a) const;
    /**
     * double dot : between a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$ and another rank-4 tensor
     * return \f$\mathbb{C}_{ijkl}=\mathbb{A}_{ijmn}\mathbb{B}_{mnkl}\f$
     * @param a right hand side rank-4 tensor
     */
    [[nodiscard]] Rank4Tensor doubledot(const Rank4Tensor &a) const;
    //*******************************
    //*** for *= operator
    //*******************************
    /**
     * *= operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor& operator*=(const double &a){
        double *vals=data();
        for(int n=0;n<81;++n) vals[n]*=a;
        return *this;
    }
    //*******************************
    //*** for / operator
    //*******************************
    /**
     * / operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor operator/(const double &a) const{
        checkDivisor(a, "/");
        const double inva=1.0/a;
        Rank4Tensor temp;
        const double *src=data();
        double *dst=temp.data();
        for(int n=0;n<81;++n) dst[n]=src[n]*inva;
        return temp;
    }
    //*******************************
    //*** for /= operator
    //*******************************
    /**
     * /= operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param a right hand side scalar
     */
    inline Rank4Tensor& operator/=(const double &a){
        checkDivisor(a, "/=");
        const double inva=1.0/a;
        double *vals=data();
        for(int n=0;n<81;++n) vals[n]*=inva;
        return *this;
    }
    //*******************************************************************
    //*** for left hand side manipulation
    //*******************************************************************
    /**
     * * operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param lhs left hand side scalar
     * @param a right hand side rank-4 tensor
     */
    friend Rank4Tensor operator*(const double &lhs,const Rank4Tensor &a);
    /**
     * * operator to a rank-4 tensor \f$\mathbb{C}_{ijkl}\f$
     * @param lhs left hand side rank-2 tensor
     * @param a right hand side rank-4 tensor
     */
    friend Rank4Tensor operator*(const Rank2Tensor &lhs,const Rank4Tensor &a);
    //**********************************************************************
    //*** for general settings
    //**********************************************************************
    /**
     * set current rank-4 tensor to 0
     */
    inline void setToZeros(){
        std::fill_n(data(), 81, 0.0);
    }
    /**
     * set current rank-4 tensor to idendity
     */
    inline void setToIdentity(){
        setToZeros();
        m_vals[0][0][0][0]=1.0;
        m_vals[1][1][1][1]=1.0;
        m_vals[2][2][2][2]=1.0;
    }
    /**
     * set current rank-4 tensor to rank-4 identity
     */
    inline void setToIdentity4(){
        // maps a rank2 tensor to itself(no symmetric consideriation here),i.e. Iden4:rank2=rank2
        setToZeros();
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                m_vals[i][j][i][j]=1.0;
            }
        }
    }
    /**
     * set current rank-4 tensor to rank-4 transposed identity
     */
    inline void setIdentity4Transpose(){
        // maps a rank-2 tensor to its transpose, A^{T}=I4:A
        setToZeros();
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                m_vals[i][j][j][i]=1.0;
            }
        }
    }
    /**
     * set current rank-4 tensor to symmetric identity rank-4 tensor
     */
    inline void setToIdentity4Symmetric(){
        // symmetric fourth-order tensor
        setToZeros();
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                m_vals[i][j][i][j]+=0.5;
                m_vals[i][j][j][i]+=0.5;
            }
        }
    }
    /**
     * set current rank-4 tensor to random values
     */
    inline void setToRandom(){
        static thread_local std::mt19937_64 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0,1.0);
        for(int n=0;n<81;++n) data()[n]=dist(gen);
    }
    /**
     * set rank-4 tensor from lamme constant and shear modulus
     * @param Lame first Lame constant
     * @param G shear modulus
     */
    void setFromLameAndG(const double &Lame,const double &G);
    /**
     * set rank-4 tensor from Youngs modulus and poisson ratio
     * @param E Youngs modulus
     * @param Nu Poisson ratio
     */
    void setFromEAndNu(const double &E,const double &Nu);
    /**
     * set rank-4 tensor from bulk modulus and shear modulus
     * @param E Bulk modulus
     * @param Nu Shear modulus
     */
    void setFromKAndG(const double &K,const double &G);
    /**
     * set rank-4 tensor from 9-component
     * @param vec 9-component for rank-4 tensor
     */
    void setFromSymmetric9(const vector<double> &vec);
    /**
     * set rank-4 tensor to Orthotropic
     * @param vec 9-component for rank-4 tensor
     */
    void setToOrthotropic(const vector<double> &vec);
    //**********************************************************************
    //*** for advanced mathematic manipulation
    //**********************************************************************
    /**
     * rotate the current a rank-4 tensor by a rank-2 rotation tensor, this will return a new and rotated
     * rank-4 tensor, the original one will not be changed
     * @param rotate right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor rotate(const Rank2Tensor &rotate) const;
    /**
     * rotate the current a rank-4 tensor by a rank-2 rotation tensor, current tensor value will be changed
     * @param rot right hand side rank-2 tensor
     */
    void rotated(const Rank2Tensor &rot);
    /**
     * pushforward current rank-4 tensor by a rank-2 rotation tensor F
     * @param F right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor pushForward(const Rank2Tensor &F) const;
    /**
     * conjugate pushforward current rank-4 tensor by a rank-2 rotation tensor F,
     * the conjugate tensor is given by:
     * C_ijkl=F_im*C_mjnl*F_kn
     * @param F right hand side rank-2 tensor
     */
    [[nodiscard]] Rank4Tensor conjPushForward(const Rank2Tensor &F) const;

private:
    /**
     * check the index's validation
     * @param i index, start from 1
     * @param j index, start from 1
     * @param k index, start from 1
     * @param l index, start from 1
     */
    inline static void checkIndex(int i,int j,int k,int l){
        if(i<1||i>3 || j<1||j>3 || k<1||k>3 || l<1||l>3){
            MessagePrinter::printErrorTxt("your i or j or k or l is out of range when you access a rank-4 tensor");
            MessagePrinter::exitPeriX();
        }
    }

    inline static void checkIKIndex(int idx,const char *name){
        if(idx<1||idx>3){
            MessagePrinter::printErrorTxt("your "+std::string(name)+"(="+std::to_string(idx)+") is out of range when you call getIKComponent");
            MessagePrinter::exitPeriX();
        }
    }

    inline static void checkDivisor(double a,const char *op){
        if(std::abs(a)<1.0e-16){
            MessagePrinter::printErrorTxt("a="+std::to_string(a)+" is singular for "+std::string(op)+" operator in rank-4 tensor");
            MessagePrinter::exitPeriX();
        }
    }

    /**
     * get the voigt position from given index
     * @param idx voigt index
     * @param a first index of voigt notation
     * @param b second index of voigt notation
     */
    inline static void voigtPair(int idx,int &a,int &b){
        switch(idx){
            case 1: a=0; b=0; return;
            case 2: a=1; b=1; return;
            case 3: a=2; b=2; return;
            case 4: a=1; b=2; return;
            case 5: a=2; b=0; return;
            case 6: a=0; b=1; return;
            default:
                MessagePrinter::printErrorTxt("invalid voigt index="+std::to_string(idx)+" in rank-4 tensor");
                MessagePrinter::exitPeriX();
        }
    }

    /**
     * get the pointer of the data
     * @return pointer of the data
     */
    inline double* data() noexcept { return &m_vals[0][0][0][0]; }
    /**
     * get the const pointer of the data
     * @return const pointer of the data
     */
    inline const double* data() const noexcept { return &m_vals[0][0][0][0]; }
private:
    double m_vals[3][3][3][3];/**< the tensor's matrix */

};
