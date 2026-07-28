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

#include "MathUtils/Rank4Tensor.h"

namespace {
    constexpr double kElasticTol = 1.0e-15;

    inline void copyRank2ToArray(const Rank2Tensor &A, double out[3][3]){
        out[0][0]=A(1,1); out[0][1]=A(1,2); out[0][2]=A(1,3);
        out[1][0]=A(2,1); out[1][1]=A(2,2); out[1][2]=A(2,3);
        out[2][0]=A(3,1); out[2][1]=A(3,2); out[2][2]=A(3,3);
    }

    inline void setMinorMajorSymmetricFamily(double C[3][3][3][3], int i, int j, int k, int l, double v){
        C[i][j][k][l] = v;
        C[j][i][k][l] = v;
        C[i][j][l][k] = v;
        C[j][i][l][k] = v;
        C[k][l][i][j] = v;
        C[l][k][i][j] = v;
        C[k][l][j][i] = v;
        C[l][k][j][i] = v;
    }

    inline void checkElasticDenominator(double denom, const char *msg){
        if(std::abs(denom) < kElasticTol){
            MessagePrinter::printErrorTxt(msg);
            MessagePrinter::exitPeriX();
        }
    }
} // namespace

Rank4Tensor::Rank4Tensor(){
    setToZeros();
}
Rank4Tensor::Rank4Tensor(double val){
    std::fill_n(data(), 81, val);
}
Rank4Tensor::Rank4Tensor(const Rank4Tensor &a){
    std::copy_n(a.data(), 81, data());
}
Rank4Tensor::Rank4Tensor(const InitMethod &method){
    switch(method){
        case ZERO:
            setToZeros();
            break;
        case IDENTITY:
            setToIdentity();
            break;
        case IDENTITY4:
            setToIdentity4();
            break;
        case IDENTITY4TRANS:
            setIdentity4Transpose();
            break;
        case IDENTITY4SYMMETRIC:
            setToIdentity4Symmetric();
            break;
        case RANDOM:
            setToRandom();
            break;
        default:
            MessagePrinter::printErrorTxt("unsupported initialize method in rank-4 tensor");
            MessagePrinter::exitPeriX();
    }
}
//************************************************************************
double Rank4Tensor::getVoigtComponent(const int &i,const int &j)const{
    int a,b,c,d;
    voigtPair(i,a,b);
    voigtPair(j,c,d);
    return m_vals[a][b][c][d];
}
//************************************************************************
double& Rank4Tensor::voigtComponent(const int &i,const int &j){
    int a,b,c,d;
    voigtPair(i,a,b);
    voigtPair(j,c,d);
    return m_vals[a][b][c][d];
}
//************************************************************************
Rank4Tensor Rank4Tensor::operator*(const Rank2Tensor &a) const{
    Rank4Tensor temp;
    const double A00=a(1,1), A01=a(1,2), A02=a(1,3);
    const double A10=a(2,1), A11=a(2,2), A12=a(2,3);
    const double A20=a(3,1), A21=a(3,2), A22=a(3,3);

    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            for(int k=0;k<3;++k){
                temp.m_vals[i][j][k][0] = m_vals[i][j][k][0]*A00 + m_vals[i][j][k][1]*A10 + m_vals[i][j][k][2]*A20;
                temp.m_vals[i][j][k][1] = m_vals[i][j][k][0]*A01 + m_vals[i][j][k][1]*A11 + m_vals[i][j][k][2]*A21;
                temp.m_vals[i][j][k][2] = m_vals[i][j][k][0]*A02 + m_vals[i][j][k][1]*A12 + m_vals[i][j][k][2]*A22;
            }
        }
    }
    return temp;
}
Rank2Tensor Rank4Tensor::doubledot(const Rank2Tensor &a) const{
    Rank2Tensor temp(0.0);
    const double A00=a(1,1), A01=a(1,2), A02=a(1,3);
    const double A10=a(2,1), A11=a(2,2), A12=a(2,3);
    const double A20=a(3,1), A21=a(3,2), A22=a(3,3);

    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            temp(i+1,j+1) = m_vals[i][j][0][0]*A00 + m_vals[i][j][0][1]*A01 + m_vals[i][j][0][2]*A02
                          + m_vals[i][j][1][0]*A10 + m_vals[i][j][1][1]*A11 + m_vals[i][j][1][2]*A12
                          + m_vals[i][j][2][0]*A20 + m_vals[i][j][2][1]*A21 + m_vals[i][j][2][2]*A22;
        }
    }
    return temp;
}
Rank4Tensor Rank4Tensor::doubledot(const Rank4Tensor &a) const{
    Rank4Tensor temp;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            for(int k=0;k<3;++k){
                for(int l=0;l<3;++l){
                    temp.m_vals[i][j][k][l] = m_vals[i][j][0][0]*a.m_vals[0][0][k][l] + m_vals[i][j][0][1]*a.m_vals[0][1][k][l] + m_vals[i][j][0][2]*a.m_vals[0][2][k][l]
                                            + m_vals[i][j][1][0]*a.m_vals[1][0][k][l] + m_vals[i][j][1][1]*a.m_vals[1][1][k][l] + m_vals[i][j][1][2]*a.m_vals[1][2][k][l]
                                            + m_vals[i][j][2][0]*a.m_vals[2][0][k][l] + m_vals[i][j][2][1]*a.m_vals[2][1][k][l] + m_vals[i][j][2][2]*a.m_vals[2][2][k][l];
                }
            }
        }
    }
    return temp;
}
//*******************************************************************
//*** for left hand side manipulation
//*******************************************************************
Rank4Tensor operator*(const double &lhs,const Rank4Tensor &a){
    Rank4Tensor temp;
    const double *src=a.data();
    double *dst=temp.data();
    for(int n=0;n<81;++n) dst[n] = lhs * src[n];
    return temp;
}
Rank4Tensor operator*(const Rank2Tensor &lhs,const Rank4Tensor &a){
    // C_ijkl=B_ip*A_pjkl
    Rank4Tensor temp;
    const double L00=lhs(1,1), L01=lhs(1,2), L02=lhs(1,3);
    const double L10=lhs(2,1), L11=lhs(2,2), L12=lhs(2,3);
    const double L20=lhs(3,1), L21=lhs(3,2), L22=lhs(3,3);

    for(int j=0;j<3;++j){
        for(int k=0;k<3;++k){
            for(int l=0;l<3;++l){
                temp.m_vals[0][j][k][l] = L00*a.m_vals[0][j][k][l] + L01*a.m_vals[1][j][k][l] + L02*a.m_vals[2][j][k][l];
                temp.m_vals[1][j][k][l] = L10*a.m_vals[0][j][k][l] + L11*a.m_vals[1][j][k][l] + L12*a.m_vals[2][j][k][l];
                temp.m_vals[2][j][k][l] = L20*a.m_vals[0][j][k][l] + L21*a.m_vals[1][j][k][l] + L22*a.m_vals[2][j][k][l];
            }
        }
    }
    return temp;
}
//*******************************************************************
//*** fill up the rank-4 tensor
//*******************************************************************
void Rank4Tensor::setFromLameAndG(const double &Lame,const double &G){
    // taken from: https://en.wikipedia.org/wiki/Linear_elasticity
    // C_ijkl = Lame*de_ij*de_kl + G*(de_ik*de_jl + de_il*de_jk)
    setToZeros();
    const double diag = Lame + 2.0*G;

    setMinorMajorSymmetricFamily(m_vals, 0,0,0,0, diag);
    setMinorMajorSymmetricFamily(m_vals, 1,1,1,1, diag);
    setMinorMajorSymmetricFamily(m_vals, 2,2,2,2, diag);

    setMinorMajorSymmetricFamily(m_vals, 0,0,1,1, Lame);
    setMinorMajorSymmetricFamily(m_vals, 0,0,2,2, Lame);
    setMinorMajorSymmetricFamily(m_vals, 1,1,2,2, Lame);

    setMinorMajorSymmetricFamily(m_vals, 1,2,1,2, G);
    setMinorMajorSymmetricFamily(m_vals, 0,2,0,2, G);
    setMinorMajorSymmetricFamily(m_vals, 0,1,0,1, G);
}
void Rank4Tensor::setFromEAndNu(const double &E,const double &Nu){
    // taken from:  https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    checkElasticDenominator(1.0+Nu, "invalid poisson ratio: 1+nu is singular in Rank4Tensor::setFromEAndNu");
    checkElasticDenominator(1.0-2.0*Nu, "invalid poisson ratio: 1-2*nu is singular in Rank4Tensor::setFromEAndNu");
    const double Lame = E*Nu/((1.0+Nu)*(1.0-2.0*Nu));
    const double G    = E/(2.0*(1.0+Nu));
    setFromLameAndG(Lame,G);
}
void Rank4Tensor::setFromKAndG(const double &K,const double &G){
    // taken from:  https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    const double Lame = K - 2.0*G/3.0;
    setFromLameAndG(Lame,G);
}
void Rank4Tensor::setFromSymmetric9(const vector<double> &vec){
    if(vec.size()<9){
        MessagePrinter::printErrorTxt("Symmetric9 fill method for rank-4 tensor need at least 9 elements!!!");
        MessagePrinter::exitPeriX();
    }
    //C1111  C1122  C1133   0     0     0
    // 0     C2222  C2233   0     0     0
    // 0      0     C3333   0     0     0
    // 0      0      0     C2323  0     0
    // 0      0      0      0    C1313  0
    // 0      0      0      0     0    C1212

    // C1111,C1122,C1133,C2222,C2233,C3333,C2323,C1313,C1212
    // C11,  C12,  C13,  C22,  C23,  C33,  C44,  C55,  C66
    setToZeros();
    setMinorMajorSymmetricFamily(m_vals, 0,0,0,0, vec[0]);
    setMinorMajorSymmetricFamily(m_vals, 0,0,1,1, vec[1]);
    setMinorMajorSymmetricFamily(m_vals, 0,0,2,2, vec[2]);
    setMinorMajorSymmetricFamily(m_vals, 1,1,1,1, vec[3]);
    setMinorMajorSymmetricFamily(m_vals, 1,1,2,2, vec[4]);
    setMinorMajorSymmetricFamily(m_vals, 2,2,2,2, vec[5]);
    setMinorMajorSymmetricFamily(m_vals, 1,2,1,2, vec[6]);
    setMinorMajorSymmetricFamily(m_vals, 0,2,0,2, vec[7]);
    setMinorMajorSymmetricFamily(m_vals, 0,1,0,1, vec[8]);
}
void Rank4Tensor::setToOrthotropic(const vector<double> &vec){
    if(vec.size()<9){
        MessagePrinter::printErrorTxt("Orthotropic fill method for rank-4 tensor need at least 9 elements!!!");
        MessagePrinter::exitPeriX();
    }
    //C1111  C1122  C1133   0     0     0
    // 0     C2222  C2233   0     0     0
    // 0      0     C3333   0     0     0
    // 0      0      0     C2323  0     0
    // 0      0      0      0    C1313  0
    // 0      0      0      0     0    C1212

    // C1111,C1122,C1133,C2222,C2233,C3333,C2323,C1313,C1212
    // C11,  C12,  C13,  C22,  C23,  C33,  C44,  C55,  C66
    setFromSymmetric9(vec);
}
//**********************************************************************
//*** for advanced mathematic manipulation
//**********************************************************************
Rank4Tensor Rank4Tensor::rotate(const Rank2Tensor &rot) const{
    //C_ijkl=C_mnpq*R_im*R_jn*R_kp*R_lq
    double R[3][3];
    copyRank2ToArray(rot, R);

    double t1[3][3][3][3];
    double t2[3][3][3][3];
    double t3[3][3][3][3];

    for(int i=0;i<3;++i)
        for(int n=0;n<3;++n)
            for(int p=0;p<3;++p)
                for(int q=0;q<3;++q)
                    t1[i][n][p][q] = R[i][0]*m_vals[0][n][p][q] + R[i][1]*m_vals[1][n][p][q] + R[i][2]*m_vals[2][n][p][q];

    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int p=0;p<3;++p)
                for(int q=0;q<3;++q)
                    t2[i][j][p][q] = R[j][0]*t1[i][0][p][q] + R[j][1]*t1[i][1][p][q] + R[j][2]*t1[i][2][p][q];

    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                for(int q=0;q<3;++q)
                    t3[i][j][k][q] = R[k][0]*t2[i][j][0][q] + R[k][1]*t2[i][j][1][q] + R[k][2]*t2[i][j][2][q];

    Rank4Tensor out;
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    out.m_vals[i][j][k][l] = R[l][0]*t3[i][j][k][0] + R[l][1]*t3[i][j][k][1] + R[l][2]*t3[i][j][k][2];

    return out;
}
void Rank4Tensor::rotated(const Rank2Tensor &rot){
    *this = rotate(rot);
}
Rank4Tensor Rank4Tensor::pushForward(const Rank2Tensor &F) const{
    // push forward the jacobian from reference one to the current configuration or similar operation
    // new Ran-4 tensor c_ijkl=F_iA*F_jB*F_kC*F_lD*C_ABCD
    double FF[3][3];
    copyRank2ToArray(F, FF);

    double t1[3][3][3][3];
    double t2[3][3][3][3];
    double t3[3][3][3][3];

    for(int i=0;i<3;++i)
        for(int B=0;B<3;++B)
            for(int C=0;C<3;++C)
                for(int D=0;D<3;++D)
                    t1[i][B][C][D] = FF[i][0]*m_vals[0][B][C][D] + FF[i][1]*m_vals[1][B][C][D] + FF[i][2]*m_vals[2][B][C][D];

    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int C=0;C<3;++C)
                for(int D=0;D<3;++D)
                    t2[i][j][C][D] = FF[j][0]*t1[i][0][C][D] + FF[j][1]*t1[i][1][C][D] + FF[j][2]*t1[i][2][C][D];

    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                for(int D=0;D<3;++D)
                    t3[i][j][k][D] = FF[k][0]*t2[i][j][0][D] + FF[k][1]*t2[i][j][1][D] + FF[k][2]*t2[i][j][2][D];

    Rank4Tensor out;
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    out.m_vals[i][j][k][l] = FF[l][0]*t3[i][j][k][0] + FF[l][1]*t3[i][j][k][1] + FF[l][2]*t3[i][j][k][2];

    return out;
}
Rank4Tensor Rank4Tensor::conjPushForward(const Rank2Tensor &F) const{
    // calculate the conjugate rank-4 tensor regulate by F
    // C_ijkl=F_im*C_mjnl*F_kn
    double FF[3][3];
    copyRank2ToArray(F, FF);

    double t1[3][3][3][3];
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int n=0;n<3;++n)
                for(int l=0;l<3;++l)
                    t1[i][j][n][l] = FF[i][0]*m_vals[0][j][n][l] + FF[i][1]*m_vals[1][j][n][l] + FF[i][2]*m_vals[2][j][n][l];

    Rank4Tensor out;
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    out.m_vals[i][j][k][l] = FF[k][0]*t1[i][j][0][l] + FF[k][1]*t1[i][j][1][l] + FF[k][2]*t1[i][j][2][l];

    return out;
}
