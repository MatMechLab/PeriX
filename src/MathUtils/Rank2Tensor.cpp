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

#include <algorithm>
#include <cmath>
#include "MathUtils/Rank2Tensor.h"
#include "Eigen/Eigen"

namespace {
    constexpr double kSymmetryTol=1.0e-12;

    inline Eigen::Matrix3d toEigenMatrix(const Rank2Tensor &A){
        Eigen::Matrix3d M;
        M << A(1,1), A(1,2), A(1,3),
             A(2,1), A(2,2), A(2,3),
             A(3,1), A(3,2), A(3,3);
        return M;
    }

    inline bool isNearlySymmetric(const Eigen::Matrix3d &M, const double tol=kSymmetryTol){
        const double skew=(M-M.transpose()).norm();
        const double scale=std::max(1.0, M.norm());
        return skew <= tol*scale;
    }

    inline void fillFromEigenVectors(const Eigen::Matrix3d &vecs, Rank2Tensor &eigvec){
        for(int i=0;i<3;i++){
            Eigen::Vector3d v=vecs.col(i);
            const double n=v.norm();
            if(n>0.0) v/=n;
            eigvec(1,i+1)=v(0);
            eigvec(2,i+1)=v(1);
            eigvec(3,i+1)=v(2);
        }
    }
    inline Eigen::Matrix3d getSpectralMatrix(const Rank2Tensor &A, const char *op, const bool require_symmetric){
        const Eigen::Matrix3d M=toEigenMatrix(A);
        if(isNearlySymmetric(M)){
            return M;
        }
        if(require_symmetric){
            MessagePrinter::printErrorTxt(std::string("Rank2Tensor::")+op+" requires a symmetric tensor input");
            MessagePrinter::exitPeriX();
        }
        return 0.5*(M+M.transpose());
    }

    inline void computeSpectralSystem(const Rank2Tensor &A,
                                      double (&eigval)[3],
                                      Rank2Tensor &eigvec,
                                      const char *op,
                                      const bool require_symmetric){
        const Eigen::Matrix3d M=getSpectralMatrix(A, op, require_symmetric);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
        if(solver.info()!=Eigen::Success){
            MessagePrinter::printErrorTxt(std::string("SelfAdjointEigenSolver failed in Rank2Tensor::")+op);
            MessagePrinter::exitPeriX();
        }
        const Eigen::Vector3d vals=solver.eigenvalues();
        eigval[0]=vals(0);
        eigval[1]=vals(1);
        eigval[2]=vals(2);
        fillFromEigenVectors(solver.eigenvectors(), eigvec);
    }

    inline Rank4Tensor calcPositiveProjectionTensorImpl(const Rank2Tensor &A,
                                                        double (&eigval)[3],
                                                        Rank2Tensor &eigvec,
                                                        const double tol,
                                                        const bool require_symmetric){
        computeSpectralSystem(A, eigval, eigvec, "getPositiveProjectionTensor", require_symmetric);
        double epos[3],diag[3];
        for(int i=0;i<3;i++){
            epos[i]=0.5*(std::abs(eigval[i])+eigval[i]);
            diag[i]=(eigval[i]>0.0) ? 1.0 : 0.0;
        }

        Rank4Tensor ProjPos(0.0);
        Rank2Tensor Ma(0.0),Mb(0.0);
        ProjPos.setToZeros();

        const Vector3d n1=eigvec.getIthCol(1);
        const Vector3d n2=eigvec.getIthCol(2);
        const Vector3d n3=eigvec.getIthCol(3);
        const Vector3d eigcols[3]={n1,n2,n3};
        for(int i=0;i<3;i++){
            Ma.setFromVectorDyad(eigcols[i],eigcols[i]);
            ProjPos+=Ma.otimes(Ma)*diag[i];
        }

        double theta_ab;
        Rank4Tensor Gab(0.0),Gba(0.0);
        for(int a=0;a<3;a++){
            for(int b=0;b<a;b++){
                Ma.setFromVectorDyad(eigcols[a],eigcols[a]);
                Mb.setFromVectorDyad(eigcols[b],eigcols[b]);

                Gab=Ma.IKxJL(Mb)+Ma.ILxJK(Mb);
                Gba=Mb.IKxJL(Ma)+Mb.ILxJK(Ma);
                if(std::abs(eigval[a]-eigval[b])<=tol){
                    theta_ab=0.25*(diag[a]+diag[b]);
                }
                else{
                    theta_ab=0.5*(epos[a]-epos[b])/(eigval[a]-eigval[b]);
                }
                ProjPos+=theta_ab*(Gab+Gba);
            }
        }
        return ProjPos;
    }
} // namespace


Rank2Tensor::Rank2Tensor(){
    std::fill_n(&m_Vals[0][0], 9, 0.0);
}
Rank2Tensor::Rank2Tensor(const double &val){
    std::fill_n(&m_Vals[0][0], 9, val);
}
Rank2Tensor::Rank2Tensor(const Rank2Tensor &a){
    for(int i=0;i<3;i++){
        m_Vals[i][0]=a.m_Vals[i][0];m_Vals[i][1]=a.m_Vals[i][1];m_Vals[i][2]=a.m_Vals[i][2];
    }
}
Rank2Tensor::Rank2Tensor(const InitMethod &initmethod){
    if(initmethod==InitMethod::ZERO){
        setToZeros();
    }
    else if(initmethod==InitMethod::IDENTITY){
        setToIdentity();
    }
    else if(initmethod==InitMethod::RANDOM){
        setToRandom();
    }
    else{
        MessagePrinter::printErrorTxt("unsupported initialize method in rank-2 tensor");
        MessagePrinter::exitPeriX();
    }
}
Rank2Tensor::~Rank2Tensor()=default;
//**********************************************************************
Rank2Tensor operator*(const double &lhs,const Rank2Tensor &a){
    Rank2Tensor temp(0.0);
    for(int i=0;i<3;i++){
        temp.m_Vals[i][0]=lhs*a.m_Vals[i][0];
        temp.m_Vals[i][1]=lhs*a.m_Vals[i][1];
        temp.m_Vals[i][2]=lhs*a.m_Vals[i][2];
    }
    return temp;
}
Vector3d operator*(const Vector3d &lhs,const Rank2Tensor &a){
    Vector3d temp(0.0);
    const double x=lhs(1), y=lhs(2), z=lhs(3);
    temp(1)=x*a.m_Vals[0][0] + y*a.m_Vals[1][0] + z*a.m_Vals[2][0];
    temp(2)=x*a.m_Vals[0][1] + y*a.m_Vals[1][1] + z*a.m_Vals[2][1];
    temp(3)=x*a.m_Vals[0][2] + y*a.m_Vals[1][2] + z*a.m_Vals[2][2];
    return temp;
}
Rank2Tensor Rank2Tensor::doubledot(const Rank4Tensor &a) const{
    // return A:B calculation
    Rank2Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            const double Aij=m_Vals[i-1][j-1];
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(k,l)+=Aij*a(i,j,k,l);
                }
            }
        }
    }
    return temp;
}
//************************************************************************
void Rank2Tensor::setFromGradU1D(const Vector3d &gradUx){
    (*this)(1,1)=gradUx(1);(*this)(1,2)=0.0;(*this)(1,3)=0.0;
    (*this)(2,1)=      0.0;(*this)(2,2)=0.0;(*this)(2,3)=0.0;
    (*this)(3,1)=      0.0;(*this)(3,2)=0.0;(*this)(3,3)=0.0;
}
void Rank2Tensor::setFromGradU2D(const Vector3d &gradUx,const Vector3d &gradUy){
    (*this)(1,1)=gradUx(1);(*this)(1,2)=gradUx(2);(*this)(1,3)=0.0;
    (*this)(2,1)=gradUy(1);(*this)(2,2)=gradUy(2);(*this)(2,3)=0.0;
    (*this)(3,1)=      0.0;(*this)(3,2)=      0.0;(*this)(3,3)=0.0;
}
void Rank2Tensor::setFromGradU3D(const Vector3d &gradUx,const Vector3d &gradUy,const Vector3d &gradUz){
    (*this)(1,1)=gradUx(1);(*this)(1,2)=gradUx(2);(*this)(1,3)=gradUx(3);
    (*this)(2,1)=gradUy(1);(*this)(2,2)=gradUy(2);(*this)(2,3)=gradUy(3);
    (*this)(3,1)=gradUz(1);(*this)(3,2)=gradUz(2);(*this)(3,3)=gradUz(3);
}
void Rank2Tensor::setRotationTensorFromEulerAngle(const double &theta1,const double &theta2,const double &theta3){
    // set a rotation tensor from Euler-angle
    const double PI=3.141592653589793238462643383279502884;
    double x1=cos(theta1*PI/180.0);
    double x2=cos(theta2*PI/180.0);
    double x3=cos(theta3*PI/180.0);
    double y1=sin(theta1*PI/180.0);
    double y2=sin(theta2*PI/180.0);
    double y3=sin(theta3*PI/180.0);

    (*this)(1,1)= x1*x3-x2*y1*y3;
    (*this)(1,2)= x3*y1+x1*x2*y3;
    (*this)(1,3)= y2*y3;

    (*this)(2,1)=-x1*y3-x2*x3*y1;
    (*this)(2,2)= x1*x2*x3-y1*y3;
    (*this)(2,3)= x3*y2;

    (*this)(3,1)= y1*y2;
    (*this)(3,2)=-x1*y2;
    (*this)(3,3)= x2;
}
//*******************************************************************
//*** for advanced math operators
//*******************************************************************
Rank2Tensor exp(const Rank2Tensor &a){
    Rank2Tensor I;
    I.setToIdentity();
    const Rank2Tensor a2=a*a;
    const Rank2Tensor a3=a2*a;
    const Rank2Tensor a4=a3*a;
    const Rank2Tensor a5=a4*a;
    const Rank2Tensor a6=a5*a;
    return I
          +a
          +a2*(1.0/2.0)
          +a3*(1.0/6.0)
          +a4*(1.0/24.0)
          +a5*(1.0/120.0)
          +a6*(1.0/720.0);
}
Rank2Tensor dexp(const double &a,const Rank2Tensor &b){
    // return dexp(ab)/da
    const double a2=a*a;
    const double a3=a2*a;
    const double a4=a3*a;
    const double a5=a4*a;

    const Rank2Tensor b2=b*b;
    const Rank2Tensor b3=b2*b;
    const Rank2Tensor b4=b3*b;
    const Rank2Tensor b5=b4*b;
    const Rank2Tensor b6=b5*b;

    return b
          +b2*a
          +b3*(a2/2.0)
          +b4*(a3/6.0)
          +b5*(a4/24.0)
          +b6*(a5/120.0);
}
//*******************************************************************
//*** some higher order tensor calculations
//*******************************************************************
Rank4Tensor Rank2Tensor::otimes(const Rank2Tensor &a) const{
    // return C_ijkl=a_ij*b_kl
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            const double Aij=m_Vals[i-1][j-1];
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=Aij*a(k,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::IJxLK(const Rank2Tensor &a) const{
    // return C_ijkl=a_ij*b_lk
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            const double Aij=m_Vals[i-1][j-1];
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=Aij*a(l,k);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::odot(const Rank2Tensor &a) const{
    // extremely useful for:
    //    dA^-1/dA=-A^-1 \odot A^-1
    //            =-0.5*Ainv\odot Ainv=rank-4 tensor
    // for nonlinear constitutive law
    // the proof can be found here:
    // https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=0.5*(m_Vals[i-1][k-1]*a(j,l)+m_Vals[i-1][l-1]*a(j,k));
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For IKxJL and IKxLJ
//********************************************************
Rank4Tensor Rank2Tensor::IKxJL(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[i-1][k-1]*a(j,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::IKxLJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[i-1][k-1]*a(l,j);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For ILxJK and ILxKJ
//********************************************************
Rank4Tensor Rank2Tensor::ILxJK(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[i-1][l-1]*a(j,k);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::ILxKJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[i-1][l-1]*a(k,j);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For JIKL and JILK
//********************************************************
Rank4Tensor Rank2Tensor::JIxKL(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][i-1]*a(k,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::JIxLK(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][i-1]*a(l,k);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For JKxIL and JKxLI
//********************************************************
Rank4Tensor Rank2Tensor::JKxIL(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][k-1]*a(i,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::JKxLI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][k-1]*a(l,i);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For JLIK and JLKI
//********************************************************
Rank4Tensor Rank2Tensor::JLxIK(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][l-1]*a(i,k);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::JLxKI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[j-1][l-1]*a(k,i);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For KIJL and KILJ
//********************************************************
Rank4Tensor Rank2Tensor::KIxJL(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][i-1]*a(j,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::KIxLJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][i-1]*a(l,j);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For KJIL and KJLI
//********************************************************
Rank4Tensor Rank2Tensor::KJxIL(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][j-1]*a(i,l);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::KJxLI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][j-1]*a(l,i);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For KLIJ and KLJI
//********************************************************
Rank4Tensor Rank2Tensor::KLxIJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][l-1]*a(i,j);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::KLxJI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[k-1][l-1]*a(j,i);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For LIKJ and LIJK
//********************************************************
Rank4Tensor Rank2Tensor::LIxKJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][i-1]*a(k,j);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::LIxJK(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][i-1]*a(j,k);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For LJIK and LJKI
//********************************************************
Rank4Tensor Rank2Tensor::LJxIK(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][j-1]*a(i,k);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::LJxKI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][j-1]*a(k,i);
                }
            }
        }
    }
    return temp;
}
//********************************************************
//*** For LKIJ and LKJI
//********************************************************
Rank4Tensor Rank2Tensor::LKxIJ(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][k-1]*a(i,j);
                }
            }
        }
    }
    return temp;
}
Rank4Tensor Rank2Tensor::LKxJI(const Rank2Tensor &a) const{
    Rank4Tensor temp(0.0);
    for(int i=1;i<=3;i++){
        for(int j=1;j<=3;j++){
            for(int k=1;k<=3;k++){
                for(int l=1;l<=3;l++){
                    temp(i,j,k,l)=m_Vals[l-1][k-1]*a(j,i);
                }
            }
        }
    }
    return temp;
}
//**************************************************************
//*** For eigen value and eigen vectors and other
//*** stress and strain decomposition related functions
//**************************************************************
double Rank2Tensor::getMaxPrincipalValue(bool require_symmetric) const{
    const Eigen::Matrix3d M=getSpectralMatrix(*this, "getMaxPrincipalValue", require_symmetric);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
    if(solver.info()!=Eigen::Success){
        MessagePrinter::printErrorTxt("SelfAdjointEigenSolver failed in Rank2Tensor::getMaxPrincipalValue");
        MessagePrinter::exitPeriX();
    }
    return solver.eigenvalues().maxCoeff();
}
double Rank2Tensor::getMinPricipalValue(bool require_symmetric) const{
    const Eigen::Matrix3d M=getSpectralMatrix(*this, "getMinPricipalValue", require_symmetric);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
    if(solver.info()!=Eigen::Success){
        MessagePrinter::printErrorTxt("SelfAdjointEigenSolver failed in Rank2Tensor::getMinPricipalValue");
        MessagePrinter::exitPeriX();
    }
    return solver.eigenvalues().minCoeff();
}
void Rank2Tensor::calcEigenValueAndEigenVectors(double (&eigval)[3],Rank2Tensor &eigvec,bool require_symmetric) const{
    computeSpectralSystem(*this, eigval, eigvec, "calcEigenValueAndEigenVectors", require_symmetric);
}
Rank4Tensor Rank2Tensor::calcPositiveProjTensor(double (&eigval)[3],Rank2Tensor &eigvec,bool require_symmetric) const{
    // Algorithm is taken from:
    // C. Miehe and M. Lambrecht, Commun. Numer. Meth. Engng 2001; 17:337~353
    // https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.404
    return calcPositiveProjectionTensorImpl(*this, eigval, eigvec, 1.0e-14, require_symmetric);
}

Rank4Tensor Rank2Tensor::getPositiveProjectionTensor(bool require_symmetric) const{
    // Algorithm is taken from:
    // C. Miehe and M. Lambrecht, Commun. Numer. Meth. Engng 2001; 17:337~353
    // https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.404
    double eigval[3];
    Rank2Tensor eigvec;
    return calcPositiveProjectionTensorImpl(*this, eigval, eigvec, 1.0e-13, require_symmetric);
}
