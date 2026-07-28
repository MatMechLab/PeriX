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
//+++ Date    : 2026.06.16
//+++ Function: explicit strong-form PDDO dynamic-fracture kernel
//+++           implementation (cached reference PDDO operators +
//+++           health-gated Navier-Cauchy nodal force -> central-difference
//+++           acceleration + irreversible tensile critical-stretch bond
//+++           failure); see the header for the full formulation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/ExplicitPDDOFracElement.h"

#include "ElmtSystem/FractureCriterion.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

#ifdef PERIX_CUDA_ASSEMBLE
#include "PDSystem/CudaAssemble.h"
#endif

void ExplicitPDDOFracElement::recalcElasticConstants() {
    if (m_E<=0.0) { m_C11=m_C12=m_C66=0.0; return; }
    const double nu=m_Nu;
    if (m_Dim>=3 || m_State==StressState::PlaneStrain) {
        const double d=(1.0+nu)*(1.0-2.0*nu);
        if (d<=0.0) {
            MessagePrinter::printErrorTxt("ExplicitPDDOFracElement: (1+nu)(1-2nu) must be > 0");
            MessagePrinter::exitPeriX();
        }
        m_C11=m_E*(1.0-nu)/d; m_C12=m_E*nu/d; m_C66=m_E/(2.0*(1.0+nu));
        return;
    }
    const double d=1.0-nu*nu;
    if (d<=0.0) { MessagePrinter::printErrorTxt("ExplicitPDDOFracElement plane_stress: nu^2<1 required"); MessagePrinter::exitPeriX(); }
    m_C11=m_E/d; m_C12=m_E*nu/d; m_C66=m_E/(2.0*(1.0+nu));
}

double ExplicitPDDOFracElement::estimateStableDt(const PDMesh &Mesh) const {
    // CFL-type bound dt ~ dx/c with the stiffest 1D modulus c=sqrt(C11/rho).
    if (m_E<=0.0 || m_Rho<=0.0) return -1.0;
    const auto &md=Mesh.getDataConstRef();
    double dx=-1.0;
    if (!md.NodeSpacing.empty()) {
        for (const double s : md.NodeSpacing) if (s>0.0 && (dx<=0.0 || s<dx)) dx=s;
    } else {
        for (const double s : {md.DX,md.DY,md.DZ}) if (s>0.0 && (dx<=0.0 || s<dx)) dx=s;
    }
    if (dx<=0.0) return -1.0;
    double c11;
    if (m_Dim>=3 || m_State==StressState::PlaneStrain) {
        const double d=(1.0+m_Nu)*(1.0-2.0*m_Nu);
        if (d<=0.0) return -1.0;
        c11=m_E*(1.0-m_Nu)/d;
    } else {
        const double d=1.0-m_Nu*m_Nu;
        if (d<=0.0) return -1.0;
        c11=m_E/d;
    }
    return dx/std::sqrt(c11/m_Rho);
}

double ExplicitPDDOFracElement::criticalStretch(const double &delta) const {
    const bool ps=(m_Dim<3 && m_State==StressState::PlaneStress);
    return FractureCriterion::criticalStretchFromGc(
        m_G0,m_E,delta,m_Dim,ps);
}

void ExplicitPDDOFracElement::preprocessIteration(const PDMesh &Mesh,
                                                  PDOperators &ops,
                                                  const LocalElmtInfo &Info,
                                                  const int &DofsPerNode,
                                                  const VectorXd &U,
                                                  const VectorXd &Uold) const {
    (void)Uold;
    if (DofsPerNode!=m_Dim) {
        MessagePrinter::printErrorTxt("ExplicitPDDOFracElement: expected DofsPerNode="
                                      +std::to_string(m_Dim)+", got "+std::to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }
    if (m_E<=0.0 || m_Rho<=0.0) {
        MessagePrinter::printErrorTxt("ExplicitPDDOFracElement: E and rho must be > 0");
        MessagePrinter::exitPeriX();
    }
    if (ops.getOrder()<2) {
        MessagePrinter::printErrorTxt("ExplicitPDDOFracElement requires PDMesh.Order >= 2");
        MessagePrinter::exitPeriX();
    }

    const auto &meshData=Mesh.getDataConstRef();
    const int N=Mesh.getNodesNum();
    const int dim=m_Dim;
    const double delta=meshData.HorizonRadius;
    m_NOps=(dim>=3)?9:5;
    const auto &nodeHorizon=meshData.NodeHorizon;
    const bool varH=!nodeHorizon.empty();

    // ===== one-off: flat neighbour list, force-only notch seed, cached ops =====
    if (!m_Initialized || static_cast<int>(m_NeighOff.size())!=N+1) {
        // 64-bit accumulation + loud guard: past 2^31 directed bonds the int
        // prefix sums would wrap silently (same overflow class as the CSR
        // guard on the implicit path).
        long long nbAcc=0;
        m_NeighOff.assign(static_cast<std::size_t>(N)+1,0);
        for (int i=1;i<=N;i++) {
            nbAcc+=static_cast<long long>(Mesh.getIthNodeNeighborNodeIDs(i).size());
            if (nbAcc>static_cast<long long>(std::numeric_limits<int>::max())) {
                MessagePrinter::printErrorTxt("ExplicitPDDOFracElement: more than 2^31-1 directed "
                                              "bonds -- the 32-bit flat neighbour cache would "
                                              "overflow; reduce the model size or horizon factor.");
                MessagePrinter::exitPeriX();
            }
            m_NeighOff[static_cast<std::size_t>(i)]=static_cast<int>(nbAcc);
        }
        const int nbtot=m_NeighOff[static_cast<std::size_t>(N)];
        m_NeighId.assign(static_cast<std::size_t>(nbtot),0);
        m_HealthFlat.assign(static_cast<std::size_t>(nbtot),1.0);
        m_OpCache.assign(static_cast<std::size_t>(nbtot)*m_NOps,0.0);
        m_Accel.assign(static_cast<std::size_t>(N)*dim,0.0);

        const bool forceOnly=meshData.ForceOnlyCracks && meshData.hasInitialCracks();
        // One-off build of the constant reference-geometry operators. Each
        // thread works on a PRIVATE copy of the shared ops object (the
        // established per-thread-copy pattern), so the moment solves fan out
        // across threads; per-bond results are identical regardless of the
        // schedule, so this stays bit-identical to the serial build.
        #pragma omp parallel if(Info.UseParallel)
        {
            PDOperators opsLocal=ops;
            #pragma omp for schedule(guided)
            for (int i=1;i<=N;i++) {
            const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
            const int b=m_NeighOff[static_cast<std::size_t>(i-1)];
            opsLocal.calcAMatrix(i,meshData);
            for (std::size_t t=0;t<neigh.size();++t) {
                const int j=neigh[t];
                m_NeighId[static_cast<std::size_t>(b)+t]=j;
                if (forceOnly && Mesh.initialCrackCutsBond(i,j))
                    m_HealthFlat[static_cast<std::size_t>(b)+t]=0.0;
                opsLocal.calcOperators(i,j,meshData);
                double *op=&m_OpCache[(static_cast<std::size_t>(b)+t)*m_NOps];
                if (dim>=3) {
                    op[0]=opsLocal("d/dx"); op[1]=opsLocal("d/dy"); op[2]=opsLocal("d/dz");
                    op[3]=opsLocal("d2/dx2"); op[4]=opsLocal("d2/dy2"); op[5]=opsLocal("d2/dz2");
                    op[6]=opsLocal("d2/dxdy"); op[7]=opsLocal("d2/dxdz"); op[8]=opsLocal("d2/dydz");
                } else {
                    op[0]=opsLocal("d/dx"); op[1]=opsLocal("d/dy");
                    op[2]=opsLocal("d2/dx2"); op[3]=opsLocal("d2/dy2"); op[4]=opsLocal("d2/dxdy");
                }
            }
            }
        }
        m_Delta=delta;
        m_Sc=criticalStretch(delta);
        m_Initialized=true;
    }

#ifdef PERIX_CUDA_ASSEMBLE
    // ---- GPU path: build the moment matrix + per-bond operators, evolve the
    //      damage, and assemble the health-gated Navier force -> acceleration on
    //      the device. A device failure falls through to the CPU loop below. ----
    if (Info.UseCuda && !m_CudaDisabled) {
        const auto &qidx=ops.getQIndices();
        const int nop=ops.getOperatorsVecSize();
        static std::vector<int> qp,qq,qr; static std::vector<double> factprod;
        qp.resize(static_cast<std::size_t>(nop)); qq.resize(static_cast<std::size_t>(nop));
        qr.resize(static_cast<std::size_t>(nop)); factprod.resize(static_cast<std::size_t>(nop));
        auto fact=[](int k){ double r=1.0; for (int q=2;q<=k;q++) r*=static_cast<double>(q); return r; };
        perix_cuda::OpSlots ss;
        for (int k=0;k<nop;k++) {
            const int pk=qidx[static_cast<std::size_t>(k)][0],qk=qidx[static_cast<std::size_t>(k)][1],rk=qidx[static_cast<std::size_t>(k)][2];
            qp[static_cast<std::size_t>(k)]=pk; qq[static_cast<std::size_t>(k)]=qk; qr[static_cast<std::size_t>(k)]=rk;
            factprod[static_cast<std::size_t>(k)]=fact(pk)*fact(qk)*fact(rk);
            if (pk==2&&qk==0&&rk==0) ss.dxx=k;
            if (pk==0&&qk==2&&rk==0) ss.dyy=k;
            if (pk==0&&qk==0&&rk==2) ss.dzz=k;
            if (pk==1&&qk==1&&rk==0) ss.dxy=k;
            if (pk==1&&qk==0&&rk==1) ss.dxz=k;
            if (pk==0&&qk==1&&rk==1) ss.dyz=k;
        }
        perix_cuda::AssembleMesh am;
        am.N=N; am.neigh_off=m_NeighOff.data(); am.neigh_id=m_NeighId.data();
        am.coords=meshData.NodeCoords.data(); am.vols=meshData.NodeVolumes.data();
        am.qp=qp.data(); am.qq=qq.data(); am.qr=qr.data(); am.factprod=factprod.data();
        am.nop=nop; am.dim=dim; am.delta=delta;
        am.dx_char=meshData.VolumeCorrection?std::max({meshData.DX,meshData.DY,meshData.DZ}):0.0;
        am.nodeHorizon=varH?nodeHorizon.data():nullptr;
        // VolumeCorrection off -> hand the device a ZEROED spacing array, not
        // the real one: devBuildNodeInverse reads nodeSpacing[i] (ignoring the
        // zeroed dx_char) whenever nodeHorizon is set, so the real spacings
        // would silently re-enable the rim partial-volume ramp on the GPU
        // while the CPU op cache runs with vc=1 -- divergent forces baked into
        // the cached device operators. Same convention as the implicit
        // dispatch in FormResidualAndJacobianCUDA.cpp.
        static std::vector<double> zeroSpacing;
        const bool noVc=!meshData.VolumeCorrection;
        if (varH && noVc) zeroSpacing.assign(static_cast<std::size_t>(N),0.0);
        am.nodeSpacing=varH?(noVc?zeroSpacing.data():meshData.NodeSpacing.data()):nullptr;
        perix_cuda::ExplicitPDDOFracParams pp;
        pp.dim=dim; pp.C11=m_C11; pp.C12=m_C12; pp.C66=m_C66; pp.rho=m_Rho;
        pp.bx=m_Bx; pp.by=m_By; pp.bz=m_Bz; pp.sc=m_Sc;
        pp.tension_only=m_TensionOnly?1:0; pp.damage_on=m_DamageOn?1:0;
        pp.G0=m_G0; pp.E=m_E; pp.plane_stress=(m_State==StressState::PlaneStress)?1:0;
        int changed=0;
        if (perix_cuda::assembleExplicitPDDOFracAccel(am,pp,ss,U.getDataPtr(),
                                                      m_HealthFlat.data(),m_Accel.data(),&changed)) {
            return;   // GPU produced the acceleration (+ evolved damage)
        }
        // Device failure: warn once, free the device context, and stay on the
        // CPU for the REST of the run. Alternating backends is unsafe here --
        // the cached device health would miss breaks the CPU loop commits
        // during the fallback steps (m_HealthFlat is only uploaded when the
        // context is rebuilt), silently violating damage irreversibility.
        MessagePrinter::printWarningTxt("explicit_pddo_frac: CUDA rate evaluation failed; falling "
                                        "back to CPU (OpenMP) for the remainder of the run");
        perix_cuda::shutdown();
        m_CudaDisabled=true;
    }
#endif

    auto coord=[&](const int id,const int ax)->double{ return meshData.NodeCoords[(id-1)*3+ax]; };

    // ===== per step: irreversible bond failure (current U) then nodal force ====
    const bool damage=m_DamageOn && m_Sc>0.0;
    #pragma omp parallel for schedule(guided) if(Info.UseParallel)
    for (int i=1;i<=N;i++) {
        const int rowI=(i-1)*dim;
        const int b=m_NeighOff[static_cast<std::size_t>(i-1)];
        const int nb=m_NeighOff[static_cast<std::size_t>(i)]-b;
        const double xi0=coord(i,0),yi0=coord(i,1),zi0=(dim>=3?coord(i,2):0.0);

        // --- bond failure (each node updates its own bonds; symmetric stretch) ---
        if (damage) {
            const double di=varH?nodeHorizon[static_cast<std::size_t>(i-1)]:delta;
            for (int t=0;t<nb;t++) {
                double &hp=m_HealthFlat[static_cast<std::size_t>(b)+t];
                if (hp<=0.0) continue;
                const int j=m_NeighId[static_cast<std::size_t>(b)+t]; const int rowJ=(j-1)*dim;
                double xi[3],eta[3];
                xi[0]=coord(j,0)-xi0; xi[1]=coord(j,1)-yi0; xi[2]=(dim>=3?coord(j,2)-zi0:0.0);
                eta[0]=U(rowJ+1)-U(rowI+1); eta[1]=U(rowJ+2)-U(rowI+2); eta[2]=(dim>=3?U(rowJ+3)-U(rowI+3):0.0);
                const double r=std::sqrt(xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2]);
                if (r<=0.0) continue;
                const double yx=xi[0]+eta[0],yy=xi[1]+eta[1],yz=xi[2]+eta[2];
                const double s=(std::sqrt(yx*yx+yy*yy+yz*yz)-r)/r;
                const double db=varH?0.5*(di+nodeHorizon[static_cast<std::size_t>(j-1)]):delta;
                const double scl=varH?criticalStretch(db):m_Sc;
                if (scl>0.0 && (m_TensionOnly?(s>scl):(std::fabs(s)>scl))) hp=0.0;
            }
        }

        // --- strong-form PDDO Navier force L_i = sum_j health * P_ij (u_j-u_i) ---
        double L[3]={0.0,0.0,0.0};
        for (int t=0;t<nb;t++) {
            const double mu=m_HealthFlat[static_cast<std::size_t>(b)+t];
            if (mu<=0.0) continue;
            const int j=m_NeighId[static_cast<std::size_t>(b)+t]; const int rowJ=(j-1)*dim;
            const double *op=&m_OpCache[(static_cast<std::size_t>(b)+t)*m_NOps];
            const double dux=U(rowJ+1)-U(rowI+1), duy=U(rowJ+2)-U(rowI+2);
            if (dim>=3) {
                const double duz=U(rowJ+3)-U(rowI+3);
                const double Gxx=op[3],Gyy=op[4],Gzz=op[5],Gxy=op[6],Gxz=op[7],Gyz=op[8];
                const double p11=m_C11*Gxx+m_C66*(Gyy+Gzz);
                const double p22=m_C11*Gyy+m_C66*(Gxx+Gzz);
                const double p33=m_C11*Gzz+m_C66*(Gxx+Gyy);
                const double a=m_C12+m_C66;
                const double p12=a*Gxy,p13=a*Gxz,p23=a*Gyz;
                L[0]+=mu*(p11*dux+p12*duy+p13*duz);
                L[1]+=mu*(p12*dux+p22*duy+p23*duz);
                L[2]+=mu*(p13*dux+p23*duy+p33*duz);
            } else {
                const double Gxx=op[2],Gyy=op[3],Gxy=op[4];
                const double p11=m_C11*Gxx+m_C66*Gyy;
                const double p22=m_C66*Gxx+m_C11*Gyy;
                const double pxy=(m_C12+m_C66)*Gxy;
                L[0]+=mu*(p11*dux+pxy*duy);
                L[1]+=mu*(pxy*dux+p22*duy);
            }
        }
        const double bvec[3]={m_Bx,m_By,m_Bz};
        double *ap=&m_Accel[static_cast<std::size_t>(i-1)*dim];
        for (int a=0;a<dim;a++) ap[a]=(L[a]+bvec[a])/m_Rho;
    }
}

void ExplicitPDDOFracElement::computeNodalResidual(const LocalElmtInfo &Info,
                                                   const int &NodeI,
                                                   const VectorXd &U_I,
                                                   const VectorXd &Uold_I,
                                                   const double  &Volume,
                                                   VectorXd &LocalR_node) const {
    (void)Info; (void)U_I; (void)Uold_I; (void)Volume;
    LocalR_node.setToZeros();
    const std::size_t base=static_cast<std::size_t>(NodeI-1)*static_cast<std::size_t>(m_Dim);
    if (!m_Initialized || base+static_cast<std::size_t>(m_Dim)>m_Accel.size()) return;
    for (int a=1;a<=m_Dim;a++) LocalR_node(a)=m_Accel[base+static_cast<std::size_t>(a-1)];
}

std::vector<ProjectionInfo> ExplicitPDDOFracElement::getAvailableProjections() const {
    return {
        {"damage",   ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"vonMises", ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"stress",   ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"strain",   ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)}
    };
}

void ExplicitPDDOFracElement::computeNodalProjection(const std::string &name,
                                                     const PDMesh &Mesh,
                                                     PDOperators &ops,
                                                     const VectorXd &U,
                                                     const int &NodeI,
                                                     const int &DofsPerNode,
                                                     std::vector<double> &out) const {
    (void)ops; (void)DofsPerNode;
    const int dim=m_Dim, i=NodeI;
    const std::size_t idx=static_cast<std::size_t>(i-1);
    if (!m_Initialized || static_cast<int>(m_NeighOff.size())!=Mesh.getNodesNum()+1) {
        std::fill(out.begin(),out.end(),0.0); return;
    }
    const int b=m_NeighOff[idx];
    const int nb=m_NeighOff[idx+1]-b;

    if (name=="damage") {
        if (out.empty()) return;
        if (nb<=0) { out[0]=0.0; return; }
        double live=0.0; for (int t=0;t<nb;t++) live+=m_HealthFlat[static_cast<std::size_t>(b)+t];
        out[0]=1.0-live/static_cast<double>(nb);
        return;
    }

    // displacement gradient from the cached PDDO first-derivative operators
    const auto &md=Mesh.getDataConstRef();
    const int rowI=(i-1)*dim;
    double gu[9]={0,0,0,0,0,0,0,0,0};   // gu[a*dim+c] = du_a/dX_c
    for (int t=0;t<nb;t++) {
        const int j=m_NeighId[static_cast<std::size_t>(b)+t]; const int rowJ=(j-1)*dim;
        const double *op=&m_OpCache[(static_cast<std::size_t>(b)+t)*m_NOps];
        const double du0=U(rowJ+1)-U(rowI+1), du1=U(rowJ+2)-U(rowI+2);
        if (dim>=3) {
            const double du2=U(rowJ+3)-U(rowI+3);
            const double Gx=op[0],Gy=op[1],Gz=op[2];
            gu[0]+=du0*Gx; gu[1]+=du0*Gy; gu[2]+=du0*Gz;
            gu[3]+=du1*Gx; gu[4]+=du1*Gy; gu[5]+=du1*Gz;
            gu[6]+=du2*Gx; gu[7]+=du2*Gy; gu[8]+=du2*Gz;
        } else {
            const double Gx=op[0],Gy=op[1];
            gu[0]+=du0*Gx; gu[1]+=du0*Gy; gu[3]+=du1*Gx; gu[4]+=du1*Gy;
        }
    }
    (void)md;
    // small strain eps = sym(grad u) (6-vector {xx,yy,zz,xy,xz,yz})
    double eps[6]={0,0,0,0,0,0};
    if (dim>=3) { eps[0]=gu[0]; eps[1]=gu[4]; eps[2]=gu[8];
                  eps[3]=0.5*(gu[1]+gu[3]); eps[4]=0.5*(gu[2]+gu[6]); eps[5]=0.5*(gu[5]+gu[7]); }
    else { eps[0]=gu[0]; eps[1]=gu[4]; eps[3]=0.5*(gu[1]+gu[3]); }
    // Cauchy stress sigma = C:eps (Navier constants). In 2D the out-of-plane
    // component follows the plane state: sigma_zz = C12*(exx+eyy) under plane
    // strain, 0 under plane stress (the in-plane relations already encode the
    // state through C11/C12/C66) -- same convention as explicit_frac_mechanics.
    const double lam=m_C12, mu=m_C66;
    const double tr=eps[0]+eps[1]+eps[2];
    double sig[6];
    sig[0]=lam*tr+2.0*mu*eps[0]; sig[1]=lam*tr+2.0*mu*eps[1];
    sig[2]=(m_Dim>=3)?(lam*tr+2.0*mu*eps[2])
                     :((m_State==StressState::PlaneStrain)?lam*tr:0.0);
    sig[3]=2.0*mu*eps[3]; sig[4]=2.0*mu*eps[4]; sig[5]=2.0*mu*eps[5];

    if (name=="vonMises") {
        if (out.empty()) return;
        const double sm=(sig[0]+sig[1]+sig[2])/3.0;
        const double a=sig[0]-sm,c=sig[1]-sm,d=sig[2]-sm;
        out[0]=std::sqrt(1.5*(a*a+c*c+d*d+2.0*(sig[3]*sig[3]+sig[4]*sig[4]+sig[5]*sig[5])));
        return;
    }
    if (name=="stress" && out.size()>=9) {
        out[0]=sig[0]; out[1]=sig[3]; out[2]=sig[4];
        out[3]=sig[3]; out[4]=sig[1]; out[5]=sig[5];
        out[6]=sig[4]; out[7]=sig[5]; out[8]=sig[2];
        return;
    }
    if (name=="strain" && out.size()>=9) {
        out[0]=eps[0]; out[1]=eps[3]; out[2]=eps[4];
        out[3]=eps[3]; out[4]=eps[1]; out[5]=eps[5];
        out[6]=eps[4]; out[7]=eps[5]; out[8]=eps[2];
        return;
    }
    std::fill(out.begin(),out.end(),0.0);
}
