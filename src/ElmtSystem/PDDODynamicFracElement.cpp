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
//+++ Date    : 2026.06.10
//+++ Function: implicit backward-Euler dynamic-fracture kernel on the
//+++           strong-form PDDO operators -- implementation (Navier
//+++           bond blocks x bond health, inertia nodal block, per-step
//+++           velocity commit and irreversible critical-stretch bond
//+++           failure); see the header for the full model.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/PDDODynamicFracElement.h"

#include "ElmtSystem/FractureCriterion.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

std::string PDDODynamicFracElement::getStateName() const {
    if (m_Dim>=3) return "3d";
    switch (m_State) {
        case StressState::PlaneStress: return "plane_stress";
        case StressState::PlaneStrain: return "plane_strain";
    }
    return "unknown";
}

void PDDODynamicFracElement::recalcElasticConstants() {
    if (m_E<=0.0) { m_C11=m_C12=m_C66=0.0; return; }
    const double nu=m_Nu;
    if (m_Dim>=3 || m_State==StressState::PlaneStrain) {
        const double d=(1.0+nu)*(1.0-2.0*nu);
        if (d<=0.0) {
            MessagePrinter::printErrorTxt("PDDODynamicFracElement: (1+nu)(1-2nu) must be > 0 (got nu="
                                          +std::to_string(nu)+")");
            MessagePrinter::exitPeriX();
        }
        m_C11=m_E*(1.0-nu)/d;       // lambda + 2 mu
        m_C12=m_E*nu/d;             // lambda
        m_C66=m_E/(2.0*(1.0+nu));   // mu
        return;
    }
    const double d=1.0-nu*nu;       // plane stress
    if (d<=0.0) {
        MessagePrinter::printErrorTxt("PDDODynamicFracElement plane_stress: nu^2 must be < 1 (got nu="
                                      +std::to_string(nu)+")");
        MessagePrinter::exitPeriX();
    }
    m_C11=m_E/d;
    m_C12=m_E*nu/d;
    m_C66=m_E/(2.0*(1.0+nu));
}

double PDDODynamicFracElement::criticalStretch(const double &delta) const {
    const bool ps=(m_Dim<3 && m_State==StressState::PlaneStress);
    return FractureCriterion::criticalStretchFromGc(
        m_G0,m_E,delta,m_Dim,ps);
}

double PDDODynamicFracElement::bondHealth(const int &NodeI,const int &NodeJ) const {
    if (!m_Initialized) return 1.0;
    const auto &lookup=m_NeighLookup[static_cast<std::size_t>(NodeI-1)];
    const auto it=std::lower_bound(lookup.begin(),lookup.end(),
                                   std::make_pair(NodeJ,-1),
                                   [](const std::pair<int,int> &a,const std::pair<int,int> &b)
                                   { return a.first<b.first; });
    if (it==lookup.end() || it->first!=NodeJ) return 1.0;
    return m_BondHealth[static_cast<std::size_t>(NodeI-1)][static_cast<std::size_t>(it->second)];
}

void PDDODynamicFracElement::preprocessIteration(const PDMesh &Mesh,
                                                 PDOperators &ops,
                                                 const LocalElmtInfo &Info,
                                                 const int &DofsPerNode,
                                                 const VectorXd &U,
                                                 const VectorXd &Uold) const {
    (void)U;
    if (DofsPerNode!=m_Dim) {
        MessagePrinter::printErrorTxt("PDDODynamicFracElement: expected DofsPerNode="
                                      +std::to_string(m_Dim)+", got "+std::to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }
    if (m_E<=0.0 || m_Rho<=0.0) {
        MessagePrinter::printErrorTxt("PDDODynamicFracElement: E and rho must be > 0");
        MessagePrinter::exitPeriX();
    }
    if (Info.Dt<=0.0 || Info.Timestep<1) {
        MessagePrinter::printErrorTxt("PDDODynamicFracElement is a transient (backward-Euler dynamics) "
                                      "kernel; set \"JobSystem\":{\"type\":\"transient\"} with dt > 0");
        MessagePrinter::exitPeriX();
    }
    if (ops.getOrder()<2) {
        MessagePrinter::printErrorTxt("PDDODynamicFracElement requires PDMesh.Order >= 2 "
                                      "(the strong form is second order)");
        MessagePrinter::exitPeriX();
    }

    const auto &meshData=Mesh.getDataConstRef();
    const int N=Mesh.getNodesNum();
    const int dim=m_Dim;
    const auto &nodeHorizon=meshData.NodeHorizon;
    const bool varH=!nodeHorizon.empty();
    const double sc=criticalStretch(meshData.HorizonRadius);

    // ---- one-off allocation + bond-health seeding (force-only notches) ----
    if (!m_Initialized || static_cast<int>(m_Vel.size())!=N*dim) {
        // the velocity history starts from the INITIAL state (Uold at step 1),
        // so a nonzero displacement IC does not inject a spurious v=u_IC/dt.
        m_UPrev.resize(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim));
        for (std::size_t k=0;k<m_UPrev.size();k++) m_UPrev[k]=Uold(static_cast<int>(k)+1);
        m_Vel.resize(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim));
        for (int i=0;i<N;i++) {
            m_Vel[static_cast<std::size_t>(i)*dim+0]=m_V0x;
            m_Vel[static_cast<std::size_t>(i)*dim+1]=m_V0y;
            if (dim>=3) m_Vel[static_cast<std::size_t>(i)*dim+2]=m_V0z;
        }
        m_BondHealth.assign(static_cast<std::size_t>(N),{});
        m_NeighLookup.assign(static_cast<std::size_t>(N),{});
        const bool forceOnlyCracks=meshData.ForceOnlyCracks && meshData.hasInitialCracks();
        for (int i=1;i<=N;i++) {
            const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
            auto &h=m_BondHealth[static_cast<std::size_t>(i-1)];
            auto &lk=m_NeighLookup[static_cast<std::size_t>(i-1)];
            h.assign(neigh.size(),1.0);
            lk.resize(neigh.size());
            for (std::size_t jdx=0;jdx<neigh.size();jdx++) {
                const int j=neigh[jdx];
                lk[jdx]={j,static_cast<int>(jdx)};
                // seed the force-only initial slit: the bond stays in the
                // family (it still builds the PDDO operators) but carries no
                // force from the outset. initialCrackCutsBond is the shared
                // ghost-aware crossing rule (a ghost endpoint is resolved to
                // its mirror bulk), so the seeding agrees with the geometric
                // family cut on which bonds an initial notch kills.
                if (forceOnlyCracks && Mesh.initialCrackCutsBond(i,j)) h[jdx]=0.0;
            }
            std::sort(lk.begin(),lk.end());
        }
        m_Initialized=true;
        m_CommittedStep=Info.Timestep;
        m_LastStepDt=Info.Dt;
        return;
    }

    // ---- per-step commit: velocity history + irreversible bond failure,
    //      both evaluated on the previous step's CONVERGED solution Uold ----
    if (Info.Timestep!=m_CommittedStep) {
        const double dtPrev=(m_LastStepDt>0.0)?m_LastStepDt:Info.Dt;
        const std::size_t n=static_cast<std::size_t>(N)*static_cast<std::size_t>(dim);
        for (std::size_t k=0;k<n;k++) {
            const double uo=Uold(static_cast<int>(k)+1);
            m_Vel[k]=(uo-m_UPrev[k])/dtPrev;
            m_UPrev[k]=uo;
        }

        if (m_DamageOn && (sc>0.0 || varH)) {
            for (int i=1;i<=N;i++) {
                const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
                auto &health=m_BondHealth[static_cast<std::size_t>(i-1)];
                const int rowI=(i-1)*dim;
                const double xi0=meshData.NodeCoords[(i-1)*3+0];
                const double yi0=meshData.NodeCoords[(i-1)*3+1];
                const double zi0=(dim>=3)?meshData.NodeCoords[(i-1)*3+2]:0.0;
                for (std::size_t jdx=0;jdx<neigh.size();jdx++) {
                    if (health[jdx]<=0.0) continue;
                    const int j=neigh[jdx];
                    const int rowJ=(j-1)*dim;
                    double xi[3],eta[3];
                    xi[0]=meshData.NodeCoords[(j-1)*3+0]-xi0;
                    xi[1]=meshData.NodeCoords[(j-1)*3+1]-yi0;
                    xi[2]=(dim>=3)?meshData.NodeCoords[(j-1)*3+2]-zi0:0.0;
                    eta[0]=Uold(rowJ+1)-Uold(rowI+1);
                    eta[1]=Uold(rowJ+2)-Uold(rowI+2);
                    eta[2]=(dim>=3)?Uold(rowJ+3)-Uold(rowI+3):0.0;
                    const double r=std::sqrt(xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2]);
                    if (r<=0.0) continue;
                    const double yx=xi[0]+eta[0],yy=xi[1]+eta[1],yz=xi[2]+eta[2];
                    const double yr=std::sqrt(yx*yx+yy*yy+yz*yz);
                    const double s=(yr-r)/r;
                    // per-bond critical stretch: symmetric 0.5(delta_i+delta_j)
                    // under a variable horizon so i and j break consistently.
                    const double db=varH?0.5*(nodeHorizon[static_cast<std::size_t>(i-1)]
                                             +nodeHorizon[static_cast<std::size_t>(j-1)])
                                        :meshData.HorizonRadius;
                    const double scl=varH?criticalStretch(db):sc;
                    if (scl<=0.0) continue;
                    if (m_TensionOnly ? (s>scl) : (std::fabs(s)>scl)) health[jdx]=0.0;
                }
            }
        }
        m_CommittedStep=Info.Timestep;
    }
    m_LastStepDt=Info.Dt;
}

void PDDODynamicFracElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
                                                            const LocalElmtInfo &Info,
                                                            const int &NodeI,
                                                            const int &NodeJ,
                                                            const VectorXd &U_I,
                                                            const VectorXd &U_J,
                                                            const VectorXd &Uold_I,
                                                            const VectorXd &Uold_J,
                                                            const double  &Volume,
                                                            VectorXd &LocalR,
                                                            MatrixXd &LocalK_II,
                                                            MatrixXd &LocalK_IJ) const {
    (void)Info; (void)Uold_I; (void)Uold_J; (void)Volume;
    LocalR.setToZeros(); LocalK_II.setToZeros(); LocalK_IJ.setToZeros();

    const double mu_ij=bondHealth(NodeI,NodeJ);
    if (mu_ij<=0.0) return;

    // PDDO 2nd-order operator values at bond (i,j); V_j already inside.
    const double Gxx=t_PDOperators("d2/dx2");
    const double Gyy=t_PDOperators("d2/dy2");
    const double Gxy=t_PDOperators("d2/dxdy");

    if (m_Dim>=3) {
        const double Gzz=t_PDOperators("d2/dz2");
        const double Gxz=t_PDOperators("d2/dxdz");
        const double Gyz=t_PDOperators("d2/dydz");
        const double a  =m_C12+m_C66;   // lambda + mu

        const double p11=mu_ij*(m_C11*Gxx+m_C66*(Gyy+Gzz));
        const double p22=mu_ij*(m_C11*Gyy+m_C66*(Gxx+Gzz));
        const double p33=mu_ij*(m_C11*Gzz+m_C66*(Gxx+Gyy));
        const double p12=mu_ij*a*Gxy;
        const double p13=mu_ij*a*Gxz;
        const double p23=mu_ij*a*Gyz;

        const double dux=U_J(1)-U_I(1);
        const double duy=U_J(2)-U_I(2);
        const double duz=U_J(3)-U_I(3);

        LocalR(1)=p11*dux+p12*duy+p13*duz;
        LocalR(2)=p12*dux+p22*duy+p23*duz;
        LocalR(3)=p13*dux+p23*duy+p33*duz;

        LocalK_II(1,1)=-p11; LocalK_II(1,2)=-p12; LocalK_II(1,3)=-p13;
        LocalK_II(2,1)=-p12; LocalK_II(2,2)=-p22; LocalK_II(2,3)=-p23;
        LocalK_II(3,1)=-p13; LocalK_II(3,2)=-p23; LocalK_II(3,3)=-p33;
        LocalK_IJ(1,1)= p11; LocalK_IJ(1,2)= p12; LocalK_IJ(1,3)= p13;
        LocalK_IJ(2,1)= p12; LocalK_IJ(2,2)= p22; LocalK_IJ(2,3)= p23;
        LocalK_IJ(3,1)= p13; LocalK_IJ(3,2)= p23; LocalK_IJ(3,3)= p33;
        return;
    }

    const double p11=mu_ij*(m_C11*Gxx+m_C66*Gyy);
    const double p22=mu_ij*(m_C66*Gxx+m_C11*Gyy);
    const double pxy=mu_ij*(m_C12+m_C66)*Gxy;

    const double dux=U_J(1)-U_I(1);
    const double duy=U_J(2)-U_I(2);

    LocalR(1)=p11*dux+pxy*duy;
    LocalR(2)=pxy*dux+p22*duy;

    LocalK_II(1,1)=-p11;  LocalK_IJ(1,1)= p11;
    LocalK_II(1,2)=-pxy;  LocalK_IJ(1,2)= pxy;
    LocalK_II(2,1)=-pxy;  LocalK_IJ(2,1)= pxy;
    LocalK_II(2,2)=-p22;  LocalK_IJ(2,2)= p22;
}

void PDDODynamicFracElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                             const int &NodeI,
                                                             const VectorXd &U_I,
                                                             const VectorXd &Uold_I,
                                                             const double  &Volume,
                                                             VectorXd &LocalR_node,
                                                             MatrixXd &LocalK_node) const {
    (void)Volume;
    LocalR_node.setToZeros();
    LocalK_node.setToZeros();
    const double dt=Info.Dt;
    const double c0=m_Rho/(dt*dt);
    const std::size_t base=static_cast<std::size_t>(NodeI-1)*static_cast<std::size_t>(m_Dim);
    const double b[3]={m_Bx,m_By,m_Bz};
    for (int a=1;a<=m_Dim;a++) {
        const double v=(m_Initialized && base+static_cast<std::size_t>(a-1)<m_Vel.size())
                       ? m_Vel[base+static_cast<std::size_t>(a-1)] : 0.0;
        // backward Euler: R = L + b - rho (u - u_old - v dt)/dt^2
        LocalR_node(a)=b[a-1]-c0*(U_I(a)-Uold_I(a)-v*dt);
        LocalK_node(a,a)=-c0;
    }
}

std::vector<ProjectionInfo> PDDODynamicFracElement::getAvailableProjections() const {
    return {
        {"strain",   ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"stress",   ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"vonMises", ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"damage",   ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)}
    };
}

void PDDODynamicFracElement::computeNodalProjection(const std::string &name,
                                                    const PDMesh &Mesh,
                                                    PDOperators &ops,
                                                    const VectorXd &U,
                                                    const int &NodeI,
                                                    const int &DofsPerNode,
                                                    std::vector<double> &out) const {
    if (name=="damage") {
        if (out.empty()) return;
        double d=0.0;
        if (m_Initialized && NodeI-1<static_cast<int>(m_BondHealth.size())) {
            const auto &health=m_BondHealth[static_cast<std::size_t>(NodeI-1)];
            if (!health.empty()) {
                double live=0.0;
                for (const double h : health) live+=h;
                d=1.0-live/static_cast<double>(health.size());
            }
        }
        out[0]=std::clamp(d,0.0,1.0);
        return;
    }

    // strain / stress / vonMises from the PDDO 1st-order operators. The
    // gradient spans the full family (the frozen-operator convention), so
    // values straddling an open crack are smeared across it -- read the
    // 'damage' field to locate the crack itself.
    const auto &neighbors=Mesh.getIthNodeNeighborNodeIDs(NodeI);
    const auto &meshData=Mesh.getDataConstRef();
    const int rowI=(NodeI-1)*DofsPerNode;
    double gu[9]={0,0,0,0,0,0,0,0,0};   // row-major du_a/dx_b, b fastest
    for (const int NodeJ : neighbors) {
        ops.calcOperators(NodeI,NodeJ,meshData);
        const double g[3]={ops("d/dx"),ops("d/dy"),(m_Dim>=3)?ops("d/dz"):0.0};
        const int rowJ=(NodeJ-1)*DofsPerNode;
        for (int a=0;a<m_Dim;a++) {
            const double du=U(rowJ+a+1)-U(rowI+a+1);
            for (int bb=0;bb<m_Dim;bb++) gu[a*3+bb]+=du*g[bb];
        }
    }

    double eps[6]={0,0,0,0,0,0};        // xx,yy,zz,xy,xz,yz
    eps[0]=gu[0];
    eps[1]=gu[4];
    eps[3]=0.5*(gu[1]+gu[3]);
    if (m_Dim>=3) {
        eps[2]=gu[8];
        eps[4]=0.5*(gu[2]+gu[6]);
        eps[5]=0.5*(gu[5]+gu[7]);
    }

    double sxx,syy,szz,sxy,sxz,syz;
    if (m_Dim>=3) {
        sxx=m_C11*eps[0]+m_C12*(eps[1]+eps[2]);
        syy=m_C11*eps[1]+m_C12*(eps[0]+eps[2]);
        szz=m_C11*eps[2]+m_C12*(eps[0]+eps[1]);
        sxy=2.0*m_C66*eps[3];
        sxz=2.0*m_C66*eps[4];
        syz=2.0*m_C66*eps[5];
    }
    else {
        sxx=m_C11*eps[0]+m_C12*eps[1];
        syy=m_C12*eps[0]+m_C11*eps[1];
        szz=(m_State==StressState::PlaneStrain)?m_C12*(eps[0]+eps[1]):0.0;
        sxy=2.0*m_C66*eps[3];
        sxz=syz=0.0;
    }

    if (name=="strain" && out.size()>=9) {
        out[0]=eps[0]; out[1]=eps[3]; out[2]=eps[4];
        out[3]=eps[3]; out[4]=eps[1]; out[5]=eps[5];
        out[6]=eps[4]; out[7]=eps[5]; out[8]=eps[2];
    }
    else if (name=="stress" && out.size()>=9) {
        out[0]=sxx; out[1]=sxy; out[2]=sxz;
        out[3]=sxy; out[4]=syy; out[5]=syz;
        out[6]=sxz; out[7]=syz; out[8]=szz;
    }
    else if (name=="vonMises" && !out.empty()) {
        const double a=sxx-syy,bb=syy-szz,c=szz-sxx;
        out[0]=std::sqrt(0.5*(a*a+bb*bb+c*c+6.0*(sxy*sxy+sxz*sxz+syz*syz)));
    }
    else {
        std::fill(out.begin(),out.end(),0.0);
    }
}
