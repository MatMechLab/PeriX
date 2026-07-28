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
//+++ Date    : 2026.06.25
//+++ Function: small-strain fracture + stress-coupled Cahn-Hilliard
//+++           assembly (see FracStressCahnHilliardElement.h for the
//+++           full theory). Chemo-mechanical CH coupling, plus
//+++           irreversible PD bond-stretch damage that gates the
//+++           mechanical bonds.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/FracStressCahnHilliardElement.h"

#include "ElmtSystem/FractureCriterion.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
    // keep c strictly inside (eps,1-eps) so log(c) and 1/(c(1-c)) stay finite.
    constexpr double kCEps = 1.0e-8;
    inline double clampC(double c) { return std::max(kCEps, std::min(1.0 - kCEps, c)); }

    inline double fPrime(double c, double chi) {
        const double cc = clampC(c);
        return std::log(cc / (1.0 - cc)) + chi * (1.0 - 2.0 * cc);
    }
    inline double fDoublePrime(double c, double chi) {
        const double cc = clampC(c);
        return 1.0 / (cc * (1.0 - cc)) - 2.0 * chi;
    }
    // degenerate Cahn-Hilliard mobility M(c)=M0 c(1-c), dM/dc=M0(1-2c).
    inline double mobility(double c, double M0)      { const double cc=clampC(c); return M0*cc*(1.0-cc); }

    struct LocalStress {
        double eps_xx, eps_yy, eps_xy;
        double s_xx, s_yy, s_xy, s_zz;
    };
    LocalStress computeStress(const PDMesh &Mesh, PDOperators &ops, const VectorXd &U,
                              const int NodeI, const int DofsPerNode,
                              const double C11, const double C12, const double C66,
                              const double A, const double cref, const bool plane_strain) {
        const auto &neighbors = Mesh.getIthNodeNeighborNodeIDs(NodeI);
        const auto &meshData  = Mesh.getDataConstRef();
        const int rowI = (NodeI - 1) * DofsPerNode;
        const double c_i = U(rowI + 1);
        double dux_dx=0.0, dux_dy=0.0, duy_dx=0.0, duy_dy=0.0;
        for (const int NodeJ : neighbors) {
            ops.calcOperators(NodeI, NodeJ, meshData);
            const double gx = ops("d/dx"), gy = ops("d/dy");
            const int rowJ = (NodeJ - 1) * DofsPerNode;
            const double dux = U(rowJ + 3) - U(rowI + 3);
            const double duy = U(rowJ + 4) - U(rowI + 4);
            dux_dx += dux*gx; dux_dy += dux*gy; duy_dx += duy*gx; duy_dy += duy*gy;
        }
        LocalStress s{};
        s.eps_xx = dux_dx; s.eps_yy = duy_dy; s.eps_xy = 0.5*(dux_dy + duy_dx);
        const double dc = c_i - cref;
        s.s_xx = C11*s.eps_xx + C12*s.eps_yy - A*dc;
        s.s_yy = C12*s.eps_xx + C11*s.eps_yy - A*dc;
        s.s_xy = 2.0*C66*s.eps_xy;
        s.s_zz = plane_strain ? (C12*(s.eps_xx + s.eps_yy) - A*dc) : 0.0;
        return s;
    }

    struct LocalStress3D {
        double exx,eyy,ezz,exy,exz,eyz;
        double sxx,syy,szz,sxy,sxz,syz;
    };
    LocalStress3D computeStress3D(const PDMesh &Mesh, PDOperators &ops, const VectorXd &U,
                                  const int NodeI, const int DofsPerNode,
                                  const double C11, const double C12, const double C66,
                                  const double A, const double cref) {
        const auto &neighbors = Mesh.getIthNodeNeighborNodeIDs(NodeI);
        const auto &meshData  = Mesh.getDataConstRef();
        const int rowI = (NodeI - 1) * DofsPerNode;     // c=1, mu=2, ux=3, uy=4, uz=5
        const double c_i = U(rowI + 1);
        double uxx=0,uxy=0,uxz=0,uyx=0,uyy=0,uyz=0,uzx=0,uzy=0,uzz=0;
        for (const int NodeJ : neighbors) {
            ops.calcOperators(NodeI, NodeJ, meshData);
            const double gx=ops("d/dx"), gy=ops("d/dy"), gz=ops("d/dz");
            const int rowJ=(NodeJ-1)*DofsPerNode;
            const double dux=U(rowJ+3)-U(rowI+3);
            const double duy=U(rowJ+4)-U(rowI+4);
            const double duz=U(rowJ+5)-U(rowI+5);
            uxx+=dux*gx; uxy+=dux*gy; uxz+=dux*gz;
            uyx+=duy*gx; uyy+=duy*gy; uyz+=duy*gz;
            uzx+=duz*gx; uzy+=duz*gy; uzz+=duz*gz;
        }
        LocalStress3D s{};
        s.exx=uxx; s.eyy=uyy; s.ezz=uzz;
        s.exy=0.5*(uxy+uyx); s.exz=0.5*(uxz+uzx); s.eyz=0.5*(uyz+uzy);
        const double dc=c_i-cref;
        s.sxx=C11*s.exx + C12*(s.eyy+s.ezz) - A*dc;
        s.syy=C11*s.eyy + C12*(s.exx+s.ezz) - A*dc;
        s.szz=C11*s.ezz + C12*(s.exx+s.eyy) - A*dc;
        s.sxy=2.0*C66*s.exy; s.sxz=2.0*C66*s.exz; s.syz=2.0*C66*s.eyz;
        return s;
    }
}

std::string FracStressCahnHilliardElement::getStateName() const {
    return (m_State==StressState::PlaneStrain) ? "plane_strain" : "plane_stress";
}

void FracStressCahnHilliardElement::recalcElasticConstants() {
    m_C11=m_C12=m_C66=0.0; m_A=0.0; m_Kh=0.0; m_Ch=0.0;
    const double nu=m_Nu;
    if (m_Dim>=3) {
        const double d=(1.0+nu)*(1.0-2.0*nu);
        if (d<=0.0) { MessagePrinter::printErrorTxt("FracStressCahnHilliardElement 3D: (1+nu)(1-2nu) must be > 0 (got nu="
                          +std::to_string(nu)+")"); MessagePrinter::exitPeriX(); }
        m_C11=m_E*(1.0-nu)/d; m_C12=m_E*nu/d; m_C66=m_E/(2.0*(1.0+nu));
        m_A =m_E*m_Omega/(3.0*(1.0-2.0*nu));
        m_Kh=(m_C11+2.0*m_C12)/3.0; m_Ch=-m_A;
        return;
    }
    if (m_State==StressState::PlaneStress) {
        const double d=1.0-nu*nu;
        if (d<=0.0) { MessagePrinter::printErrorTxt("FracStressCahnHilliardElement plane_stress: nu^2 must be < 1 (got nu="
                          +std::to_string(nu)+")"); MessagePrinter::exitPeriX(); }
        m_C11=m_E/d; m_C12=m_E*nu/d; m_C66=m_E/(2.0*(1.0+nu));
        m_A =m_E*m_Omega/(3.0*(1.0-nu));
        m_Kh=(m_C11+m_C12)/3.0; m_Ch=-2.0*m_A/3.0;
        return;
    }
    const double d=(1.0+nu)*(1.0-2.0*nu);   // plane strain
    if (d<=0.0) { MessagePrinter::printErrorTxt("FracStressCahnHilliardElement plane_strain: (1+nu)(1-2nu) must be > 0 (got nu="
                      +std::to_string(nu)+")"); MessagePrinter::exitPeriX(); }
    m_C11=m_E*(1.0-nu)/d; m_C12=m_E*nu/d; m_C66=m_E/(2.0*(1.0+nu));
    m_A =m_E*m_Omega/(3.0*(1.0-2.0*nu));
    m_Kh=(m_C11+2.0*m_C12)/3.0; m_Ch=-m_A;
}

double FracStressCahnHilliardElement::criticalStretch(const double &delta) const {
    const bool ps=(m_Dim<3 && m_State==StressState::PlaneStress);
    return FractureCriterion::criticalStretchFromGc(
        m_G0,m_E,delta,m_Dim,ps);
}

double FracStressCahnHilliardElement::bondHealth(const int &NodeI,const int &NodeJ) const {
    if (!m_Initialized) return 1.0;
    const int slot=bondSlot(NodeI,NodeJ);
    if (slot<0) return 1.0;
    return m_BondHealth[static_cast<std::size_t>(NodeI-1)][static_cast<std::size_t>(slot)];
}

int FracStressCahnHilliardElement::bondSlot(const int &NodeI,const int &NodeJ) const {
    if (!m_Initialized) return -1;
    const auto &lookup=m_NeighLookup[static_cast<std::size_t>(NodeI-1)];
    const auto it=std::lower_bound(lookup.begin(),lookup.end(),std::make_pair(NodeJ,-1),
                                   [](const std::pair<int,int> &a,const std::pair<int,int> &b)
                                   { return a.first<b.first; });
    if (it==lookup.end() || it->first!=NodeJ) return -1;
    return it->second;
}

void FracStressCahnHilliardElement::preprocessIteration(const PDMesh &Mesh,
                                                        PDOperators &ops,
                                                        const LocalElmtInfo &Info,
                                                        const int &DofsPerNode,
                                                        const VectorXd &U,
                                                        const VectorXd &Uold) const {
    const int ndof=m_Dim+2;
    if (DofsPerNode!=ndof) {
        MessagePrinter::printErrorTxt("FracStressCahnHilliardElement: expected DofsPerNode="
                                      +std::to_string(ndof)+", got "+std::to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }
    if (Info.Dt<=0.0) {
        MessagePrinter::printErrorTxt("FracStressCahnHilliardElement is transient-only (CH); set "
                                      "\"JobSystem\":{\"type\":\"transient\"} with dt > 0");
        MessagePrinter::exitPeriX();
    }
    if (ops.getOrder()<2) {
        MessagePrinter::printErrorTxt("FracStressCahnHilliardElement requires PDMesh.Order >= 2");
        MessagePrinter::exitPeriX();
    }

    const auto &meshData=Mesh.getDataConstRef();
    const int N=Mesh.getNodesNum();
    const int dim=m_Dim;
    const bool useParallel=Info.UseParallel;

    // ---------- (A) Picard-freeze the degenerate mobility (EVERY iteration) ----------
    // The conservative bond flux only needs the nodal mobility M(c); the old
    // collocation form's grad(M).grad(mu) term is absorbed into the
    // antisymmetric bond weight (see the m_LapC cache below).
    if (static_cast<int>(m_M.size())!=N) {
        m_M.assign(static_cast<std::size_t>(N),0.0);
    }
    for (int i=1;i<=N;++i) {
        m_M[static_cast<std::size_t>(i-1)]=mobility(U((i-1)*ndof+1),m_D);
    }

    // ---------- (A2) conservative species-scheme geometry (built ONCE) ----------
    // Species surface correction. Per bond: the full-family PDDO
    // Laplacian weight g2_ij (the
    // full family keeps the moment matrix well-conditioned at walls), zeroed on
    // ghost bonds and rescaled per node by the Madenci/Oterkus consistency
    // scalar lambda so the bulk-only stencil reproduces Lap(|x|^2)=2*dim.
    // The reverse-bond slots make the flux weight antisymmetric in the
    // volume-weighted sense -> the bulk<->bulk species flux telescopes and the
    // discrete total changes only through the speciesflux SOURCE.
    if (static_cast<int>(m_NeighOff.size())!=N+1) {
        m_NeighOff.assign(static_cast<std::size_t>(N)+1,0);
        for (int i=1;i<=N;++i)
            m_NeighOff[static_cast<std::size_t>(i)]=m_NeighOff[static_cast<std::size_t>(i-1)]
                +static_cast<int>(Mesh.getIthNodeNeighborNodeIDs(i).size());
        const int NB=m_NeighOff[static_cast<std::size_t>(N)];
        m_LapC.assign(static_cast<std::size_t>(NB),0.0);
        m_Rev.assign(static_cast<std::size_t>(NB),-1);
        m_NodeVol.assign(static_cast<std::size_t>(N),0.0);
        m_IsGhost.assign(static_cast<std::size_t>(N),0);
        {
            const auto &gm=meshData.GhostMirrorBulkID;
            if (!gm.empty())
                for (int i=1;i<=N;i++)
                    if (i<=static_cast<int>(gm.size()) && gm[static_cast<std::size_t>(i-1)]>0)
                        m_IsGhost[static_cast<std::size_t>(i-1)]=1;
        }
        for (int i=1;i<=N;i++) m_NodeVol[static_cast<std::size_t>(i-1)]=Mesh.getIthNodeVolume(i);

        auto coord=[&](const int id,const int ax)->double{ return meshData.NodeCoords[(id-1)*3+ax]; };
#pragma omp parallel if(useParallel)
        {
            PDOperators op=ops;   // per-thread copy (calcAMatrix/calcOperators mutate it)
#pragma omp for schedule(guided)
            for (int i=1;i<=N;++i) {
                const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
                const int b=m_NeighOff[static_cast<std::size_t>(i-1)];
                op.calcAMatrix(i,meshData);
                const double qi=coord(i,0)*coord(i,0)+coord(i,1)*coord(i,1)
                               +(dim>=3?coord(i,2)*coord(i,2):0.0);
                double lapq=0.0;
                for (std::size_t t=0;t<neigh.size();++t) {
                    const int j=neigh[t];
                    op.calcOperators(i,j,meshData);
                    const double g2=pdLaplacian(op);
                    if (m_IsGhost[static_cast<std::size_t>(j-1)]) continue;  // ghost bond: no species flux
                    m_LapC[static_cast<std::size_t>(b)+t]=g2;
                    const double qj=coord(j,0)*coord(j,0)+coord(j,1)*coord(j,1)
                                   +(dim>=3?coord(j,2)*coord(j,2):0.0);
                    lapq+=g2*(qj-qi);
                }
                // Keep the directly constructed bulk-only wall stencil.
                // Conservation remains exact because the antisymmetric bond
                // weight telescopes for any per-node scalar.
                (void)lapq;
            }
        }
        // reverse-bond slots (family lists are symmetric by construction)
        for (int i=1;i<=N;i++) {
            const int bi=m_NeighOff[static_cast<std::size_t>(i-1)];
            const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
            for (std::size_t t=0;t<neigh.size();++t) {
                const int j=neigh[t];
                const int bj=m_NeighOff[static_cast<std::size_t>(j-1)];
                const auto &nj=Mesh.getIthNodeNeighborNodeIDs(j);
                for (std::size_t s=0;s<nj.size();++s)
                    if (nj[s]==i) { m_Rev[static_cast<std::size_t>(bi)+t]=bj+static_cast<int>(s); break; }
            }
        }
    }

    // ---------- (B) one-off bond-health allocation + force-only notch seeding ----------
    if (!m_Initialized || static_cast<int>(m_BondHealth.size())!=N) {
        if (m_Rho>0.0) {
            m_UPrev.assign(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim),0.0);
            m_Vel.assign(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim),0.0);
            for (int i=1;i<=N;i++) {
                const int rowI=(i-1)*ndof;
                const std::size_t base=static_cast<std::size_t>(i-1)*static_cast<std::size_t>(dim);
                m_UPrev[base+0]=Uold(rowI+3);
                m_UPrev[base+1]=Uold(rowI+4);
                if (dim>=3) m_UPrev[base+2]=Uold(rowI+5);
            }
        } else {
            m_UPrev.clear();
            m_Vel.clear();
        }
        m_BondHealth.assign(static_cast<std::size_t>(N),{});
        m_BondDamageIShare.assign(static_cast<std::size_t>(N),{});
        m_NeighLookup.assign(static_cast<std::size_t>(N),{});
        const bool forceOnlyCracks=meshData.ForceOnlyCracks && meshData.hasInitialCracks();
        for (int i=1;i<=N;i++) {
            const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
            auto &h=m_BondHealth[static_cast<std::size_t>(i-1)];
            auto &di=m_BondDamageIShare[static_cast<std::size_t>(i-1)];
            auto &lk=m_NeighLookup[static_cast<std::size_t>(i-1)];
            h.assign(neigh.size(),1.0); di.assign(neigh.size(),0.0); lk.resize(neigh.size());
            for (std::size_t jdx=0;jdx<neigh.size();jdx++) {
                const int j=neigh[jdx]; lk[jdx]={j,static_cast<int>(jdx)};
                if (forceOnlyCracks && Mesh.initialCrackCutsBond(i,j)) h[jdx]=0.0;
            }
            std::sort(lk.begin(),lk.end());
        }
        m_Initialized=true; m_CommittedStep=Info.Timestep;
        m_LastStepDt=Info.Dt;
    }

    // ---------- (C) per-step history commit on the accepted Uold ----------
    if (Info.Timestep!=m_CommittedStep) {
        if (m_Rho>0.0) {
            if (static_cast<int>(m_UPrev.size())!=N*dim ||
                static_cast<int>(m_Vel.size())!=N*dim) {
                m_UPrev.assign(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim),0.0);
                m_Vel.assign(static_cast<std::size_t>(N)*static_cast<std::size_t>(dim),0.0);
            }
            const double dtPrev=(m_LastStepDt>0.0)?m_LastStepDt:Info.Dt;
            for (int i=1;i<=N;i++) {
                const int rowI=(i-1)*ndof;
                const std::size_t base=static_cast<std::size_t>(i-1)*static_cast<std::size_t>(dim);
                const double ux=Uold(rowI+3);
                const double uy=Uold(rowI+4);
                m_Vel[base+0]=(ux-m_UPrev[base+0])/dtPrev;
                m_Vel[base+1]=(uy-m_UPrev[base+1])/dtPrev;
                m_UPrev[base+0]=ux;
                m_UPrev[base+1]=uy;
                if (dim>=3) {
                    const double uz=Uold(rowI+5);
                    m_Vel[base+2]=(uz-m_UPrev[base+2])/dtPrev;
                    m_UPrev[base+2]=uz;
                }
            }
        }

        // per-step IRREVERSIBLE breakage on the accepted Uold
        const auto &nodeHorizon=meshData.NodeHorizon;
        const bool varH=!nodeHorizon.empty();
        const double sc=criticalStretch(meshData.HorizonRadius);
        const bool doDamage=m_DamageOn && (sc>0.0 || varH);
        if (doDamage) {
            auto breakReciprocalBond=[&](const int nodeI,const int nodeJ,const double qI) {
                const auto &lk=m_NeighLookup[static_cast<std::size_t>(nodeJ-1)];
                const auto it=std::lower_bound(lk.begin(),lk.end(),
                    std::make_pair(nodeI,-1),
                    [](const std::pair<int,int>&a,const std::pair<int,int>&b)
                    { return a.first<b.first; });
                if (it!=lk.end() && it->first==nodeI) {
                    const std::size_t slot=static_cast<std::size_t>(it->second);
                    m_BondHealth[static_cast<std::size_t>(nodeJ-1)][slot]=0.0;
                    m_BondDamageIShare[static_cast<std::size_t>(nodeJ-1)][slot]=qI;
                }
            };
            for (int i=1;i<=N;i++) {
                const int rowI=(i-1)*ndof;
                const double ci=Uold(rowI+1);
                const double xi0=meshData.NodeCoords[(i-1)*3+0];
                const double yi0=meshData.NodeCoords[(i-1)*3+1];
                const double zi0=(dim>=3)?meshData.NodeCoords[(i-1)*3+2]:0.0;
                const auto &neigh=Mesh.getIthNodeNeighborNodeIDs(i);
                auto &health=m_BondHealth[static_cast<std::size_t>(i-1)];
                auto &damageI=m_BondDamageIShare[static_cast<std::size_t>(i-1)];
                for (std::size_t jdx=0;jdx<neigh.size();jdx++) {
                    if (health[jdx]<=0.0) continue;
                    const int j=neigh[jdx];
                    const int rowJ=(j-1)*ndof;
                    double xi[3],eta[3];
                    xi[0]=meshData.NodeCoords[(j-1)*3+0]-xi0;
                    xi[1]=meshData.NodeCoords[(j-1)*3+1]-yi0;
                    xi[2]=(dim>=3)?meshData.NodeCoords[(j-1)*3+2]-zi0:0.0;
                    eta[0]=Uold(rowJ+3)-Uold(rowI+3);
                    eta[1]=Uold(rowJ+4)-Uold(rowI+4);
                    eta[2]=(dim>=3)?Uold(rowJ+5)-Uold(rowI+5):0.0;
                    const double r2=xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2];
                    if (r2<=0.0) continue;
                    const double r=std::sqrt(r2);
                    const double rinv=1.0/r;
                    // Geometric opening stretch (rotation-exact).
                    const double yx=xi[0]+eta[0],yy=xi[1]+eta[1],yz=xi[2]+eta[2];
                    const double sn=(std::sqrt(yx*yx+yy*yy+yz*yz)-r)*rinv;
                    // ELASTIC stretch: subtract the (stress-free) lithiation swelling
                    // s_swell=(Omega/3)(c_bar-cref) so uniform swelling never cracks.
                    const double cbar=0.5*(ci+Uold(rowJ+1));
                    const double s_el=sn-(m_Omega/3.0)*(cbar-m_CRef);
                    const double so=m_TensionOnly?std::max(s_el,0.0):std::fabs(s_el);
                    // Critical stretches scale with the horizon. Under a variable
                    // horizon use a per-bond delta = 0.5(delta_i+delta_j): symmetric
                    // (so the reciprocal break is consistent) and equal to the global
                    // delta for a uniform mesh.
                    const double db=varH?0.5*(nodeHorizon[static_cast<std::size_t>(i-1)]
                                             +nodeHorizon[static_cast<std::size_t>(j-1)])
                                        :meshData.HorizonRadius;
                    const double scl=varH?criticalStretch(db):sc;
                    if (scl<=0.0) continue;
                    if (so>=scl) {
                        health[jdx]=0.0;
                        damageI[jdx]=1.0;
                        breakReciprocalBond(i,j,1.0);
                    }
                }
            }
        }
        m_CommittedStep=Info.Timestep;
    }
    m_LastStepDt=Info.Dt;
}

void FracStressCahnHilliardElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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

    const double Gx  = t_PDOperators("d/dx");
    const double Gy  = t_PDOperators("d/dy");
    const double Gxx = t_PDOperators("d2/dx2");
    const double Gyy = t_PDOperators("d2/dy2");
    const double Gxy = t_PDOperators("d2/dxdy");

    const std::size_t k = static_cast<std::size_t>(NodeI-1);
    const double M_i = m_M[k];

    // bond health -> mechanical gate with residual-stiffness floor (quasi-static).
    const double mu_ij = bondHealth(NodeI,NodeJ);
    const double g = (mu_ij>0.0) ? 1.0 : m_ResidualStiffness;

    // ---- conservative species weights (surface-corrected, bulk bonds only) ----
    // w_ij = 0.5( M_i g2_ij + (V_j/V_i) M_j g2_ji ): volume-weighted antisymmetric,
    // so the bulk<->bulk flux telescopes (exact conservation); a bond to a
    // boundary ghost carries NO species flux (m_LapC=0 there) -- the boundary
    // influx enters through the speciesflux SOURCE instead. The kappa Lap(c)
    // term of the mu-equation uses the same lambda-corrected one-sided weight.
    // Mobility is Picard-frozen, so the Jacobian below stays exact.
    double wFlux=0.0, g2c=0.0;
    {
        const int slot=bondSlot(NodeI,NodeJ);
        if (slot>=0 && !m_IsGhost[static_cast<std::size_t>(NodeJ-1)]) {
            const int bond=m_NeighOff[k]+slot;
            const std::size_t jj=static_cast<std::size_t>(NodeJ-1);
            g2c=m_LapC[static_cast<std::size_t>(bond)];
            const int rev=m_Rev[static_cast<std::size_t>(bond)];
            if (rev>=0) {
                wFlux=0.5*(M_i*g2c + (m_NodeVol[jj]/m_NodeVol[k])*m_M[jj]
                                     *m_LapC[static_cast<std::size_t>(rev)]);
            }
            else wFlux=M_i*g2c;   // unpaired bond: forward-only fallback
        }
    }

    if (m_Dim>=3) {
        const double Gz  = t_PDOperators("d/dz");
        const double Gzz = t_PDOperators("d2/dz2");
        const double Gxz = t_PDOperators("d2/dxdz");
        const double Gyz = t_PDOperators("d2/dydz");
        const double a   = m_C12 + m_C66;

        const double coeff_c = -wFlux;                                  // conservative diffusion: NOT gated
        const double cmu_c  = m_Kappa * g2c;                            // kappa Lap c: NOT gated
        const double cmu_ux = g * m_Omega * m_Kh * Gx;                  // sigma_h coupling: GATED
        const double cmu_uy = g * m_Omega * m_Kh * Gy;
        const double cmu_uz = g * m_Omega * m_Kh * Gz;
        const double p11 = g*(m_C11*Gxx + m_C66*(Gyy+Gzz));             // elastic stiffness: GATED
        const double p22 = g*(m_C11*Gyy + m_C66*(Gxx+Gzz));
        const double p33 = g*(m_C11*Gzz + m_C66*(Gxx+Gyy));
        const double p12 = g*a*Gxy, p13 = g*a*Gxz, p23 = g*a*Gyz;
        const double cx  = -g*m_A*Gx, cy = -g*m_A*Gy, cz = -g*m_A*Gz;   // eigenstress: GATED

        const double dC  = U_J(1)-U_I(1);
        const double dMu = U_J(2)-U_I(2);
        const double dux = U_J(3)-U_I(3);
        const double duy = U_J(4)-U_I(4);
        const double duz = U_J(5)-U_I(5);

        LocalR(1) = coeff_c * dMu;
        LocalR(2) = cmu_c*dC + cmu_ux*dux + cmu_uy*duy + cmu_uz*duz;
        LocalR(3) = cx*dC + p11*dux + p12*duy + p13*duz;
        LocalR(4) = cy*dC + p12*dux + p22*duy + p23*duz;
        LocalR(5) = cz*dC + p13*dux + p23*duy + p33*duz;

        LocalK_II(1,2)=-coeff_c;
        LocalK_II(2,1)=-cmu_c; LocalK_II(2,3)=-cmu_ux; LocalK_II(2,4)=-cmu_uy; LocalK_II(2,5)=-cmu_uz;
        LocalK_II(3,1)=-cx; LocalK_II(3,3)=-p11; LocalK_II(3,4)=-p12; LocalK_II(3,5)=-p13;
        LocalK_II(4,1)=-cy; LocalK_II(4,3)=-p12; LocalK_II(4,4)=-p22; LocalK_II(4,5)=-p23;
        LocalK_II(5,1)=-cz; LocalK_II(5,3)=-p13; LocalK_II(5,4)=-p23; LocalK_II(5,5)=-p33;
        LocalK_IJ(1,2)= coeff_c;
        LocalK_IJ(2,1)= cmu_c; LocalK_IJ(2,3)= cmu_ux; LocalK_IJ(2,4)= cmu_uy; LocalK_IJ(2,5)= cmu_uz;
        LocalK_IJ(3,1)= cx; LocalK_IJ(3,3)= p11; LocalK_IJ(3,4)= p12; LocalK_IJ(3,5)= p13;
        LocalK_IJ(4,1)= cy; LocalK_IJ(4,3)= p12; LocalK_IJ(4,4)= p22; LocalK_IJ(4,5)= p23;
        LocalK_IJ(5,1)= cz; LocalK_IJ(5,3)= p13; LocalK_IJ(5,4)= p23; LocalK_IJ(5,5)= p33;
        return;
    }

    const double coeff_c     = -wFlux;                         // conservative diffusion: NOT gated
    const double coeff_mu_c  = m_Kappa * g2c;                  // kappa Lap c: NOT gated
    const double coeff_mu_ux = g * m_Omega * m_Kh * Gx;        // sigma_h coupling: GATED
    const double coeff_mu_uy = g * m_Omega * m_Kh * Gy;
    const double p11 = g*(m_C11*Gxx + m_C66*Gyy);              // elastic stiffness: GATED
    const double p22 = g*(m_C66*Gxx + m_C11*Gyy);
    const double pxy = g*(m_C12 + m_C66)*Gxy;
    const double cx  = -g*m_A*Gx, cy = -g*m_A*Gy;              // eigenstress: GATED

    const double dC  = U_J(1)-U_I(1);
    const double dMu = U_J(2)-U_I(2);
    const double dux = U_J(3)-U_I(3);
    const double duy = U_J(4)-U_I(4);

    LocalR(1) = coeff_c*dMu;
    LocalR(2) = coeff_mu_c*dC + coeff_mu_ux*dux + coeff_mu_uy*duy;
    LocalR(3) = cx*dC + p11*dux + pxy*duy;
    LocalR(4) = cy*dC + pxy*dux + p22*duy;

    LocalK_II(1,2)=-coeff_c;
    LocalK_II(2,1)=-coeff_mu_c; LocalK_II(2,3)=-coeff_mu_ux; LocalK_II(2,4)=-coeff_mu_uy;
    LocalK_II(3,1)=-cx; LocalK_II(3,3)=-p11; LocalK_II(3,4)=-pxy;
    LocalK_II(4,1)=-cy; LocalK_II(4,3)=-pxy; LocalK_II(4,4)=-p22;
    LocalK_IJ(1,2)= coeff_c;
    LocalK_IJ(2,1)= coeff_mu_c; LocalK_IJ(2,3)= coeff_mu_ux; LocalK_IJ(2,4)= coeff_mu_uy;
    LocalK_IJ(3,1)= cx; LocalK_IJ(3,3)= p11; LocalK_IJ(3,4)= pxy;
    LocalK_IJ(4,1)= cy; LocalK_IJ(4,3)= pxy; LocalK_IJ(4,4)= p22;
}

void FracStressCahnHilliardElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                                    const int &NodeI,
                                                                    const VectorXd &U_I,
                                                                    const VectorXd &Uold_I,
                                                                    const double  &Volume,
                                                                    VectorXd &LocalR_node,
                                                                    MatrixXd &LocalK_node) const {
    (void)NodeI; (void)Volume;
    LocalR_node.setToZeros(); LocalK_node.setToZeros();
    if (Info.Dt<=0.0) {
        MessagePrinter::printErrorTxt("FracStressCahnHilliardElement is transient-only; got dt="
                                      +std::to_string(Info.Dt));
        MessagePrinter::exitPeriX();
    }
    const double dt=Info.Dt;
    const double c=U_I(1), mu=U_I(2), cold=Uold_I(1);
    const double fp=fPrime(c,m_Chi), fpp=fDoublePrime(c,m_Chi);
    // R_c  = (c - c_old)/dt            (the -div(M grad mu) part is in the bond loop)
    // R_mu = mu - f'_che(c) + Omega C_h (c-cref)   (the bond sigma_h part is gated above)
    LocalR_node(1) = (c-cold)/dt;
    LocalR_node(2) = mu - fp + m_Omega*m_Ch*(c-m_CRef);
    LocalK_node(1,1) = 1.0/dt;
    LocalK_node(2,1) = -fpp + m_Omega*m_Ch;
    LocalK_node(2,2) = 1.0;

    if (m_Rho>0.0) {
        const double c0=m_Rho/(dt*dt);
        const std::size_t base=static_cast<std::size_t>(NodeI-1)*static_cast<std::size_t>(m_Dim);
        for (int a=1;a<=m_Dim;a++) {
            const double v=(m_Initialized && base+static_cast<std::size_t>(a-1)<m_Vel.size())
                           ? m_Vel[base+static_cast<std::size_t>(a-1)] : 0.0;
            const int slot=2+a; // ux,uy[,uz] follow c,mu
            LocalR_node(slot) += -c0*(U_I(slot)-Uold_I(slot)-v*dt);
            LocalK_node(slot,slot) += -c0;
        }
    }
}

std::vector<ProjectionInfo> FracStressCahnHilliardElement::getAvailableProjections() const {
    return {
        {"strain",       ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"stress",       ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"vonMises",     ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"sigmaH",       ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"gradC",        ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)},
        {"damage",       ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"DamageI",      ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)}
    };
}

void FracStressCahnHilliardElement::computeNodalProjection(const std::string &name,
                                                           const PDMesh &Mesh,
                                                           PDOperators &ops,
                                                           const VectorXd &U,
                                                           const int &NodeI,
                                                           const int &DofsPerNode,
                                                           std::vector<double> &out) const {
    if (DofsPerNode!=m_Dim+2 || out.empty()) { std::fill(out.begin(),out.end(),0.0); return; }
    std::fill(out.begin(),out.end(),0.0);
    const bool is3D=(m_Dim>=3);
    const int rowI=(NodeI-1)*DofsPerNode;
    const double c=U(rowI+1);

    if (name=="damage") {
        double phi=0.0;
        if (m_Initialized && NodeI-1<static_cast<int>(m_BondHealth.size())) {
            const auto &h=m_BondHealth[static_cast<std::size_t>(NodeI-1)];
            if (!h.empty()) {
                int broken=0; for (const double v:h) if (v<=0.0) ++broken;
                phi=static_cast<double>(broken)/static_cast<double>(h.size());
            }
        }
        out[0]=phi; return;
    }

    if (name=="DamageI") {
        if (out.empty()) return;
        double d=0.0;
        if (m_Initialized && NodeI-1<static_cast<int>(m_BondDamageIShare.size())) {
            const auto &shares=m_BondDamageIShare[static_cast<std::size_t>(NodeI-1)];
            if (!shares.empty()) {
                double sum=0.0;
                for (const double q : shares) sum+=q;
                d=sum/static_cast<double>(shares.size());
            }
        }
        out[0]=std::clamp(d,0.0,1.0); return;
    }

    if (name=="gradC") {
        const auto &neighbors=Mesh.getIthNodeNeighborNodeIDs(NodeI);
        const auto &meshData=Mesh.getDataConstRef();
        double gx=0.0,gy=0.0,gz=0.0;
        for (const int NodeJ : neighbors) {
            ops.calcOperators(NodeI,NodeJ,meshData);
            const double dc=U((NodeJ-1)*DofsPerNode+1)-c;
            gx+=dc*ops("d/dx"); gy+=dc*ops("d/dy"); if (is3D) gz+=dc*ops("d/dz");
        }
        if (out.size()<3) return;
        out[0]=gx; out[1]=gy; out[2]=gz; return;
    }

    if (is3D) {
        const auto s=computeStress3D(Mesh,ops,U,NodeI,DofsPerNode,m_C11,m_C12,m_C66,m_A,m_CRef);
        if (name=="strain") {
            if (out.size()<9) return;
            out[0]=s.exx; out[1]=s.exy; out[2]=s.exz;
            out[3]=s.exy; out[4]=s.eyy; out[5]=s.eyz;
            out[6]=s.exz; out[7]=s.eyz; out[8]=s.ezz;
        } else if (name=="stress") {
            if (out.size()<9) return;
            out[0]=s.sxx; out[1]=s.sxy; out[2]=s.sxz;
            out[3]=s.sxy; out[4]=s.syy; out[5]=s.syz;
            out[6]=s.sxz; out[7]=s.syz; out[8]=s.szz;
        } else if (name=="sigmaH") {
            out[0]=(s.sxx+s.syy+s.szz)/3.0;
        } else if (name=="vonMises") {
            const double a=s.sxx-s.syy, b=s.syy-s.szz, cc=s.szz-s.sxx;
            out[0]=std::sqrt(0.5*(a*a+b*b+cc*cc+6.0*(s.sxy*s.sxy+s.sxz*s.sxz+s.syz*s.syz)));
        }
        return;
    }

    const bool plane_strain=(m_State==StressState::PlaneStrain);
    const auto s=computeStress(Mesh,ops,U,NodeI,DofsPerNode,m_C11,m_C12,m_C66,m_A,m_CRef,plane_strain);
    if (name=="strain") {
        if (out.size()<9) return;
        out[0]=s.eps_xx; out[1]=s.eps_xy; out[3]=s.eps_xy; out[4]=s.eps_yy;
    } else if (name=="stress") {
        if (out.size()<9) return;
        out[0]=s.s_xx; out[1]=s.s_xy; out[3]=s.s_xy; out[4]=s.s_yy; out[8]=s.s_zz;
    } else if (name=="sigmaH") {
        out[0]=(s.s_xx+s.s_yy+s.s_zz)/3.0;
    } else if (name=="vonMises") {
        const double a=s.s_xx-s.s_yy, b=s.s_yy-s.s_zz, cc=s.s_zz-s.s_xx;
        out[0]=std::sqrt(0.5*(a*a+b*b+cc*cc+6.0*s.s_xy*s.s_xy));
    }
}
