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
//+++ Date    : 2026.05.30
//+++ Function: residual / Jacobian assembly for the stress-coupled
//+++           Cahn-Hilliard kernel. Picard linearisation freezes the
//+++           concentration-dependent mobility M(c)=D c (1-c) and its
//+++           gradient at the start of every Newton iteration; the
//+++           remaining elastic, gradient-energy and chemical-potential
//+++           couplings are treated with full Newton.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/StressCahnHilliardElement.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
    // Mobility floor: keep c in (kCEps,1-kCEps) only for the degenerate mobility
    // D c(1-c), which must stay non-negative.
    constexpr double kCEps = 1.0e-8;

    inline double clampC(double c) {
        return std::max(kCEps, std::min(1.0 - kCEps, c));
    }

    // C1 REGULARISED LOGARITHMIC BARRIER for the regular-solution entropy term.
    // True L=ln(c/(1-c)), Lp=1/(c(1-c)) inside [delta,1-delta]; OUTSIDE, L is
    // continued by its tangent line so it is C1, always evaluable, and the boundary
    // stiffness 1/(delta(1-delta)) is retained -- the diverging log confines c to
    // [0,1]. Replaces a hard clamp that froze the restoring force and the Jacobian
    // stiffness outside the bounds, which let c drift to unphysical (negative) values.
    constexpr double kBarrierDelta = 1.0e-3;
    inline void regLog(double c, double delta, double &L, double &Lp) {
        if (c < delta) { const double Lpd=1.0/(delta*(1.0-delta)); L=std::log(delta/(1.0-delta))+Lpd*(c-delta); Lp=Lpd; }
        else if (c > 1.0-delta) { const double cb=1.0-delta, Lpb=1.0/(cb*(1.0-cb)); L=std::log(cb/(1.0-cb))+Lpb*(c-cb); Lp=Lpb; }
        else { L=std::log(c/(1.0-c)); Lp=1.0/(c*(1.0-c)); }
    }

    inline double fPrime(double c, double chi) {
        double L,Lp; regLog(c,kBarrierDelta,L,Lp);
        return L + chi * (1.0 - 2.0 * c);
    }

    inline double fDoublePrime(double c, double chi) {
        double L,Lp; regLog(c,kBarrierDelta,L,Lp);
        return Lp - 2.0 * chi;
    }

    inline double mobility(double c, double D) {
        const double cc = clampC(c);
        return D * cc * (1.0 - cc);
    }

    inline double mobilityPrime(double c, double D) {
        const double cc = clampC(c);
        return D * (1.0 - 2.0 * cc);
    }
}

std::string StressCahnHilliardElement::getStateName() const {
    switch (m_State) {
        case StressState::PlaneStress: return "plane_stress";
        case StressState::PlaneStrain: return "plane_strain";
    }
    return "unknown";
}

void StressCahnHilliardElement::recalcElasticConstants() {
    m_C11 = m_C12 = m_C66 = 0.0;
    m_A = 0.0; m_Kh = 0.0; m_Ch = 0.0;
    if (m_E <= 0.0) return;
    const double nu = m_Nu;
    if (m_Dim >= 3) {
        const double d = (1.0 + nu) * (1.0 - 2.0 * nu);
        if (d <= 0.0) {
            MessagePrinter::printErrorTxt("StressCahnHilliardElement 3D: (1+nu)(1-2nu) must be > 0 (got nu="
                                          + std::to_string(nu) + ")");
            MessagePrinter::exitPeriX();
        }
        m_C11 = m_E * (1.0 - nu) / d;       // lambda + 2 mu
        m_C12 = m_E * nu / d;               // lambda
        m_C66 = m_E / (2.0 * (1.0 + nu));   // mu
        m_A   = m_E * m_Omega / (3.0 * (1.0 - 2.0 * nu));
        m_Kh  = (m_C11 + 2.0 * m_C12) / 3.0;   // 3D bulk modulus K
        m_Ch  = -m_A;
        return;
    }
    if (m_State == StressState::PlaneStress) {
        const double d = 1.0 - nu * nu;
        if (d <= 0.0) {
            MessagePrinter::printErrorTxt("StressCahnHilliardElement plane_stress: nu^2 must be < 1 (got nu="
                                          + std::to_string(nu) + ")");
            MessagePrinter::exitPeriX();
        }
        m_C11 = m_E / d;
        m_C12 = m_E * nu / d;
        m_C66 = m_E / (2.0 * (1.0 + nu));
        m_A   = m_E * m_Omega / (3.0 * (1.0 - nu));
        m_Kh  = (m_C11 + m_C12) / 3.0;
        m_Ch  = -2.0 * m_A / 3.0;
    }
    else {
        const double d = (1.0 + nu) * (1.0 - 2.0 * nu);
        if (d <= 0.0) {
            MessagePrinter::printErrorTxt("StressCahnHilliardElement plane_strain: (1+nu)(1-2nu) must be > 0 (got nu="
                                          + std::to_string(nu) + ")");
            MessagePrinter::exitPeriX();
        }
        m_C11 = m_E * (1.0 - nu) / d;       // lambda + 2 mu
        m_C12 = m_E * nu / d;               // lambda
        m_C66 = m_E / (2.0 * (1.0 + nu));   // mu
        m_A   = m_E * m_Omega / (3.0 * (1.0 - 2.0 * nu));
        m_Kh  = (m_C11 + 2.0 * m_C12) / 3.0;
        m_Ch  = -m_A;
    }
}

int StressCahnHilliardElement::bondSlot(const int i, const int j) const {
    if (i < 1 || static_cast<std::size_t>(i) >= m_Off.size()) return -1;
    const int b = m_Off[static_cast<std::size_t>(i - 1)];
    const int n = m_Off[static_cast<std::size_t>(i)] - b;
    for (int t = 0; t < n; ++t)
        if (m_Nbr[static_cast<std::size_t>(b) + t] == j) return b + t;
    return -1;
}

void StressCahnHilliardElement::preprocessIteration(const PDMesh &Mesh,
                                                    PDOperators &ops,
                                                    const LocalElmtInfo &Info,
                                                    const int &DofsPerNode,
                                                    const VectorXd &U,
                                                    const VectorXd &Uold) const {
    (void)Uold;
    // OpenMP node loop (set when JobSystem.assemble==openmp). Each node fills
    // only its own mobility / grad-mobility cache and reads U read-only; per
    // thread we take a private PDOperators copy because calcAMatrix/calcOperators
    // mutate it. Bitwise identical to serial.
    [[maybe_unused]] const bool useParallel = Info.UseParallel;
    if (DofsPerNode != m_Dim+2) {
        MessagePrinter::printErrorTxt("StressCahnHilliardElement: expected DofsPerNode="
                                      + std::to_string(m_Dim+2) + ", got "
                                      + std::to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }
    if (ops.getOrder() < 2) {
        MessagePrinter::printErrorTxt("StressCahnHilliardElement requires PDMesh.Order >= 2");
        MessagePrinter::exitPeriX();
    }

    const int N = Mesh.getNodesNum();
    const bool is3D = (m_Dim >= 3);
    if (static_cast<int>(m_M.size()) != N) {
        m_M.assign(static_cast<std::size_t>(N), 0.0);
        m_GradMx.assign(static_cast<std::size_t>(N), 0.0);
        m_GradMy.assign(static_cast<std::size_t>(N), 0.0);
        m_GradMz.assign(static_cast<std::size_t>(N), 0.0);
    }

    const auto &meshData = Mesh.getDataConstRef();

    // ---- one-off geometry: CSR neighbour list, per-bond PDDO Laplacian g2_ij,
    //      reverse-bond index (slot of j->i) and node volumes. These feed the
    //      CONSERVATIVE species flux w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji)
    //      assembled in computeBondResidualAndJacobian; the reverse weight g2_ji
    //      is a node-j quantity the per-bond callback cannot see, so it is cached
    //      here. Geometry only ->
    //      built once, reused every Newton iteration. ----
    if (static_cast<int>(m_Off.size()) != N + 1) {
        m_Off.assign(static_cast<std::size_t>(N) + 1, 0);
        for (int i = 1; i <= N; ++i)
            m_Off[static_cast<std::size_t>(i)] = m_Off[static_cast<std::size_t>(i - 1)]
                + static_cast<int>(Mesh.getIthNodeNeighborNodeIDs(i).size());
        const int NB = m_Off[static_cast<std::size_t>(N)];
        m_Nbr.assign(static_cast<std::size_t>(NB), 0);
        m_LapBond.assign(static_cast<std::size_t>(NB), 0.0);
        m_Rev.assign(static_cast<std::size_t>(NB), -1);
        m_Vol.assign(static_cast<std::size_t>(N), 0.0);

        // boundary-ghost flags: a ghost holds one reflected value yet sits in
        // several boundary families, so its off-mirror bonds inject a spurious
        // diffusive flux (over-injection + a corner spike). The species flux and
        // Laplacian therefore DROP ghost bonds (surface correction); mechanics
        // keeps them.
        m_IsGhost.assign(static_cast<std::size_t>(N), 0);
        {
            const auto &gm = meshData.GhostMirrorBulkID;
            if (!gm.empty())
                for (int i = 1; i <= N; ++i)
                    if (i <= static_cast<int>(gm.size()) && gm[static_cast<std::size_t>(i - 1)] > 0)
                        m_IsGhost[static_cast<std::size_t>(i - 1)] = 1;
        }

        for (int i = 1; i <= N; ++i) {
            m_Vol[static_cast<std::size_t>(i - 1)] = Mesh.getIthNodeVolume(i);
            const int b = m_Off[static_cast<std::size_t>(i - 1)];
            ops.calcAMatrix(i, meshData);
            const auto &neigh = Mesh.getIthNodeNeighborNodeIDs(i);
            // raw per-bond PDDO Laplacian, plus a per-node lambda that restores
            // Lap(q)=2*dim for q=|x|^2 on the BULK-ONLY (ghost-excluded) family,
            // so dropping ghost bonds keeps 2nd-order consistency. lambda is a
            // scalar, so the antisymmetric bond flux stays conservative.
            const double xi0 = meshData.NodeCoords[(i - 1) * 3 + 0];
            const double yi0 = meshData.NodeCoords[(i - 1) * 3 + 1];
            const double zi0 = (m_Dim >= 3) ? meshData.NodeCoords[(i - 1) * 3 + 2] : 0.0;
            const double qi  = xi0*xi0 + yi0*yi0 + zi0*zi0;
            double sLap = 0.0, lapq = 0.0;
            for (std::size_t t = 0; t < neigh.size(); ++t) {
                const int j = neigh[t];
                m_Nbr[static_cast<std::size_t>(b) + t] = j;
                ops.calcOperators(i, j, meshData);
                const double w = pdLaplacian(ops);
                m_LapBond[static_cast<std::size_t>(b) + t] = w;
                if (!m_IsGhost[static_cast<std::size_t>(j - 1)]) {
                    sLap += w;
                    const double xj = meshData.NodeCoords[(j - 1) * 3 + 0];
                    const double yj = meshData.NodeCoords[(j - 1) * 3 + 1];
                    const double zj = (m_Dim >= 3) ? meshData.NodeCoords[(j - 1) * 3 + 2] : 0.0;
                    lapq += w * (xj*xj + yj*yj + zj*zj - qi);
                }
            }
            const double lam = (lapq > 1.0e-12) ? (2.0 * static_cast<double>(m_Dim) / lapq) : 1.0;
            for (std::size_t t = 0; t < neigh.size(); ++t)
                if (!m_IsGhost[static_cast<std::size_t>(m_Nbr[static_cast<std::size_t>(b) + t] - 1)])
                    m_LapBond[static_cast<std::size_t>(b) + t] *= lam;
            (void)sLap;
        }
        for (int i = 1; i <= N; ++i) {
            const int bi = m_Off[static_cast<std::size_t>(i - 1)];
            const int ni = m_Off[static_cast<std::size_t>(i)] - bi;
            for (int t = 0; t < ni; ++t) {
                const int j = m_Nbr[static_cast<std::size_t>(bi) + t];
                const int bj = m_Off[static_cast<std::size_t>(j - 1)];
                const int nj = m_Off[static_cast<std::size_t>(j)] - bj;
                int rev = -1;
                for (int s = 0; s < nj; ++s)
                    if (m_Nbr[static_cast<std::size_t>(bj) + s] == i) { rev = bj + s; break; }
                m_Rev[static_cast<std::size_t>(bi) + t] = rev;
            }
        }
    }

#pragma omp parallel if(useParallel)
    {
        PDOperators op = ops;   // per-thread copy (calcAMatrix/calcOperators mutate it)
#pragma omp for schedule(guided)
        for (int i = 1; i <= N; ++i) {
            op.calcAMatrix(i, meshData);
            const int rowI = (i - 1) * DofsPerNode;
            const double c_i = U(rowI + 1);

            double gx = 0.0, gy = 0.0, gz = 0.0;
            for (const int NodeJ : Mesh.getIthNodeNeighborNodeIDs(i)) {
                op.calcOperators(i, NodeJ, meshData);
                const int rowJ = (NodeJ - 1) * DofsPerNode;
                const double dc = U(rowJ + 1) - c_i;
                gx += dc * op("d/dx");
                gy += dc * op("d/dy");
                if (is3D) gz += dc * op("d/dz");
            }
            const std::size_t k = static_cast<std::size_t>(i - 1);
            m_M[k]            = mobility(c_i, m_D);
            const double dMdc = mobilityPrime(c_i, m_D);
            m_GradMx[k]       = dMdc * gx;
            m_GradMy[k]       = dMdc * gy;
            m_GradMz[k]       = dMdc * gz;
        }
    }
}

void StressCahnHilliardElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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
    (void)Info; (void)NodeJ; (void)Uold_I; (void)Uold_J; (void)Volume;

    LocalR.setToZeros();
    LocalK_II.setToZeros();
    LocalK_IJ.setToZeros();

    // V_j is baked into all operator values by PDOperators.
    const double Gx  = t_PDOperators("d/dx");
    const double Gy  = t_PDOperators("d/dy");
    const double Gxx = t_PDOperators("d2/dx2");
    const double Gyy = t_PDOperators("d2/dy2");
    const double Gxy = t_PDOperators("d2/dxdy");
    const double Lap = pdLaplacian(t_PDOperators);

    // CONSERVATIVE (species-conserving) degenerate-mobility flux for div(M grad mu).
    // The antisymmetric bond weight w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji)
    // satisfies V_i w_ij = V_j w_ji, so the interior flux telescopes and the total
    // species changes only by the boundary source. The OLD collocation weight
    //   coeff_c = -(M_i Lap + grad(M)_i . grad)
    // used node-i mobility for BOTH bond directions, so it did NOT telescope and
    // manufactured Li wherever the degenerate mobility M(c)=D c(1-c) varied (the
    // documented ~+37% drift; here it drove the corner to c~1). g2_ij is the cached
    // lambda-corrected Laplacian; g2_ji is its reverse. Bonds to boundary GHOSTS are
    // dropped from the diffusive flux AND the kappa-Laplacian (surface correction)
    // -> exact species conservation.
    // Per-bond residual: r^c += coeff_c*(mu_j-mu_i) = -w_ij*(mu_j-mu_i).
    const std::size_t k = static_cast<std::size_t>(NodeI - 1);
    const bool   ghostJ = (!m_IsGhost.empty()) && m_IsGhost[static_cast<std::size_t>(NodeJ - 1)];
    const int    bo     = bondSlot(NodeI, NodeJ);
    const double g2ij   = (bo >= 0) ? m_LapBond[static_cast<std::size_t>(bo)] : Lap;
    double coeff_c = 0.0;                       // c-flux weight = -w_ij (0 on dropped ghost bonds)
    if (!ghostJ) {
        const double M_i = m_M[k];
        const double M_j = m_M[static_cast<std::size_t>(NodeJ - 1)];
        const double V_i = m_Vol[k];
        const double V_j = m_Vol[static_cast<std::size_t>(NodeJ - 1)];
        const int rev = (bo >= 0) ? m_Rev[static_cast<std::size_t>(bo)] : -1;
        double w_ij;
        if (rev >= 0 && V_i > 0.0)
            w_ij = 0.5 * (M_i * g2ij + (V_j / V_i) * M_j * m_LapBond[static_cast<std::size_t>(rev)]);
        else
            w_ij = M_i * g2ij;   // unpaired bulk bond: forward only
        coeff_c = -w_ij;
    }
    // lambda-corrected kappa-Laplacian weight for the mu equation; dropped on ghost bonds.
    const double kappaLap = ghostJ ? 0.0 : (m_Kappa * g2ij);

    if (m_Dim >= 3) {
        // DoF layout: c=1, mu=2, ux=3, uy=4, uz=5.
        const double Gz  = t_PDOperators("d/dz");
        const double Gzz = t_PDOperators("d2/dz2");
        const double Gxz = t_PDOperators("d2/dxdz");
        const double Gyz = t_PDOperators("d2/dydz");
        const double a   = m_C12 + m_C66;

        // coeff_c (conservative species flux weight) computed above; shared by 2D/3D.
        const double cmu_c  = kappaLap;
        const double cmu_ux = m_Omega * m_Kh * Gx;
        const double cmu_uy = m_Omega * m_Kh * Gy;
        const double cmu_uz = m_Omega * m_Kh * Gz;
        const double p11 = m_C11*Gxx + m_C66*(Gyy+Gzz);
        const double p22 = m_C11*Gyy + m_C66*(Gxx+Gzz);
        const double p33 = m_C11*Gzz + m_C66*(Gxx+Gyy);
        const double p12 = a*Gxy, p13 = a*Gxz, p23 = a*Gyz;
        const double cx  = -m_A*Gx, cy = -m_A*Gy, cz = -m_A*Gz;

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

    // c equation: coeff_c*(mu_j-mu_i) with the conservative weight computed above.

    // mu equation: kappa*Lap*(c_j-c_i) + Omega*Kh*(Gx*dux + Gy*duy)
    // The chemical part of -Omega*sigma_h (the C_h*(c-cref) piece) is a
    // pure nodal term and lives in the nodal Jacobian; here we only carry
    // the bond pieces (which come from -Omega*K_h*(eps_xx+eps_yy)).
    const double coeff_mu_c  = kappaLap;
    const double coeff_mu_ux = m_Omega * m_Kh * Gx;
    const double coeff_mu_uy = m_Omega * m_Kh * Gy;

    // Mechanical PDDO Navier-Cauchy coefficients plus the chemo-mechanical
    // -A grad c body force.
    const double p11 = m_C11 * Gxx + m_C66 * Gyy;
    const double p22 = m_C66 * Gxx + m_C11 * Gyy;
    const double pxy = (m_C12 + m_C66) * Gxy;
    const double cx  = -m_A * Gx;     // R_ux <- cx * (c_j - c_i)
    const double cy  = -m_A * Gy;     // R_uy <- cy * (c_j - c_i)

    const double dC  = U_J(1) - U_I(1);
    const double dMu = U_J(2) - U_I(2);
    const double dux = U_J(3) - U_I(3);
    const double duy = U_J(4) - U_I(4);

    // ---------------- residual ----------------
    LocalR(1) = coeff_c   * dMu;
    LocalR(2) = coeff_mu_c * dC + coeff_mu_ux * dux + coeff_mu_uy * duy;
    LocalR(3) = cx * dC + p11 * dux + pxy * duy;
    LocalR(4) = cy * dC + pxy * dux + p22 * duy;

    // ---------------- Jacobian (LocalK_II = +dR/du at i) -------------
    LocalK_II(1, 2) = -coeff_c;
    LocalK_II(2, 1) = -coeff_mu_c;
    LocalK_II(2, 3) = -coeff_mu_ux;
    LocalK_II(2, 4) = -coeff_mu_uy;
    LocalK_II(3, 1) = -cx;  LocalK_II(3, 3) = -p11; LocalK_II(3, 4) = -pxy;
    LocalK_II(4, 1) = -cy;  LocalK_II(4, 3) = -pxy; LocalK_II(4, 4) = -p22;

    // LocalK_IJ = +dR/du at j = -LocalK_II (since dC = u_j - u_i etc).
    LocalK_IJ(1, 2) =  coeff_c;
    LocalK_IJ(2, 1) =  coeff_mu_c;
    LocalK_IJ(2, 3) =  coeff_mu_ux;
    LocalK_IJ(2, 4) =  coeff_mu_uy;
    LocalK_IJ(3, 1) =  cx;  LocalK_IJ(3, 3) =  p11; LocalK_IJ(3, 4) =  pxy;
    LocalK_IJ(4, 1) =  cy;  LocalK_IJ(4, 3) =  pxy; LocalK_IJ(4, 4) =  p22;
}

void StressCahnHilliardElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                                 const int &NodeI,
                                                                 const VectorXd &U_I,
                                                                 const VectorXd &Uold_I,
                                                                 const double  &Volume,
                                                                 VectorXd &LocalR_node,
                                                                 MatrixXd &LocalK_node) const {
    (void)NodeI; (void)Volume;

    LocalR_node.setToZeros();
    LocalK_node.setToZeros();

    if (Info.Dt <= 0.0) {
        MessagePrinter::printErrorTxt("StressCahnHilliardElement is transient-only; got dt="
                                      + std::to_string(Info.Dt));
        MessagePrinter::exitPeriX();
    }

    const double dt   = Info.Dt;
    const double c    = U_I(1);
    const double mu   = U_I(2);
    const double cold = Uold_I(1);

    const double fp  = fPrime(c, m_Chi);
    const double fpp = fDoublePrime(c, m_Chi);

    // R_c[I]   = (c - c_old)/dt        (the -div(M grad mu) part lives in the bond loop)
    // R_mu[I]  = mu - f'_chem(c) + Omega * C_h * (c - cref)
    //            (the Omega * K_h * (eps_xx+eps_yy) and kappa*Lap c parts live in the bond loop)
    LocalR_node(1) = (c - cold) / dt;
    LocalR_node(2) = mu - fp + m_Omega * m_Ch * (c - m_CRef);
    // R_ux, R_uy have no nodal contribution (no body force, no time derivative).

    LocalK_node(1, 1) = 1.0 / dt;
    LocalK_node(2, 1) = -fpp + m_Omega * m_Ch;
    LocalK_node(2, 2) = 1.0;
}

std::vector<ProjectionInfo> StressCahnHilliardElement::getAvailableProjections() const {
    return {
        {"strain",       ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"stress",       ProjectionType::Tensor, projectionComponentsFor(ProjectionType::Tensor)},
        {"vonMises",     ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"sigmaH",       ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"fPrime",       ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"fDoublePrime", ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"gradC",        ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)}
    };
}

namespace {
    struct LocalStress {
        double eps_xx, eps_yy, eps_xy;
        double s_xx, s_yy, s_xy, s_zz;
    };

    LocalStress computeStress(const PDMesh &Mesh,
                              PDOperators &ops,
                              const VectorXd &U,
                              const int NodeI,
                              const int DofsPerNode,
                              const double C11,
                              const double C12,
                              const double C66,
                              const double A,
                              const double cref,
                              const bool plane_strain) {
        const auto &neighbors = Mesh.getIthNodeNeighborNodeIDs(NodeI);
        const auto &meshData  = Mesh.getDataConstRef();
        const int rowI = (NodeI - 1) * DofsPerNode;
        const double c_i = U(rowI + 1);

        double dux_dx=0.0, dux_dy=0.0, duy_dx=0.0, duy_dy=0.0;
        for (const int NodeJ : neighbors) {
            ops.calcOperators(NodeI, NodeJ, meshData);
            const double gx = ops("d/dx");
            const double gy = ops("d/dy");
            const int rowJ = (NodeJ - 1) * DofsPerNode;
            const double dux = U(rowJ + 3) - U(rowI + 3);
            const double duy = U(rowJ + 4) - U(rowI + 4);
            dux_dx += dux * gx;
            dux_dy += dux * gy;
            duy_dx += duy * gx;
            duy_dy += duy * gy;
        }

        LocalStress s{};
        s.eps_xx = dux_dx;
        s.eps_yy = duy_dy;
        s.eps_xy = 0.5 * (dux_dy + duy_dx);

        const double dc = c_i - cref;
        s.s_xx = C11*s.eps_xx + C12*s.eps_yy - A*dc;
        s.s_yy = C12*s.eps_xx + C11*s.eps_yy - A*dc;
        s.s_xy = 2.0 * C66 * s.eps_xy;
        s.s_zz = plane_strain ? (C12 * (s.eps_xx + s.eps_yy) - A*dc) : 0.0;
        return s;
    }

    struct LocalStress3D {
        double exx,eyy,ezz,exy,exz,eyz;
        double sxx,syy,szz,sxy,sxz,syz;
    };

    LocalStress3D computeStress3D(const PDMesh &Mesh, PDOperators &ops,
                                  const VectorXd &U, const int NodeI,
                                  const int DofsPerNode, const double C11,
                                  const double C12, const double C66,
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

void StressCahnHilliardElement::computeNodalProjection(const std::string &name,
                                                       const PDMesh &Mesh,
                                                       PDOperators &ops,
                                                       const VectorXd &U,
                                                       const int &NodeI,
                                                       const int &DofsPerNode,
                                                       std::vector<double> &out) const {
    if (DofsPerNode != m_Dim+2 || out.empty()) {
        std::fill(out.begin(), out.end(), 0.0);
        return;
    }
    std::fill(out.begin(), out.end(), 0.0);
    const bool is3D = (m_Dim>=3);

    const int rowI = (NodeI - 1) * DofsPerNode;
    const double c = U(rowI + 1);

    if (name == "fPrime")        { out[0] = fPrime(c, m_Chi);       return; }
    if (name == "fDoublePrime")  { out[0] = fDoublePrime(c, m_Chi); return; }

    if (name == "gradC") {
        const auto &neighbors = Mesh.getIthNodeNeighborNodeIDs(NodeI);
        const auto &meshData  = Mesh.getDataConstRef();
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (const int NodeJ : neighbors) {
            ops.calcOperators(NodeI, NodeJ, meshData);
            const double dc = U((NodeJ - 1) * DofsPerNode + 1) - c;
            gx += dc * ops("d/dx");
            gy += dc * ops("d/dy");
            if (is3D) gz += dc * ops("d/dz");
        }
        if (out.size() < 3) return;
        out[0] = gx; out[1] = gy; out[2] = gz;
        return;
    }

    if (is3D) {
        const auto s = computeStress3D(Mesh, ops, U, NodeI, DofsPerNode,
                                       m_C11, m_C12, m_C66, m_A, m_CRef);
        if (name == "strain") {
            if (out.size()<9) return;
            out[0]=s.exx; out[1]=s.exy; out[2]=s.exz;
            out[3]=s.exy; out[4]=s.eyy; out[5]=s.eyz;
            out[6]=s.exz; out[7]=s.eyz; out[8]=s.ezz;
        }
        else if (name == "stress") {
            if (out.size()<9) return;
            out[0]=s.sxx; out[1]=s.sxy; out[2]=s.sxz;
            out[3]=s.sxy; out[4]=s.syy; out[5]=s.syz;
            out[6]=s.sxz; out[7]=s.syz; out[8]=s.szz;
        }
        else if (name == "sigmaH") {
            out[0]=(s.sxx+s.syy+s.szz)/3.0;
        }
        else if (name == "vonMises") {
            const double a=s.sxx-s.syy, b=s.syy-s.szz, cc=s.szz-s.sxx;
            out[0]=std::sqrt(0.5*(a*a+b*b+cc*cc+6.0*(s.sxy*s.sxy+s.sxz*s.sxz+s.syz*s.syz)));
        }
        return;
    }

    const bool plane_strain = (m_State == StressState::PlaneStrain);
    const auto s = computeStress(Mesh, ops, U, NodeI, DofsPerNode,
                                 m_C11, m_C12, m_C66, m_A, m_CRef,
                                 plane_strain);

    if (name == "strain") {
        if (out.size() < 9) return;
        out[0] = s.eps_xx; out[1] = s.eps_xy;
        out[3] = s.eps_xy; out[4] = s.eps_yy;
        return;
    }
    if (name == "stress") {
        if (out.size() < 9) return;
        out[0] = s.s_xx; out[1] = s.s_xy;
        out[3] = s.s_xy; out[4] = s.s_yy;
        out[8] = s.s_zz;
        return;
    }
    if (name == "sigmaH") {
        out[0] = (s.s_xx + s.s_yy + s.s_zz) / 3.0;
        return;
    }
    if (name == "vonMises") {
        const double d1 = s.s_xx - s.s_yy;
        const double d2 = s.s_yy - s.s_zz;
        const double d3 = s.s_zz - s.s_xx;
        out[0] = std::sqrt(0.5 * (d1*d1 + d2*d2 + d3*d3 + 6.0 * s.s_xy * s.s_xy));
        return;
    }
}
