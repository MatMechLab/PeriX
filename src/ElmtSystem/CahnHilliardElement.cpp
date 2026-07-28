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
//+++ Date    : 2026.05.25
//+++ Function: Cahn-Hilliard kernel residual/Jacobian assembly,
//+++           including the f'(c) and f''(c) free-energy
//+++           contributions and the kappa*Laplacian gradient term.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/CahnHilliardElement.h"

#include <algorithm>
#include <cmath>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
    // Keep c strictly inside (eps, 1-eps) so log(c) and 1/(c(1-c)) stay
    // finite even if a transient Newton overshoot pushes c slightly out
    // of physical bounds.
    constexpr double kCEps = 1.0e-8;

    inline double clampC(double c) {
        return std::max(kCEps, std::min(1.0 - kCEps, c));
    }

    inline double fPrime(double c, double chi) {
        const double cc = clampC(c);
        return std::log(cc / (1.0 - cc)) + chi * (1.0 - 2.0 * cc);
    }

    inline double fDoublePrime(double c, double chi) {
        const double cc = clampC(c);
        return 1.0 / (cc * (1.0 - cc)) - 2.0 * chi;
    }

    // Degenerate Cahn-Hilliard mobility M(c)=M0 c(1-c) and dM/dc=M0(1-2c); the
    // factor c(1-c) makes the flux vanish at the saturation limits c=0,1.
    inline double mobility(double c, double M0) {
        const double cc = clampC(c);
        return M0 * cc * (1.0 - cc);
    }
    inline double mobilityPrime(double c, double M0) {
        const double cc = clampC(c);
        return M0 * (1.0 - 2.0 * cc);
    }
}

int CahnHilliardElement::bondSlot(const int i, const int j) const {
    if (i < 1 || static_cast<std::size_t>(i) >= m_Off.size()) return -1;
    const int b = m_Off[static_cast<std::size_t>(i - 1)];
    const int n = m_Off[static_cast<std::size_t>(i)] - b;
    for (int t = 0; t < n; ++t)
        if (m_Nbr[static_cast<std::size_t>(b) + t] == j) return b + t;
    return -1;
}

void CahnHilliardElement::preprocessIteration(const PDMesh &Mesh,
                                              PDOperators &ops,
                                              const LocalElmtInfo &Info,
                                              const int &DofsPerNode,
                                              const VectorXd &U,
                                              const VectorXd &Uold) const {
    (void)Uold;
    [[maybe_unused]] const bool useParallel = Info.UseParallel;
    if (DofsPerNode != 2) {
        MessagePrinter::printErrorTxt("CahnHilliardElement: expected DofsPerNode=2, got "
                                      + std::to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }
    const int N = Mesh.getNodesNum();
    if (static_cast<int>(m_Mnode.size()) != N) {
        m_Mnode.assign(static_cast<std::size_t>(N), 0.0);
        m_GradMx.assign(static_cast<std::size_t>(N), 0.0);
        m_GradMy.assign(static_cast<std::size_t>(N), 0.0);
    }
    const auto &meshData = Mesh.getDataConstRef();

    // ---- one-off geometry: CSR neighbour list, per-bond PDDO Laplacian g2_ij
    //      (lambda-corrected on the bulk family), reverse-bond index, ghost flags
    //      and node volumes. These feed the CONSERVATIVE species flux
    //      w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji) assembled in the bond
    //      callback; the reverse weight g2_ji is a node-j quantity the per-bond
    //      callback cannot see. Geometry only -> built once. ----
    if (static_cast<int>(m_Off.size()) != N + 1) {
        const int dim = (m_Dim >= 3) ? 3 : 2;
        m_Off.assign(static_cast<std::size_t>(N) + 1, 0);
        for (int i = 1; i <= N; ++i)
            m_Off[static_cast<std::size_t>(i)] = m_Off[static_cast<std::size_t>(i - 1)]
                + static_cast<int>(Mesh.getIthNodeNeighborNodeIDs(i).size());
        const int NB = m_Off[static_cast<std::size_t>(N)];
        m_Nbr.assign(static_cast<std::size_t>(NB), 0);
        m_LapBond.assign(static_cast<std::size_t>(NB), 0.0);
        m_Rev.assign(static_cast<std::size_t>(NB), -1);
        m_Vol.assign(static_cast<std::size_t>(N), 0.0);
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
            const double xi0 = meshData.NodeCoords[(i - 1) * 3 + 0];
            const double yi0 = meshData.NodeCoords[(i - 1) * 3 + 1];
            const double zi0 = (dim >= 3) ? meshData.NodeCoords[(i - 1) * 3 + 2] : 0.0;
            const double qi  = xi0*xi0 + yi0*yi0 + zi0*zi0;
            double lapq = 0.0;
            for (std::size_t t = 0; t < neigh.size(); ++t) {
                const int j = neigh[t];
                m_Nbr[static_cast<std::size_t>(b) + t] = j;
                ops.calcOperators(i, j, meshData);
                const double w = pdLaplacian(ops);
                m_LapBond[static_cast<std::size_t>(b) + t] = w;
                if (!m_IsGhost[static_cast<std::size_t>(j - 1)]) {
                    const double xj = meshData.NodeCoords[(j - 1) * 3 + 0];
                    const double yj = meshData.NodeCoords[(j - 1) * 3 + 1];
                    const double zj = (dim >= 3) ? meshData.NodeCoords[(j - 1) * 3 + 2] : 0.0;
                    lapq += w * (xj*xj + yj*yj + zj*zj - qi);
                }
            }
            const double lam = (lapq > 1.0e-12) ? (2.0 * static_cast<double>(dim) / lapq) : 1.0;
            for (std::size_t t = 0; t < neigh.size(); ++t)
                if (!m_IsGhost[static_cast<std::size_t>(m_Nbr[static_cast<std::size_t>(b) + t] - 1)])
                    m_LapBond[static_cast<std::size_t>(b) + t] *= lam;
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
            double gx = 0.0, gy = 0.0;
            for (const int NodeJ : Mesh.getIthNodeNeighborNodeIDs(i)) {
                op.calcOperators(i, NodeJ, meshData);
                const double dc = U((NodeJ - 1) * DofsPerNode + 1) - c_i;
                gx += dc * op("d/dx"); gy += dc * op("d/dy");
            }
            const std::size_t k = static_cast<std::size_t>(i - 1);
            m_Mnode[k]  = mobility(c_i, m_M);
            const double dMdc = mobilityPrime(c_i, m_M);
            m_GradMx[k] = dMdc * gx;
            m_GradMy[k] = dMdc * gy;
        }
    }
}

void CahnHilliardElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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
    (void)Info; (void)Uold_I; (void)Uold_J;

    LocalR.setToZeros();
    LocalK_II.setToZeros();
    LocalK_IJ.setToZeros();

    // V_j is already baked into the operator value by PDOperators.
    const double Lap = pdLaplacian(t_PDOperators);

    //  eq c (DoF 1): -div(M grad mu) assembled as a CONSERVATIVE antisymmetric bond
    //  flux w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji). Because V_i w_ij = V_j w_ji
    //  the interior flux telescopes, so the species total changes only through the
    //  boundary source; bonds to boundary ghosts are dropped. The old collocation
    //  weight -(M_i Lap + grad(M).grad) used the node-i mobility for BOTH bond
    //  directions, did not telescope, and manufactured species under a boundary flux
    //  with the degenerate mobility M=M0 c(1-c).
    //  eq mu (DoF 2): +kappa * Lap(c) couples through (c_j - c_i); ghost bonds dropped.
    //  M(c) is Picard-frozen per node in preprocessIteration; g2_ij is the cached
    //  lambda-corrected Laplacian, g2_ji its reverse.
    const std::size_t ki = static_cast<std::size_t>(NodeI - 1);
    const bool   ghostJ = (!m_IsGhost.empty()) && m_IsGhost[static_cast<std::size_t>(NodeJ - 1)];
    const int    bo     = bondSlot(NodeI, NodeJ);
    const double g2ij   = (bo >= 0) ? m_LapBond[static_cast<std::size_t>(bo)] : Lap;
    double coeff_c = 0.0;                       // = -w_ij (0 on dropped ghost bonds)
    if (!ghostJ) {
        const double M_i = m_Mnode.empty() ? mobility(U_I(1), m_M) : m_Mnode[ki];
        const double M_j = m_Mnode.empty() ? mobility(U_J(1), m_M)
                                           : m_Mnode[static_cast<std::size_t>(NodeJ - 1)];
        const double V_i = m_Vol.empty() ? Volume : m_Vol[ki];
        const double V_j = m_Vol.empty() ? Volume : m_Vol[static_cast<std::size_t>(NodeJ - 1)];
        const int rev = (bo >= 0) ? m_Rev[static_cast<std::size_t>(bo)] : -1;
        double w_ij;
        if (rev >= 0 && V_i > 0.0)
            w_ij = 0.5 * (M_i * g2ij + (V_j / V_i) * M_j * m_LapBond[static_cast<std::size_t>(rev)]);
        else
            w_ij = M_i * g2ij;   // unpaired bulk bond: forward only
        coeff_c = -w_ij;
    }
    const double coeff_mu = ghostJ ? 0.0 : (m_Kappa * g2ij);   // R_mu += coeff_mu*(c_j-c_i)

    const double dC  = U_J(1) - U_I(1);
    const double dMu = U_J(2) - U_I(2);

    LocalR(1) = coeff_c  * dMu;
    LocalR(2) = coeff_mu * dC;

    // dR(a)/du_i(b) and dR(a)/du_j(b): same structure as the Poisson /
    // diffusion kernels — for R = c_coeff*(u_j - u_i) we get -c_coeff
    // at i and +c_coeff at j on the matching DoF.
    //
    //   R_c  depends on mu_j, mu_i        -> K[1,2] entries
    //   R_mu depends on c_j , c_i         -> K[2,1] entries
    //   The other K entries from the bond are 0; the nodal block fills
    //   in the c-c, mu-mu, and cross f''(c) couplings.
    LocalK_II(1, 2) = -coeff_c;
    LocalK_IJ(1, 2) =  coeff_c;
    LocalK_II(2, 1) = -coeff_mu;
    LocalK_IJ(2, 1) =  coeff_mu;
}

void CahnHilliardElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
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
        MessagePrinter::printErrorTxt("CahnHilliardElement is transient-only; got dt="
                                      + std::to_string(Info.Dt));
        MessagePrinter::exitPeriX();
    }

    const double dt    = Info.Dt;
    const double c     = U_I(1);
    const double mu    = U_I(2);
    const double cold  = Uold_I(1);

    const double fp  = fPrime(c, m_Chi);
    const double fpp = fDoublePrime(c, m_Chi);

    // Strong-form residual at NodeI (sum over horizon then gives
    // -M*Lap(mu) and +kappa*Lap(c) automatically):
    //   R_c  = (c - c_old) / dt   - M * Lap(mu)
    //   R_mu = mu - f'(c)          + kappa * Lap(c)
    LocalR_node(1) = (c - cold) / dt;
    LocalR_node(2) = mu - fp;

    // Jacobian (nodal piece):
    //   dR_c  / dc    = 1/dt
    //   dR_c  / dmu   = 0
    //   dR_mu / dc    = -f''(c)
    //   dR_mu / dmu   = 1
    LocalK_node(1, 1) = 1.0 / dt;
    LocalK_node(1, 2) = 0.0;
    LocalK_node(2, 1) = -fpp;
    LocalK_node(2, 2) = 1.0;
}

std::vector<ProjectionInfo> CahnHilliardElement::getAvailableProjections() const {
    return {
        {"fPrime",       ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)},
        {"fDoublePrime", ProjectionType::Scalar, projectionComponentsFor(ProjectionType::Scalar)}
    };
}

void CahnHilliardElement::computeNodalProjection(const std::string &name,
                                                 const PDMesh &Mesh,
                                                 PDOperators &ops,
                                                 const VectorXd &U,
                                                 const int &NodeI,
                                                 const int &DofsPerNode,
                                                 std::vector<double> &out) const {
    (void)Mesh; (void)ops;

    if (DofsPerNode < 2 || out.empty()) {
        std::fill(out.begin(), out.end(), 0.0);
        return;
    }

    const int rowI = (NodeI - 1) * DofsPerNode;
    const double c = U(rowI + 1);

    if (name == "fPrime") {
        out[0] = fPrime(c, m_Chi);
    }
    else if (name == "fDoublePrime") {
        out[0] = fDoublePrime(c, m_Chi);
    }
    else {
        std::fill(out.begin(), out.end(), 0.0);
    }
}
