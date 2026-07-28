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
//+++ Function: Cahn-Hilliard phase-field kernel. Two DoFs per PD
//+++           node:
//+++             DoF 1: c     (concentration, conserved field)
//+++             DoF 2: mu    (chemical potential)
//+++
//+++           Free energy:
//+++             F[c] = int { f(c) + (kappa/2) |grad c|^2 } dV
//+++             f(c) = c log(c) + (1-c) log(1-c) + chi * c (1-c)
//+++           User parameters:
//+++             chi    -- regular-solution interaction (chi > 2 gives
//+++                       a double-well and spinodal decomposition)
//+++             kappa  -- gradient-energy / interface-thickness param;
//+++                       resolved interfaces require kappa large
//+++                       enough that sqrt(kappa / |f''(0.5)|) spans
//+++                       a few PD cells
//+++             M      -- mobility (default 1.0)
//+++
//+++           4th-order Cahn-Hilliard is split into two 2nd-order
//+++           equations so the PDDO Order=2 operators are enough:
//+++             eq c : dc/dt - M * laplacian(mu) = 0
//+++             eq mu: mu - f'(c) + kappa * laplacian(c) = 0
//+++           with f'(c) = log(c/(1-c)) + chi*(1-2c). The Jacobian
//+++           includes the f''(c) coupling between the c and mu
//+++           rows so Newton converges quadratically.
//+++
//+++           At equilibrium the chemical potential is uniform; for
//+++           the symmetric regular-solution model that uniform value
//+++           is 0, so the bulk-phase concentrations are exactly the
//+++           roots of f'(c) = 0 (other than the unstable c=0.5
//+++           midpoint). This is what bench/cahnhilliard_equilibrium
//+++           verifies.
//+++
//+++           Species flux: div(M grad mu) is assembled as the
//+++           CONSERVATIVE antisymmetric bond flux
//+++             w_ij = 1/2 ( M_i g2_ij + (V_j/V_i) M_j g2_ji ),
//+++           which telescopes (V_i w_ij = V_j w_ji), so the discrete
//+++           species total sum_i V_i c_i changes only through the
//+++           boundary source -- exact conservation (a closed spinodal
//+++           conserves to machine precision; the older per-node
//+++           collocation M_i Lap(mu)+grad(M).grad(mu) drifted and, under
//+++           a boundary flux, manufactured species). Bonds to boundary
//+++           ghosts are dropped from the flux and the kappa-Laplacian.
//+++
//+++           Boundary conditions: for a SEALED cell, zero flux of c and
//+++           mu on every boundary (homogeneous-Neumann mirror, offset 0
//+++           on both DoFs) -- or leave c,mu unconstrained, the
//+++           conservative flux closes the cell by itself. To DRIVE a
//+++           boundary influx, use the bulk-source speciesflux BC on c,
//+++           paired with neumann 0 on mu. A ghost-gradient condition does
//+++           not inject species because the conservative flux drops ghost bonds.
//+++
//+++           Numerical note: f'(c) and f''(c) diverge at c=0 and c=1.
//+++           To keep Newton robust we clamp c into (eps, 1-eps) before
//+++           evaluating the log / 1/(c(1-c)) terms (eps = 1e-8). The
//+++           clamped Jacobian is consistent with the clamped residual.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "ElmtSystem/ElementBase.h"

class CahnHilliardElement final : public ElementBase {
public:
    CahnHilliardElement()=default;

    void setChi(const double &v)             { m_Chi=v; }
    void setKappa(const double &v)           { m_Kappa=v; }
    void setMobility(const double &v)        { m_M=v; }

    [[nodiscard]] double getChi() const      { return m_Chi; }
    [[nodiscard]] double getKappa() const    { return m_Kappa; }
    [[nodiscard]] double getMobility() const { return m_M; }

    [[nodiscard]] int getDofsPerNode() const override { return 2; }
    [[nodiscard]] std::string getElementType() const override { return "cahnhilliard"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override { return {"c", "mu"}; }
    // c (slot 0) and mu (slot 1): the species flux and mu-Laplacian drop ghost
    // bonds, so a Dirichlet on either at boundary ghosts is inert (use a flux BC).
    [[nodiscard]] std::vector<int> getGhostDropSpeciesDofSlots() const override { return {0,1}; }

    void computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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
                                        MatrixXd &LocalK_IJ) const override;

    void computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                         const int &NodeI,
                                         const VectorXd &U_I,
                                         const VectorXd &Uold_I,
                                         const double  &Volume,
                                         VectorXd &LocalR_node,
                                         MatrixXd &LocalK_node) const override;

    [[nodiscard]] std::vector<ProjectionInfo> getAvailableProjections() const override;

    void computeNodalProjection(const std::string &name,
                                const PDMesh &Mesh,
                                PDOperators &ops,
                                const VectorXd &U,
                                const int &NodeI,
                                const int &DofsPerNode,
                                std::vector<double> &out) const override;

    // Freeze the concentration-dependent (degenerate) mobility M(c)=M*c(1-c) and
    // its gradient per node before each Newton assembly (Picard linearisation), so
    // the c-equation flux div(M grad mu) stays linear in mu within the step.
    void preprocessIteration(const PDMesh &Mesh,
                             PDOperators &ops,
                             const LocalElmtInfo &Info,
                             const int &DofsPerNode,
                             const VectorXd &U,
                             const VectorXd &Uold) const override;

    // CSR slot of bond (i->j) in m_Nbr, or -1 if j is not a neighbour of i.
    [[nodiscard]] int bondSlot(const int i, const int j) const;

    // ---- accessors for the CUDA bespoke conservative-flux assembler. Valid
    //      after preprocessIteration() has run: it builds the one-off geometry
    //      (neighbour CSR, lambda-corrected Laplacian, reverse-bond map, node
    //      volumes, ghost flags) and refreshes the Picard-frozen mobility. ----
    [[nodiscard]] const std::vector<int>&    getNbrOffsets()   const { return m_Off; }
    [[nodiscard]] const std::vector<int>&    getNbrIds()       const { return m_Nbr; }
    [[nodiscard]] const std::vector<int>&    getRevBond()      const { return m_Rev; }
    [[nodiscard]] const std::vector<double>& getLapBond()      const { return m_LapBond; }
    [[nodiscard]] const std::vector<char>&   getGhostFlags()   const { return m_IsGhost; }
    [[nodiscard]] const std::vector<double>& getNodeVolumes()  const { return m_Vol; }
    [[nodiscard]] const std::vector<double>& getNodeMobility() const { return m_Mnode; }

private:
    double m_Chi   = 3.0;
    double m_Kappa = 5.0e-3;
    double m_M     = 1.0;   // mobility PREFACTOR M0 in M(c)=M0*c*(1-c)

    // Picard-frozen per-node mobility caches (preprocessIteration), 2D.
    mutable std::vector<double> m_Mnode;   // M_I = M0 c_I (1-c_I)
    mutable std::vector<double> m_GradMx;  // (dM/dc)_I * (dc/dx)_I = M0(1-2c_I)*(dc/dx)_I
    mutable std::vector<double> m_GradMy;  // (dM/dc)_I * (dc/dy)_I

    // ---- one-off geometry caches for the CONSERVATIVE species flux ----
    // The c-equation flux div(M grad mu) uses the antisymmetric bond weight
    //   w_ij = 0.5 ( M_i g2_ij + (V_j/V_i) M_j g2_ji ),
    // which telescopes (V_i w_ij = V_j w_ji) and is therefore species-conservative
    // even for the degenerate mobility M(c)=M0 c(1-c). The reverse weight g2_ji is
    // a node-j quantity unavailable in the per-bond callback, so the per-bond PDDO
    // Laplacian (lambda surface-corrected on the bulk family) and a reverse-bond
    // index are cached once from the geometry. Ghost bonds are dropped from the flux
    // and the kappa-Laplacian; boundary influx is carried by the source-form
    // speciesflux condition.
    mutable std::vector<int>    m_Off;      // CSR neighbour offsets, size N+1
    mutable std::vector<int>    m_Nbr;      // CSR neighbour ids,     size NB
    mutable std::vector<int>    m_Rev;      // reverse-bond index (-1 if unpaired)
    mutable std::vector<double> m_LapBond;  // g2_ij per bond (lambda-corrected bulk Laplacian)
    mutable std::vector<char>   m_IsGhost;  // boundary-ghost flag per node (size N)
    mutable std::vector<double> m_Vol;      // node volume V_i, size N
};
