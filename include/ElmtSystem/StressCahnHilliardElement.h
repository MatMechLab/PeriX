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
//+++ Function: stress-coupled Cahn-Hilliard (chemo-mechanical phase
//+++           field) PDDO kernel. Four DoFs per PD node:
//+++             DoF 1: c   (concentration, conserved field)
//+++             DoF 2: mu  (chemical potential)
//+++             DoF 3: u_x (in-plane displacement, x)
//+++             DoF 4: u_y (in-plane displacement, y)
//+++
//+++           Governing equations (strong form):
//+++             div(sigma) = 0                                  (mechanics)
//+++             dc/dt = div(M grad mu)         with M = D c (1-c)
//+++             mu    = df/dc - kappa * Lap c
//+++
//+++           Constitutive law (chemical eigenstrain):
//+++             sigma_ij = C_ijkl ( eps_kl - delta_kl (c-cref) Omega/3 )
//+++
//+++           Total free energy:
//+++             f = c log c + (1-c) log(1-c) + chi c (1-c)
//+++                 + (1/2) sigma_ij ( eps_ij - delta_ij (c-cref) Omega/3 )
//+++           Differentiating w.r.t. c gives
//+++             df/dc = log(c/(1-c)) + chi(1-2c) - Omega sigma_h,
//+++           where sigma_h = tr(sigma)/3 is the 3D hydrostatic
//+++           stress. The mu equation therefore reads
//+++             mu = log(c/(1-c)) + chi(1-2c) - Omega sigma_h - kappa Lap c.
//+++
//+++           sigma_h depends linearly on the in-plane strain trace
//+++           (eps_xx + eps_yy) and on (c-cref):
//+++             plane stress :  sigma_h = (C11+C12)/3 * (eps_xx+eps_yy)
//+++                                       - (2A/3) * (c-cref)
//+++             plane strain :  sigma_h = (C11+2C12)/3 * (eps_xx+eps_yy)
//+++                                       - A * (c-cref)
//+++           A = E Omega / (3(1-nu))      (plane stress)
//+++           A = E Omega / (3(1-2nu))     (plane strain)
//+++           are the chemo-mechanical coupling coefficients for the
//+++           swelling eigenstrain.
//+++
//+++           Linearisation: full Newton for the elastic and gradient
//+++           operators (which are linear in u, c, mu). Picard
//+++           linearisation for the concentration-dependent mobility
//+++           M(c) = D c (1-c): preprocessIteration freezes M_I per node
//+++           and caches the per-bond PDDO Laplacian + a reverse-bond
//+++           index. The c-equation flux div(M grad mu) is then a
//+++           CONSERVATIVE antisymmetric bond flux
//+++             R_c -= w_IJ (mu_J - mu_I),
//+++             w_IJ = 1/2 ( M_I g2_IJ + (V_J/V_I) M_J g2_JI ),
//+++           which satisfies V_I w_IJ = V_J w_JI, so the interior flux
//+++           telescopes -> exact species conservation. Bonds to boundary
//+++           ghosts are dropped from the flux AND the kappa-Laplacian
//+++           (a per-node lambda = 2 dim / Lap_bulk(|x|^2) surface
//+++           correction keeps the ghost-excluded Laplacian consistent).
//+++           The earlier collocation form
//+++             R_c -= [ M_I Lap + grad(M)_I . grad ] (mu_J - mu_I)
//+++           used node-I mobility for BOTH bond directions and was NOT
//+++           species-conservative when M(c) varied (it manufactured Li).
//+++
//+++           f' and f'' diverge at c = 0 and c = 1; both are clamped
//+++           to (eps, 1-eps) with eps = 1e-8 so transient Newton
//+++           overshoots stay finite. The clamped Jacobian matches the
//+++           clamped residual.
//+++
//+++           Boundary conditions:
//+++             * zero flux of c and mu      -> NeumannBC offset 0 on
//+++                                             both DoFs;
//+++             * rigid-body removal         -> DirichletBC on u_x at
//+++                                             one edge and u_y at
//+++                                             another (e.g. left
//+++                                             u_x=0, bottom u_y=0);
//+++             * boundary flux on c         -> a BULK-SOURCE flux BC
//+++                                             (speciesflux / cyclicflux)
//+++                                             on the boundary-ghost
//+++                                             group; it injects
//+++                                             j / thickness on the
//+++                                             near-wall bulk c-row. The
//+++                                             old mu-gradient-at-ghost
//+++                                             trick is NOT supported: the
//+++                                             conservative flux drops
//+++                                             ghost bonds.
//+++
//+++           Available projections: "strain", "stress", "vonMises",
//+++           "sigmaH", "fPrime", "fDoublePrime", "gradC".
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>
#include <vector>

#include "ElmtSystem/ElementBase.h"

class StressCahnHilliardElement final : public ElementBase {
public:
    enum class StressState {
        PlaneStress,
        PlaneStrain
    };

    StressCahnHilliardElement()=default;

    void setE(const double &v)             { m_E=v;       recalcElasticConstants(); }
    void setNu(const double &v)            { m_Nu=v;      recalcElasticConstants(); }
    void setState(const StressState &s)    { m_State=s;   recalcElasticConstants(); }
    void setOmega(const double &v)         { m_Omega=v;   recalcElasticConstants(); }
    void setCref(const double &v)          { m_CRef=v; }
    void setDiffusivity(const double &v)   { m_D=v; }
    void setChi(const double &v)           { m_Chi=v; }
    void setKappa(const double &v)         { m_Kappa=v; }

    // dimension is set after construction; refresh the elastic constants.
    void setDim(const int &dim) override   { m_Dim=dim; recalcElasticConstants(); }

    [[nodiscard]] double getE() const            { return m_E; }
    [[nodiscard]] double getNu() const           { return m_Nu; }
    [[nodiscard]] StressState getState() const   { return m_State; }
    [[nodiscard]] std::string getStateName() const;
    [[nodiscard]] double getOmega() const        { return m_Omega; }
    [[nodiscard]] double getCref() const         { return m_CRef; }
    [[nodiscard]] double getDiffusivity() const  { return m_D; }
    [[nodiscard]] double getChi() const          { return m_Chi; }
    [[nodiscard]] double getKappa() const        { return m_Kappa; }
    [[nodiscard]] double getC11() const          { return m_C11; }
    [[nodiscard]] double getC12() const          { return m_C12; }
    [[nodiscard]] double getC66() const          { return m_C66; }
    [[nodiscard]] double getCoupling() const     { return m_A; }
    [[nodiscard]] double getKh() const           { return m_Kh; }
    [[nodiscard]] double getCh() const           { return m_Ch; }

    // DoF layout: concentration, chemical potential, then displacements.
    [[nodiscard]] int getDofsPerNode() const override { return m_Dim+2; }
    [[nodiscard]] std::string getElementType() const override { return "stress_cahnhilliard"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override {
        if (m_Dim>=3) return {"c", "mu", "ux", "uy", "uz"};
        return {"c", "mu", "ux", "uy"};
    }
    // c (slot 0) and mu (slot 1): the species flux and mu-Laplacian drop ghost
    // bonds, so a Dirichlet on either at boundary ghosts is inert (use a flux BC).
    [[nodiscard]] std::vector<int> getGhostDropSpeciesDofSlots() const override { return {0,1}; }

    void preprocessIteration(const PDMesh &Mesh,
                             PDOperators &ops,
                             const LocalElmtInfo &Info,
                             const int &DofsPerNode,
                             const VectorXd &U,
                             const VectorXd &Uold) const override;

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

    // ---- accessors for the CUDA bespoke assembler (host-built conservative-flux
    //      geometry + Picard-frozen mobility; valid after preprocessIteration).
    //      The neighbour CSR (m_Off/m_Nbr) matches the mesh neighbour list, so the
    //      device kernel indexes lapbond/rev with the flattened neigh_off/id. ----
    [[nodiscard]] const std::vector<int>&    getRevBond()      const { return m_Rev; }
    [[nodiscard]] const std::vector<double>& getLapBond()      const { return m_LapBond; }
    [[nodiscard]] const std::vector<char>&   getGhostFlags()   const { return m_IsGhost; }
    [[nodiscard]] const std::vector<double>& getNodeMobility() const { return m_M; }

private:
    void recalcElasticConstants();

    // CSR slot of bond (i->j) in m_Nbr, or -1 if j is not a neighbour of i.
    [[nodiscard]] int bondSlot(const int i, const int j) const;

    // ---- user inputs --------------------------------------------------
    double m_E      = 1.0;
    double m_Nu     = 0.3;
    StressState m_State = StressState::PlaneStress;
    double m_D      = 1.0;
    double m_Omega  = 0.0;
    double m_CRef   = 0.0;
    double m_Chi    = 0.0;
    double m_Kappa  = 0.0;

    // ---- derived elastic constants ------------------------------------
    double m_C11    = 0.0;
    double m_C12    = 0.0;
    double m_C66    = 0.0;
    double m_A      = 0.0;  // chemo-mechanical coupling, see header docstring
    double m_Kh     = 0.0;  // sigma_h = m_Kh * (eps_xx+eps_yy) + m_Ch * (c-cref)
    double m_Ch     = 0.0;

    // ---- Picard-frozen per-node caches (preprocessIteration) ---------
    mutable std::vector<double> m_M;       // M_I  = D c_I (1-c_I)
    mutable std::vector<double> m_GradMx;  // (dM/dc)_I * (d c / d x)_I
    mutable std::vector<double> m_GradMy;  // (dM/dc)_I * (d c / d y)_I
    mutable std::vector<double> m_GradMz;  // (dM/dc)_I * (d c / d z)_I (3D)

    // ---- one-off geometry caches for the CONSERVATIVE species flux ----
    // The c-equation flux div(M grad mu) is discretised with an
    // antisymmetric bond weight,
    //   w_ij = 0.5 ( M_i g2_ij + (V_j/V_i) M_j g2_ji ),
    // which telescopes (V_i w_ij = V_j w_ji) and is therefore species-
    // conservative even when the degenerate mobility M(c)=D c(1-c) varies
    // strongly across the domain. The reverse weight g2_ji is a NODE-j
    // quantity unavailable in the per-bond callback, so the per-bond PDDO
    // Laplacian and a reverse-bond index are cached once from the geometry.
    mutable std::vector<int>    m_Off;      // CSR neighbour offsets, size N+1
    mutable std::vector<int>    m_Nbr;      // CSR neighbour ids,     size NB
    mutable std::vector<int>    m_Rev;      // reverse-bond index (-1 if unpaired)
    mutable std::vector<double> m_LapBond;  // g2_ij per bond: PDDO Laplacian, lambda-
                                            // surface-corrected on the bulk (non-ghost)
                                            // family so Lap(|x|^2)=2*dim one-sided
    mutable std::vector<char>   m_IsGhost;  // boundary-ghost flag per node (size N)
    mutable std::vector<double> m_Vol;      // node volume V_i, size N
};
