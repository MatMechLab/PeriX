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
//+++ Function: SMALL-STRAIN fracture + stress-coupled Cahn-Hilliard.
//+++           A chemo-mechanical Cahn-Hilliard kernel (c, mu, u)
//+++           augmented with IRREVERSIBLE peridynamic bond-stretch
//+++           damage -- the native PD fracture model. There is NO
//+++           separate phase-field damage DoF: fracture lives in
//+++           the bonds.
//+++
//+++           DoFs per node:
//+++             c=1, mu=2, ux=3, uy=4         (2D)
//+++             c=1, mu=2, ux=3, uy=4, uz=5   (3D)
//+++
//+++           ----------------------------------------------------------
//+++           GOVERNING EQUATIONS (small strain, regular-solution f):
//+++             dc/dt = div( M(c) grad mu ),   M(c)=D c(1-c)  (degenerate)
//+++             mu    = f'_che(c) - Omega sigma_h - kappa Lap c
//+++             div( g_d : sigma ) = 0,   sigma = C:(eps - eps*),
//+++             eps* = (Omega/3)(c-cref) I          (swelling eigenstrain)
//+++             sigma_h = (1/3) tr(sigma) = Kh tr(eps) + Ch (c-cref)
//+++           assembled on the direct PDDO operators exactly as the
//+++           stress-CH twin; identical coefficients m_A, m_Kh, m_Ch
//+++           (plane_stress / plane_strain / 3D).
//+++
//+++           ----------------------------------------------------------
//+++           BOND-STRETCH DAMAGE (Silling-Askari critical stretch,
//+++           irreversible -- the PD-native crack model; no phase
//+++           field). Each bond (i,j) carries a health mu_ij in {0,1},
//+++           committed ONCE per converged step from the accepted
//+++           solution Uold. A bond breaks (1->0, never restored) when
//+++           its ELASTIC stretch exceeds the critical stretch:
//+++             s        = (|xi+eta| - |xi|)/|xi| ,  eta = u_j - u_i
//+++             s_swell  = (Omega/3) ( c_bar_ij - cref ),  c_bar=(c_i+c_j)/2
//+++             s_elastic= s - s_swell
//+++             break if   s_elastic > s_cr   (tension_only, default), or
//+++                        |s_elastic| > s_cr (tension_only=false).
//+++           Subtracting the lithiation swelling s_swell is ESSENTIAL:
//+++           uniform (stress-free) swelling stretches every bond but
//+++           must NOT crack -- only the swelling MISMATCH (grad c), which
//+++           generates stress, breaks bonds. This follows the established
//+++           coupled-lithiation-fracture PD criterion
//+++           (s - s_Li, with s_Li = (1/3)log(1+Omega c_bar) linearised).
//+++           s_cr from the classical bond-PD calibration of G0:
//+++             3D           : s_cr = sqrt( 5 G0 /(6 E delta))
//+++             plane stress : s_cr = sqrt( 4 pi G0 /( 9 E delta))
//+++             plane strain : s_cr = sqrt( 5 pi G0 /(12 E delta)).
//+++           Variable horizon uses the
//+++           symmetric 0.5(delta_i+delta_j) so i,j break consistently.
//+++
//+++           ----------------------------------------------------------
//+++           DAMAGE COUPLING. mu_ij multiplies ONLY the MECHANICAL bond
//+++           terms -- the elastic stiffness, the eigenstress (-A grad c)
//+++           body force, and the sigma_h = Omega*Kh*div(u) coupling in
//+++           the mu-equation -- so a stress-free (cracked) fragment
//+++           relaxes, sigma_h -> 0 and the chemo-mechanical coupling
//+++           self-consistently disappears (a crack carries no stress and
//+++           cannot bias the chemical potential). The DIFFUSION (M grad
//+++           mu) and the gradient energy (kappa Lap c) are NOT gated:
//+++           damage here is purely mechanical (a crack neither blocks nor
//+++           fast-tracks Li -- that crack/diffusion interaction is a
//+++           documented extension). A residual-stiffness floor k_res
//+++           keeps a fully-severed node non-singular in the quasi-static
//+++           (inertia-free) matrix:  g_eff = mu_ij>0 ? 1 : k_res.
//+++
//+++           MECHANICAL TIME INTEGRATION. By default the coupled fracture
//+++           mechanics is quasi-static, which is the intended lithiation-
//+++           fracture setting and therefore needs rigid-mode constraints
//+++           under all-traction boundaries. If 'rho'>0 is supplied, the
//+++           displacement rows receive the same backward-Euler inertia as
//+++           pddo_dynamic_frac:
//+++             R_u += -rho (u-u_old-v_old*dt)/dt^2,
//+++           making the Omega=0 dynamic limit comparable to the pure
//+++           pddo_dynamic_frac element without artificial pins.
//+++
//+++           mu_ij is FROZEN during a Newton solve (committed only
//+++           between steps from Uold), so the tangent is exact with no
//+++           d(mu_ij)/dU terms; PARDISO + OpenMP, transient (CH) driver.
//+++
//+++           Projections: strain, stress, vonMises, sigmaH, gradC, and
//+++             damage : phi_i = 1 - n_intact/n_family (PD local damage).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/FractureCriterion.h"

class FracStressCahnHilliardElement final : public ElementBase {
public:
    enum class StressState {
        PlaneStress,
        PlaneStrain
    };

    FracStressCahnHilliardElement()=default;

    // ---- chemo-mechanical setters (same physics as stress_cahnhilliard) ----
    void setE(const double &v)           { m_E=v;      recalcElasticConstants(); }
    void setNu(const double &v)          { m_Nu=v;     recalcElasticConstants(); }
    void setState(const StressState &s)  { m_State=s;  recalcElasticConstants(); }
    void setOmega(const double &v)       { m_Omega=v;  recalcElasticConstants(); }
    void setDiffusivity(const double &v) { m_D=v; }
    void setCref(const double &v)        { m_CRef=v; }
    void setChi(const double &v)         { m_Chi=v; }
    void setKappa(const double &v)       { m_Kappa=v; }
    void setRho(const double &v)         { m_Rho=v; }

    // ---- bond-stretch fracture setters ----
    void setG0(const double &v)          { m_G0=v; }

    void setTensionOnly(const bool &b)   { m_TensionOnly=b; }
    void setDamageOn(const bool &b)      { m_DamageOn=b; }
    void setResidualStiffness(const double &v) { m_ResidualStiffness=v; }

    void setDim(const int &dim) override { m_Dim=dim; recalcElasticConstants(); }

    [[nodiscard]] double getE() const     { return m_E; }
    [[nodiscard]] double getNu() const    { return m_Nu; }
    [[nodiscard]] double getOmega() const { return m_Omega; }
    [[nodiscard]] double getRho() const   { return m_Rho; }
    [[nodiscard]] double getG0() const    { return m_G0; }
    [[nodiscard]] StressState getState() const { return m_State; }
    [[nodiscard]] std::string getStateName() const;

    // ---- accessors for the CUDA assembler (derived constants + host-built caches;
    //      caches valid after preprocessIteration). frac reuses the stress-CH kernel
    //      with m_LapC as both flux/kappa operator + the frozen per-bond health gate
    //      + optional inertia (committed velocity). ----
    [[nodiscard]] double getC11() const   { return m_C11; }
    [[nodiscard]] double getC12() const   { return m_C12; }
    [[nodiscard]] double getC66() const   { return m_C66; }
    [[nodiscard]] double getCoupling() const { return m_A; }
    [[nodiscard]] double getKh() const    { return m_Kh; }
    [[nodiscard]] double getCh() const    { return m_Ch; }
    [[nodiscard]] double getCref() const  { return m_CRef; }
    [[nodiscard]] double getChi() const   { return m_Chi; }
    [[nodiscard]] double getKappa() const { return m_Kappa; }
    [[nodiscard]] double getResidualStiffness() const { return m_ResidualStiffness; }
    [[nodiscard]] const std::vector<int>&    getNeighOff()    const { return m_NeighOff; }
    [[nodiscard]] const std::vector<int>&    getRevBond()     const { return m_Rev; }
    [[nodiscard]] const std::vector<double>& getLapC()        const { return m_LapC; }
    [[nodiscard]] const std::vector<char>&   getGhostFlags()  const { return m_IsGhost; }
    [[nodiscard]] const std::vector<double>& getNodeMobility()const { return m_M; }
    [[nodiscard]] const std::vector<std::vector<double>>& getBondHealth() const { return m_BondHealth; }
    [[nodiscard]] const std::vector<double>& getVel()         const { return m_Vel; }
    [[nodiscard]] bool getInitialized() const { return m_Initialized; }

    [[nodiscard]] int getDofsPerNode() const override { return m_Dim+2; }
    [[nodiscard]] std::string getElementType() const override { return "frac_stress_cahnhilliard"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override {
        if (m_Dim>=3) return {"c","mu","ux","uy","uz"};
        return {"c","mu","ux","uy"};
    }
    // c (slot 0) and mu (slot 1): the species flux and mu-Laplacian drop ghost
    // bonds, so a Dirichlet on either at boundary ghosts is inert (use a flux BC).
    [[nodiscard]] std::vector<int> getGhostDropSpeciesDofSlots() const override { return {0,1}; }

    /** Picard-freeze the degenerate mobility every iteration AND commit the
     *  irreversible bond breakage once per converged step (from Uold). */
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

private:
    void recalcElasticConstants();
    /** critical stretch for a bond whose (symmetric) horizon is delta. */
    [[nodiscard]] double criticalStretch(const double &delta) const;
    /** bond health mu_ij of bond NodeI -> NodeJ (1 intact, 0 broken). */
    [[nodiscard]] double bondHealth(const int &NodeI,const int &NodeJ) const;
    /** family slot of NodeJ in NodeI's neighbour list, -1 if absent. */
    [[nodiscard]] int bondSlot(const int &NodeI,const int &NodeJ) const;

    // ---- elastic / chemo-mechanical constants (recalcElasticConstants) ----
    double m_E=120.0;
    double m_Nu=0.25;
    StressState m_State=StressState::PlaneStress;
    double m_C11=0.0, m_C12=0.0, m_C66=0.0;
    double m_A=0.0, m_Kh=0.0, m_Ch=0.0;

    // ---- Cahn-Hilliard parameters ----
    double m_D=1.0;        // mobility prefactor M0 in M(c)=M0 c(1-c)
    double m_Omega=0.0;    // chemo-mechanical (swelling) coupling
    double m_CRef=0.0;     // stress-free reference concentration
    double m_Chi=3.0;      // regular-solution interaction
    double m_Kappa=1.0e-2; // gradient-energy coefficient
    double m_Rho=0.0;      // optional mechanical density; <=0 => quasi-static mechanics

    // ---- bond-stretch fracture parameters ----
    double m_G0=-1.0;                   // fracture energy (<=0 => failure off)
    bool   m_TensionOnly=true;
    bool   m_DamageOn=true;
    double m_ResidualStiffness=1.0e-8;  // k_res: severed-bond stiffness floor

    // ---- Picard-frozen degenerate-mobility cache (per iteration) ----
    mutable std::vector<double> m_M;       // M_I  = D c_I (1-c_I)

    // ---- conservative species scheme caches (geometry-only, built once) ----
    // Species surface correction: the species flux is the antisymmetric bond form
    //   w_ij = 0.5 ( M_i g2_ij + (V_j/V_i) M_j g2_ji )
    // over BULK bonds only -- bonds to boundary ghosts are dropped, so the
    // boundary influx is carried exclusively by the speciesflux SOURCE (a
    // zero-flux wall contributes nothing) and the discrete species total
    // changes exactly by the prescribed boundary flux (the bulk<->bulk flux
    // telescopes). g2 is the full-family PDDO Laplacian weight rescaled per
    // node by the Madenci/Oterkus surface-correction scalar lambda (so the
    // bulk-only stencil reproduces Lap(|x|^2)=2*dim); lambda is a per-node
    // scalar, so the volume-weighted antisymmetry -- and hence conservation --
    // is untouched.
    mutable std::vector<int>    m_NeighOff; // CSR offsets into the bond arrays (N+1)
    mutable std::vector<int>    m_Rev;      // slot of the reverse bond (j,i); -1 if unpaired
    mutable std::vector<double> m_LapC;     // bulk-only Laplacian weight per bond (0 on ghost bonds)
    mutable std::vector<char>   m_IsGhost;  // boundary-ghost flag (GhostMirrorBulkID>0)
    mutable std::vector<double> m_NodeVol;  // node volumes (antisymmetric weight)

    // ---- irreversible per-bond health (committed once per step from Uold) ----
    mutable bool   m_Initialized=false;
    mutable int    m_CommittedStep=-1;
    mutable double m_LastStepDt=0.0;
    mutable std::vector<double> m_UPrev; // previous committed displacement history
    mutable std::vector<double> m_Vel;   // committed displacement velocity
    mutable std::vector<std::vector<double>> m_BondHealth;            // [i][family slot]
    mutable std::vector<std::vector<double>> m_BondDamageIShare;      // per-bond Mode-I share at break
    mutable std::vector<std::vector<std::pair<int,int>>> m_NeighLookup; // sorted (j, slot)
};
