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
//+++ Function: IMPLICIT dynamic-fracture kernel on the strong-form
//+++           PDDO operators
//+++           companion of dynamic_frac_mechanics, and the PARDISO
//+++           driver for the implicit 2D/3D Kalthoff--Winkler
//+++           example.
//+++
//+++             DoFs per node: ux, uy        (2D)
//+++                            ux, uy, uz    (3D)
//+++             isExplicit()=false -> Newton + linear solve path.
//+++
//+++           --------------------------------------------------------
//+++           Equation of motion (force-density form):
//+++             rho * u''_a = L_a(u) + b_a ,
//+++           with L the strong-form Navier--Cauchy operator written in
//+++           the node's own PDDO second derivatives. Per bond (i,j),
//+++           with V_j baked into the operator values g_pq by
//+++           PDOperators and mu_ij in {0,1} the bond health:
//+++             2D:  r1 = mu_ij [ (C11 gxx + C66 gyy) dux
//+++                              + (C12+C66) gxy      duy ]
//+++                  r2 = mu_ij [ (C12+C66) gxy       dux
//+++                              + (C66 gxx + C11 gyy) duy ]
//+++             3D:  r_a = mu_ij [ C11 g_aa + C66 (g_bb + g_cc) ] du_a
//+++                       + mu_ij (C12+C66) g_ab du_b + ... ,
//+++           du = u_j - u_i, C11 = lambda+2mu, C12 = lambda, C66 = mu
//+++           (plane stress / plane strain in 2D). Summed over the
//+++           family this recovers
//+++             L = mu lap(u) + (lambda+mu) grad(div u)
//+++           exactly for polynomial fields up to the PDDO order.
//+++
//+++           --------------------------------------------------------
//+++           TIME INTEGRATION (implicit backward Euler, first order,
//+++           dissipative -- the implicit transient driver supplies dt
//+++           and the step index; this kernel owns the velocity
//+++           history):
//+++             v_n     = (u_n - u_{n-1}) / dt_n          (committed
//+++                       once per converged step from Uold),
//+++             residual R_a = L_a(u) + b_a
//+++                            - rho (u_a - u_a^old - v_a dt)/dt^2 ,
//+++           which is rho*(v_{n+1}-v_n)/dt = L + b with
//+++           v_{n+1} = (u_{n+1}-u_n)/dt. The model is linear between
//+++           breakage events, so Newton converges in 1--2 iterations,
//+++           and the tangent's VALUES are constant while no bond
//+++           breaks and dt is unchanged -- PARDISO reuses one
//+++           factorization across those steps.
//+++
//+++           --------------------------------------------------------
//+++           BOND FAILURE (tensile critical stretch -- irreversible).
//+++           After each converged step, every still-intact bond is
//+++           tested on the committed solution u_n:
//+++             s = (|xi + eta| - |xi|)/|xi| ,  eta = u_j - u_i ,
//+++           and severed (mu_ij: 1 -> 0, never restored) when
//++++            s > s_cr   (tension_only=true, the default), or
//+++             |s| > s_cr (tension_only=false).
//+++           s_cr comes from the classical bond-PD energy calibration
//+++           of the fracture energy G0:
//+++             3D           : s_cr = sqrt( 5 G0 /(6 E delta))
//+++             plane stress : s_cr = sqrt( 4 pi G0 /( 9 E delta))
//+++             plane strain : s_cr = sqrt( 5 pi G0 /(12 E delta)).
//+++           Under a variable horizon the per-bond delta is the
//+++           symmetric 0.5(delta_i+delta_j), so i and j break
//+++           consistently.
//+++
//+++           Pre-existing notches come from the MeshModify block. With
//+++           treatment=force_only the crossing bonds stay in the
//+++           family (the PDDO operators span the slit -- the smeared
//+++           notch of the reference drivers) and this kernel zeroes
//+++           their health at start-up; with the geometric treatment
//+++           the bonds are simply absent from the family. The bond
//+++           health multiplies ONLY the internal-force terms; the
//+++           operators themselves are never rebuilt after breakage,
//+++           matching the frozen-operator convention of the
//+++           Kalthoff--Winkler reference drivers.
//+++
//+++           --------------------------------------------------------
//+++           BOUNDARY CONDITIONS. Natural (traction) faces are NOT
//+++           handled here: register a 'pdtraction' BC per face, which
//+++           replaces the fictitious-layer rows with the strong-form
//+++           sigma.n = t condition built from the same PDDO operators
//+++           (t = 0 gives a genuinely traction-free face). Impact velocity
//+++           is initialized through ICSystem.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <utility>
#include <vector>

#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/FractureCriterion.h"

class PDDODynamicFracElement final : public ElementBase {
public:
    enum class StressState {
        PlaneStress,   ///< 2D plane stress
        PlaneStrain    ///< 2D plane strain
    };

    PDDODynamicFracElement()=default;

    [[nodiscard]] int getDofsPerNode() const override { return m_Dim; }
    [[nodiscard]] std::string getElementType() const override { return "pddo_dynamic_frac"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override {
        if (m_Dim>=3) return {"ux","uy","uz"};
        return {"ux","uy"};
    }

    void setE(const double &E)          { m_E=E;   recalcElasticConstants(); }
    void setNu(const double &nu)        { m_Nu=nu; recalcElasticConstants(); }
    void setRho(const double &rho)      { m_Rho=rho; }
    void setState(const StressState &s) { m_State=s; recalcElasticConstants(); }
    [[nodiscard]] std::string getStateName() const;
    [[nodiscard]] double getE() const   { return m_E; }
    [[nodiscard]] double getNu() const  { return m_Nu; }
    [[nodiscard]] double getRho() const { return m_Rho; }
    [[nodiscard]] double getG0() const  { return m_G0; }
    [[nodiscard]] bool getDamageOn() const     { return m_DamageOn; }
    [[nodiscard]] bool getTensionOnly() const  { return m_TensionOnly; }

    // ---- accessors for the CUDA cached-callback assembler (derived elastic
    //      constants + committed per-step state; state valid after
    //      preprocessIteration). Health is per (node, family slot) exactly as
    //      the mesh neighbour list orders the family. ----
    [[nodiscard]] double getC11() const { return m_C11; }
    [[nodiscard]] double getC12() const { return m_C12; }
    [[nodiscard]] double getC66() const { return m_C66; }
    [[nodiscard]] double getBx() const  { return m_Bx; }
    [[nodiscard]] double getBy() const  { return m_By; }
    [[nodiscard]] double getBz() const  { return m_Bz; }
    [[nodiscard]] const std::vector<double>& getVel() const { return m_Vel; }
    [[nodiscard]] const std::vector<std::vector<double>>& getBondHealth() const { return m_BondHealth; }
    [[nodiscard]] bool getInitialized() const { return m_Initialized; }

    void setDim(const int &dim) override { m_Dim=dim; recalcElasticConstants(); }

    /** fracture energy G0 -> critical stretch via the manuscript calibration. */
    void setG0(const double &g0)        { m_G0=g0; }

    void setDamageOn(const bool &on)    { m_DamageOn=on; }
    void setTensionOnly(const bool &on) { m_TensionOnly=on; }

    void setBodyForce(const double &bx,const double &by,const double &bz) {
        m_Bx=bx; m_By=by; m_Bz=bz;
    }
    /** Uniform initial velocity supplied by ICSystem. */
    void setInitialVelocity(const double &vx,const double &vy,const double &vz) {
        m_V0x=vx; m_V0y=vy; m_V0z=vz;
    }

    /** per-step commit (velocity history + irreversible bond failure) and
     *  one-off state allocation / force-only notch seeding. */
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

    double m_E=0.0;
    double m_Nu=0.25;
    double m_Rho=0.0;
    StressState m_State=StressState::PlaneStrain;

    double m_C11=0.0;
    double m_C12=0.0;
    double m_C66=0.0;

    double m_G0=-1.0;
    bool   m_DamageOn=true;
    bool   m_TensionOnly=true;

    double m_Bx=0.0,m_By=0.0,m_Bz=0.0;
    double m_V0x=0.0,m_V0y=0.0,m_V0z=0.0;

    // ---- per-run state (committed once per step in preprocessIteration;
    //      read-only during the threaded assembly) ----
    mutable bool   m_Initialized=false;
    mutable int    m_CommittedStep=-1;
    mutable double m_LastStepDt=0.0;
    mutable std::vector<double> m_UPrev;   ///< last committed displacement u_{n-1}
    mutable std::vector<double> m_Vel;     ///< committed velocity v_n=(u_n-u_{n-1})/dt
    mutable std::vector<std::vector<double>> m_BondHealth; ///< per (i, family slot)
    /** per node: (neighbour id, family slot) sorted by id, so the per-bond
     *  assembly callback can find its health in O(log family). */
    mutable std::vector<std::vector<std::pair<int,int>>> m_NeighLookup;
};
