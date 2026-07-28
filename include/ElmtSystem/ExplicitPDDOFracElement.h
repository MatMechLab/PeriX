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
//+++ Function: EXPLICIT (central-difference) STRONG-FORM PDDO dynamic
//+++           fracture kernel -- the matrix-free companion of the implicit
//+++           pddo_dynamic_frac (PDDODynamicFracElement). Each node carries
//+++           one displacement component per axis (DoF 1..dim).
//+++
//+++           ----------------------------------------------------------
//+++           Governing equation: elastodynamics rho u'' = Div(sigma) + b,
//+++           with the Cauchy stress divergence written directly on the
//+++           per-node PDDO second-derivative operators (the strong-form
//+++           Navier-Cauchy operator, exactly the implicit kernel's L_i):
//+++             L_i = sum_j health_ij * P_ij (u_j - u_i),
//+++           P_ij the Navier-Cauchy stiffness built from the bond's PDDO
//+++           operators G_xx,G_yy,(G_zz),G_xy,(G_xz,G_yz):
//+++             2D: p11=C11 Gxx+C66 Gyy, p22=C66 Gxx+C11 Gyy,
//+++                 pxy=(C12+C66) Gxy ;
//+++             3D: p11=C11 Gxx+C66(Gyy+Gzz), ..., p12=(C12+C66)Gxy, ...
//+++           The nodal acceleration a_i = (L_i + b)/rho is cached per step
//+++           in preprocessIteration and gathered by the central-difference
//+++           integrator (u^{n+1}=2u^n-u^{n-1}+dt^2 a). The PDDO operators
//+++           are CONSTANT (reference geometry), so they are built once and
//+++           cached; each step is then a matrix-free force evaluation.
//+++
//+++           Fracture: irreversible per-bond tensile critical stretch,
//+++           s = (|y|-|xi|)/|xi|, evaluated each step;
//+++             s_cr = sqrt(5 G0/(6 E delta))      (3D)
//+++                  = sqrt(4 pi G0/(9 E delta))   (plane stress)
//+++                  = sqrt(5 pi G0/(12 E delta))  (plane strain),
//+++           "tension_only": false also breaks compressed bonds. A broken
//+++           bond's force contribution is zeroed
//+++           (the operators are frozen / span the slit), matching the
//+++           implicit kernel and its MeshModify force-only notch.
//+++
//+++           Projections: damage (broken-bond fraction), vonMises, stress,
//+++           strain. CFL-limited (dt < ~delta/c_d); use assemble=openmp on
//+++           CPU and assemble=cuda for the GPU device port.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/FractureCriterion.h"

class ExplicitPDDOFracElement final : public ElementBase {
public:
    enum class StressState { PlaneStress, PlaneStrain };

    ExplicitPDDOFracElement()=default;

    void setE(const double &E)          { m_E=E;   recalcElasticConstants(); }
    void setNu(const double &nu)        { m_Nu=nu; recalcElasticConstants(); }
    void setRho(const double &rho)      { m_Rho=rho; }
    void setG0(const double &g0)        { m_G0=g0; }

    void setTensionOnly(const bool &t)  { m_TensionOnly=t; }
    void setDamageOn(const bool &d)     { m_DamageOn=d; }
    void setState(const StressState &s) { m_State=s; recalcElasticConstants(); }
    void setBodyForce(const double &bx,const double &by,const double &bz) { m_Bx=bx; m_By=by; m_Bz=bz; }

    void setDim(const int &dim) override { m_Dim=dim; recalcElasticConstants(); }

    [[nodiscard]] double getE() const   { return m_E; }
    [[nodiscard]] double getNu() const  { return m_Nu; }
    [[nodiscard]] double getRho() const { return m_Rho; }
    [[nodiscard]] StressState getState() const { return m_State; }

    [[nodiscard]] int getDofsPerNode() const override { return m_Dim; }
    [[nodiscard]] std::string getElementType() const override { return "explicit_pddo_frac"; }
    [[nodiscard]] bool isExplicit() const override { return true; }
    [[nodiscard]] int  getTimeOrder() const override { return 2; }   // rho u'' = F
    /** the strong-form Navier force reads the PDDO second-derivative operators;
     *  they are CONSTANT so the element builds + caches them once itself (it does
     *  not need the assembler to prime them per bond). */
    [[nodiscard]] std::vector<std::string> getDofNames() const override {
        if (m_Dim>=3) return {"ux","uy","uz"};
        return {"ux","uy"};
    }

    /** per step: irreversible bond failure (current u) then the strong-form
     *  PDDO nodal force -> acceleration cache (CPU or GPU). */
    void preprocessIteration(const PDMesh &Mesh,
                             PDOperators &ops,
                             const LocalElmtInfo &Info,
                             const int &DofsPerNode,
                             const VectorXd &U,
                             const VectorXd &Uold) const override;

    /** explicit rate gather: return the cached nodal acceleration a_i. */
    void computeNodalResidual(const LocalElmtInfo &Info,
                              const int &NodeI,
                              const VectorXd &U_I,
                              const VectorXd &Uold_I,
                              const double  &Volume,
                              VectorXd &LocalR_node) const override;

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
    /** critical stretch at horizon delta (state-dependent), 0 => failure off. */
    [[nodiscard]] double criticalStretch(const double &delta) const;

public:
    /** CFL-type stable-dt estimate dx/sqrt(C11/rho) for the explicit driver banner. */
    [[nodiscard]] double estimateStableDt(const PDMesh &Mesh) const override;

private:

    // ---- inputs ----
    double m_E=0.0, m_Nu=0.25, m_Rho=0.0;
    StressState m_State=StressState::PlaneStrain;
    double m_G0=-1.0;
    bool   m_TensionOnly=true, m_DamageOn=true;
    double m_Bx=0.0, m_By=0.0, m_Bz=0.0;

    // ---- derived elastic constants (Navier-Cauchy) ----
    double m_C11=0.0, m_C12=0.0, m_C66=0.0;

    // ---- cached state (built once in preprocessIteration) ----
    mutable bool m_Initialized=false;
    mutable double m_Delta=0.0;
    mutable double m_Sc=0.0;                       // critical stretch at delta
    mutable int m_NOps=3;                          // ops/bond: 5 (2D) or 9 (3D), 1st+2nd derivatives
    mutable std::vector<int> m_NeighOff;           // (N+1) flat neighbour offsets
    mutable std::vector<int> m_NeighId;            // (nbtot) 1-based neighbour ids
    mutable std::vector<double> m_OpCache;         // (nbtot*NOps) per-bond PDDO 1st/2nd-deriv ops
    mutable std::vector<double> m_HealthFlat;      // (nbtot) per-bond health (persistent, irreversible)
    mutable std::vector<double> m_Accel;           // (N*dim) cached acceleration

    // ---- CUDA fallback state ----
    // Once the device path fails mid-run the element stays on the CPU for the
    // remainder (alternating backends would fork the irreversible bond health
    // between host and device); the flags are latched, warn-once.
    mutable bool m_CudaDisabled=false;
};
