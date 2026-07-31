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
//+++ Date    : 2026.07.28
//+++ Function: strong small-strain PDDO traction condition
//+++           sigma.n=t, including the manuscript's optional
//+++           isotropic chemical eigenstress.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <array>
#include <vector>

#include "BCSystem/BCBase.h"

class PDTractionBC final : public BCBase {
public:
    enum class StressState { PlaneStress, PlaneStrain };

    PDTractionBC()=default;
    PDTractionBC(const double E,const double nu,const StressState state,
                 const std::array<double,3> &traction);

    [[nodiscard]] std::string getBCType() const override {
        return "pdtraction";
    }
    [[nodiscard]] bool requiresLinearSystem() const override { return true; }

    void setElastic(const double E,const double nu,const StressState state);
    void setTraction(const std::array<double,3> &traction) {
        m_Traction=traction;
    }
    [[nodiscard]] const std::array<double,3>& getTraction() const {
        return m_Traction;
    }

    /**
     * One-based displacement slots. Leave empty for a pure mechanics field
     * laid out as [ux,uy] or [ux,uy,uz].
     */
    void setDisplacementComponents(const std::vector<int> &components) {
        m_DisplacementComponents=components;
    }
    [[nodiscard]] const std::vector<int>& getDisplacementComponents() const {
        return m_DisplacementComponents;
    }

    /**
     * Enable sigma=C:epsilon-A(c-cref)I, with isotropic chemical strain
     * epsilon_chem=Omega(c-cref)I/3. cComponent is a one-based DoF slot.
     */
    void setChemicalExpansion(const double Omega,const double cref,
                              const int cComponent) {
        m_HasChemicalExpansion=true;
        m_Omega=Omega;
        m_CRef=cref;
        m_CComponent=cComponent;
    }
    [[nodiscard]] bool hasChemicalExpansion() const {
        return m_HasChemicalExpansion;
    }
    [[nodiscard]] int getConcentrationComponent() const {
        return m_CComponent;
    }

    void presetSolution(const PDMesh &Mesh,
                        const std::vector<int> &NodeIDs,
                        const int &DofsPerNode,
                        VectorXd &U) const override {
        (void)Mesh;
        (void)NodeIDs;
        (void)DofsPerNode;
        (void)U;
    }

    void apply(const PDMesh &Mesh,
               const std::vector<int> &NodeIDs,
               const int &DofsPerNode,
               const VectorXd &U,
               SparseMatrix &K,
               VectorXd &RHS) const override {
        (void)Mesh;
        (void)NodeIDs;
        (void)DofsPerNode;
        (void)U;
        (void)K;
        (void)RHS;
    }

    void presetControlledRows(const PDMesh &Mesh,
                              const std::vector<int> &NodeIDs,
                              const int &DofsPerNode,
                              std::vector<char> &mask) const override;

    void applyWithOperators(const PDMesh &Mesh,
                            PDOperators &Operators,
                            const std::vector<int> &NodeIDs,
                            const int &DofsPerNode,
                            const VectorXd &U,
                            SparseMatrix &K,
                            VectorXd &RHS) const override;

private:
    double m_E=0.0;
    double m_Nu=0.0;
    StressState m_State=StressState::PlaneStress;
    std::array<double,3> m_Traction{0.0,0.0,0.0};
    std::vector<int> m_DisplacementComponents;

    bool m_ElasticConfigured=false;
    bool m_HasChemicalExpansion=false;
    double m_Omega=0.0;
    double m_CRef=0.0;
    int m_CComponent=1;
};
