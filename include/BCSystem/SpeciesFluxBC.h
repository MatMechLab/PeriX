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
//+++ Function: prescribed species flux converted to an equivalent
//+++           source on the outermost bulk-node layer.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "BCSystem/BCBase.h"

class SpeciesFluxBC final : public BCBase {
public:
    SpeciesFluxBC()=default;
    explicit SpeciesFluxBC(const std::vector<double> &values):m_Values(values) {}
    explicit SpeciesFluxBC(const double value):m_Values{value} {}

    [[nodiscard]] std::string getBCType() const override {
        return "speciesflux";
    }
    [[nodiscard]] bool requiresLinearSystem() const override { return true; }

    void setValues(const std::vector<double> &values) { m_Values=values; }
    void setValue(const double value) { m_Values={value}; }
    [[nodiscard]] const std::vector<double>& getValues() const {
        return m_Values;
    }

    /** One-based target slots, normally the concentration DoF. */
    void setComponents(const std::vector<int> &components) {
        m_Components=components;
    }
    [[nodiscard]] const std::vector<int>& getComponents() const {
        return m_Components;
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

    void presetControlledRows(const PDMesh &Mesh,
                              const std::vector<int> &NodeIDs,
                              const int &DofsPerNode,
                              std::vector<char> &mask) const override;

    void apply(const PDMesh &Mesh,
               const std::vector<int> &NodeIDs,
               const int &DofsPerNode,
               const VectorXd &U,
               SparseMatrix &K,
               VectorXd &RHS) const override;

private:
    std::vector<double> m_Values{0.0};
    std::vector<int> m_Components;
};
