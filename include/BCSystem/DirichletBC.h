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
//+++ Function: exact Dirichlet row replacement with the default
//+++           wall-midpoint reflection or optional direct ghost
//+++           assignment.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "BCSystem/BCBase.h"

class DirichletBC final : public BCBase {
public:
    DirichletBC()=default;
    explicit DirichletBC(const std::vector<double> &values):m_Values(values) {}
    explicit DirichletBC(const double value):m_Values{value} {}

    [[nodiscard]] std::string getBCType() const override {
        return "dirichlet";
    }

    void setValues(const std::vector<double> &values) { m_Values=values; }
    void setValue(const double value) { m_Values={value}; }
    [[nodiscard]] const std::vector<double>& getValues() const {
        return m_Values;
    }

    /** One-based DoF slots; an empty list constrains every node DoF. */
    void setComponents(const std::vector<int> &components) {
        m_Components=components;
    }
    [[nodiscard]] const std::vector<int>& getComponents() const {
        return m_Components;
    }

    /**
     * false (default): u_ghost + u_bulk = 2*g at the wall midpoint.
     * true:            u_ghost = g.
     */
    void setDirect(const bool direct) { m_Direct=direct; }
    [[nodiscard]] bool getDirect() const { return m_Direct; }

    void presetSolution(const PDMesh &Mesh,
                        const std::vector<int> &NodeIDs,
                        const int &DofsPerNode,
                        VectorXd &U) const override;

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
    bool m_Direct=false;
};
