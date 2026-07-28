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
//+++ Function: homogeneous mirror condition u_ghost=u_bulk.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "BCSystem/BCBase.h"

class NeumannBC final : public BCBase {
public:
    NeumannBC()=default;

    [[nodiscard]] std::string getBCType() const override {
        return "neumann";
    }

    /** One-based DoF slots; an empty list mirrors every node DoF. */
    void setComponents(const std::vector<int> &components) {
        m_Components=components;
    }
    [[nodiscard]] const std::vector<int>& getComponents() const {
        return m_Components;
    }

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
    std::vector<int> m_Components;
};
