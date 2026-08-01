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

#include <array>

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
     * Optional loading velocity per pinned DoF. When non-empty the
     * prescribed value ramps in time: g(t) = value + velocity * t.
     * Used for constant-velocity loading (e.g. the Kalthoff--Winkler
     * impactor). The count must be 1 (broadcast) or equal to the number
     * of pinned DoFs, exactly like the value list. Empty (default) means
     * a time-independent value.
     */
    void setVelocity(const std::vector<double> &velocity) {
        m_Velocity=velocity;
    }
    [[nodiscard]] const std::vector<double>& getVelocity() const {
        return m_Velocity;
    }

    /**
     * Optional axis-aligned coordinate box that restricts this BC to the
     * subset of the bound physical group whose PD-node coordinates lie
     * inside it. Each axis is an independent [min,max] window; an inactive
     * axis (min>max) is unconstrained. This is how a load is applied to a
     * portion of a boundary edge or a pin to the middle of a body.
     * Inactive by default (whole group).
     */
    void setBox(const std::array<double,6> &box) {
        m_Box=box;
        m_HasBox=true;
    }
    [[nodiscard]] bool hasBox() const { return m_HasBox; }

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

    void presetSolutionAtTime(const PDMesh &Mesh,
                              const std::vector<int> &NodeIDs,
                              const int &DofsPerNode,
                              VectorXd &U,
                              const double &time) const override;

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

    void applyAtTime(const PDMesh &Mesh,
                     const std::vector<int> &NodeIDs,
                     const int &DofsPerNode,
                     const VectorXd &U,
                     SparseMatrix &K,
                     VectorXd &RHS,
                     const double &time) const override;

private:
    /** True iff PD node NodeID (one-based) lies inside the active box. */
    [[nodiscard]] bool insideBox(const PDMesh &Mesh,const int NodeID) const;

    /** One-time diagnostic for a box selection that matches no group node. */
    void warnEmptySelection(const int matched,
                            const std::vector<int> &NodeIDs) const;

    std::vector<double> m_Values{0.0};
    std::vector<double> m_Velocity;
    std::vector<int> m_Components;
    std::array<double,6> m_Box{1.0,-1.0,1.0,-1.0,1.0,-1.0};
    bool m_HasBox=false;
    bool m_Direct=false;
    mutable bool m_SelectionWarned=false;
};
