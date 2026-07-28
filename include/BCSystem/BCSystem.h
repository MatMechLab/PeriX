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
//+++ Function: registry and dispatcher for the public boundary
//+++           conditions described in the PeriX manuscript.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "BCSystem/BCBase.h"
#include "ElmtSystem/LocalElmtInfo.h"
#include "PDMesh/PDMesh.h"

class ElmtSystem;

class BCSystem {
public:
    BCSystem()=default;
    ~BCSystem()=default;

    BCSystem(const BCSystem&)=delete;
    BCSystem& operator=(const BCSystem&)=delete;
    BCSystem(BCSystem&&) noexcept=default;
    BCSystem& operator=(BCSystem&&) noexcept=default;

    void addBC(const std::string &name,
               const std::string &phyName,
               std::unique_ptr<BCBase> bc);

    [[nodiscard]] bool hasBC(const std::string &name) const;
    [[nodiscard]] int getBCsNum() const {
        return static_cast<int>(m_BCs.size());
    }

    /**
     * Reject a Dirichlet condition on a Cahn--Hilliard species DoF when
     * applied to ghost nodes. The conservative species discretization omits
     * ghost bonds, so such a value would be inert; the manuscript's source
     * form SpeciesFluxBC must be used for an imposed species influx.
     */
    [[nodiscard]] bool validateSpeciesDirichletOnGhosts(
        const PDMesh &Mesh,const ElmtSystem &ElmtSys) const;

    /**
     * Apply every preset-based boundary condition. The time argument remains
     * in the driver-facing interface for a uniform transient-driver API; the
     * public BC kernels are time independent.
     */
    void presetSolution(const PDMesh &Mesh,
                        const int &DofsPerNode,
                        VectorXd &U,
                        const double &time=0.0) const;

    [[nodiscard]] std::vector<char> collectPresetDofMask(
        const PDMesh &Mesh,const int &DofsPerNode) const;

    /**
     * Apply all assembled boundary equations and sources. Uold and Info are
     * accepted because they are part of the nonlinear-solver interface; the
     * manuscript-scope BC kernels do not require history variables.
     */
    void applyBCs(const PDMesh &Mesh,
                  PDOperators &Operators,
                  const int &DofsPerNode,
                  const VectorXd &U,
                  const VectorXd &Uold,
                  const LocalElmtInfo &Info,
                  SparseMatrix &K,
                  VectorXd &RHS) const;

    [[nodiscard]] std::vector<std::string> getLinearSystemOnlyBCs() const;

    void printBCSystemInfo() const;

private:
    struct BCEntry {
        std::string Name;
        std::string PhyName;
        std::unique_ptr<BCBase> BC;
    };

    std::vector<BCEntry> m_BCs;
    mutable std::unordered_set<std::string> m_WarnedEmptyGroups;
};
