//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ICSystem/ICBase.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"

/** Field initialized by an IC entry. */
enum class ICField {
    Solution,
    Velocity
};

/**
 * Registry for the manuscript-supported initial-condition profiles.
 *
 * Entries are applied in declaration order, so a later entry may deliberately
 * override an earlier entry on the same node/DoF pair. An empty physical-group
 * name selects every PD node. Solution and velocity are separate fields:
 * velocity ICs initialize the central-difference history and are not boundary
 * conditions that remain active after t=0.
 */
class ICSystem {
public:
    ICSystem()=default;
    ~ICSystem()=default;

    ICSystem(const ICSystem&)=delete;
    ICSystem& operator=(const ICSystem&)=delete;
    ICSystem(ICSystem&&) noexcept=default;
    ICSystem& operator=(ICSystem&&) noexcept=default;

    void addIC(const std::string &name,
               const std::string &phyName,
               std::unique_ptr<ICBase> ic,
               const ICField &field=ICField::Solution);

    [[nodiscard]] bool hasIC(const std::string &name) const;
    [[nodiscard]] int getICsNum() const {
        return static_cast<int>(m_ICs.size());
    }
    [[nodiscard]] bool hasVelocityIC() const;

    /** Apply solution ICs only. */
    void applyInitialConditions(const PDMesh &Mesh,
                                const int &DofsPerNode,
                                VectorXd &Solution) const;

    /** Apply both solution and initial-velocity ICs. */
    void applyInitialConditions(const PDMesh &Mesh,
                                const int &DofsPerNode,
                                VectorXd &Solution,
                                VectorXd &InitialVelocity) const;

    void printICSystemInfo() const;

private:
    struct ICEntry {
        std::string Name;
        std::string PhyName;
        ICField Field=ICField::Solution;
        std::unique_ptr<ICBase> IC;
    };

    void applyField(const PDMesh &Mesh,
                    const int &DofsPerNode,
                    const ICField &field,
                    VectorXd &Values) const;

    std::vector<ICEntry> m_ICs;
};
