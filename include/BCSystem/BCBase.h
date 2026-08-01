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
//+++ Function: public boundary-condition kernel interface.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"

class PDMesh;
class PDOperators;

/**
 * Return a representative scale for a replacement row. Using the assembled
 * diagonal, or the largest row entry when the diagonal vanishes, keeps an
 * exact boundary equation on the same numerical scale as the surrounding
 * equations without introducing a penalty parameter.
 */
[[nodiscard]] inline double bcRowScale(const SparseMatrix &K,const int row) {
    constexpr double zeroFloor=1.0e-100;
    double scale=K.getDiagonal(row);
    if (std::fabs(scale)<zeroFloor) scale=K.getRowMaxAbs(row);
    if (std::fabs(scale)<zeroFloor) scale=1.0;
    return scale;
}

class BCBase {
public:
    BCBase()=default;
    virtual ~BCBase()=default;

    [[nodiscard]] virtual std::string getBCType() const=0;

    /**
     * True when the condition acts only through the assembled linear system.
     * Matrix-free explicit and ADR drivers reject such conditions rather than
     * silently ignoring them.
     */
    [[nodiscard]] virtual bool requiresLinearSystem() const { return false; }

    /**
     * Write constrained or mirrored values into the global solution vector.
     * Node and degree-of-freedom indices are one-based throughout PeriX.
     */
    virtual void presetSolution(const PDMesh &Mesh,
                                const std::vector<int> &NodeIDs,
                                const int &DofsPerNode,
                                VectorXd &U) const=0;

    /**
     * Time-aware preset used by the transient drivers. A time-dependent
     * condition (the velocity-ramped Dirichlet hold) overrides this; the
     * default forwards to the time-independent preset.
     */
    virtual void presetSolutionAtTime(const PDMesh &Mesh,
                                      const std::vector<int> &NodeIDs,
                                      const int &DofsPerNode,
                                      VectorXd &U,
                                      const double &time) const {
        (void)time;
        presetSolution(Mesh,NodeIDs,DofsPerNode,U);
    }

    /**
     * Mark rows written by presetSolution. The ADR driver excludes these rows
     * from its free-degree-of-freedom damping estimate.
     */
    virtual void presetControlledRows(const PDMesh &Mesh,
                                      const std::vector<int> &NodeIDs,
                                      const int &DofsPerNode,
                                      std::vector<char> &mask) const {
        (void)Mesh;
        (void)NodeIDs;
        (void)DofsPerNode;
        (void)mask;
    }

    /**
     * Apply an exact row constraint or an assembled source to K and RHS.
     */
    virtual void apply(const PDMesh &Mesh,
                       const std::vector<int> &NodeIDs,
                       const int &DofsPerNode,
                       const VectorXd &U,
                       SparseMatrix &K,
                       VectorXd &RHS) const=0;

    /**
     * Time-aware row constraint. The imposed value must agree with
     * presetSolutionAtTime at the same time, otherwise the Newton increment
     * fights the preset ramp. The default forwards to the time-independent
     * apply.
     */
    virtual void applyAtTime(const PDMesh &Mesh,
                             const std::vector<int> &NodeIDs,
                             const int &DofsPerNode,
                             const VectorXd &U,
                             SparseMatrix &K,
                             VectorXd &RHS,
                             const double &time) const {
        (void)time;
        apply(Mesh,NodeIDs,DofsPerNode,U,K,RHS);
    }

    /**
     * Operator-aware form used by strong PDDO natural conditions. Conditions
     * that do not need differential operators use the default forwarding
     * implementation.
     */
    virtual void applyWithOperators(const PDMesh &Mesh,
                                    PDOperators &Operators,
                                    const std::vector<int> &NodeIDs,
                                    const int &DofsPerNode,
                                    const VectorXd &U,
                                    SparseMatrix &K,
                                    VectorXd &RHS,
                                    const double &time) const {
        (void)Operators;
        applyAtTime(Mesh,NodeIDs,DofsPerNode,U,K,RHS,time);
    }
};
