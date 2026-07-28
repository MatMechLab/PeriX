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
//+++ Date    : 2026.05.08
//+++ Function: ElmtSystem owns the registry of element kernels
//+++           ("models") for the current simulation. PDSystem
//+++           queries it to obtain per-bond and per-node residual
//+++           / jacobian contributions during global assembly.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/LocalElmtInfo.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDOperators/PDOperators.h"

class PDMesh;

class ElmtSystem {
public:
    ElmtSystem()=default;
    ~ElmtSystem()=default;

    ElmtSystem(const ElmtSystem&)=delete;
    ElmtSystem& operator=(const ElmtSystem&)=delete;
    ElmtSystem(ElmtSystem&&) noexcept=default;
    ElmtSystem& operator=(ElmtSystem&&) noexcept=default;

    void addElement(const std::string &name,std::unique_ptr<ElementBase> elmt);

    [[nodiscard]] bool hasElement(const std::string &name) const;
    [[nodiscard]] const ElementBase& getElement(const std::string &name) const;
    [[nodiscard]] const ElementBase& getElementByIndex(const int &i) const;
    [[nodiscard]] const std::string& getElementNameByIndex(const int &i) const;

    [[nodiscard]] int getElementsNum() const { return static_cast<int>(m_Elements.size()); }
    [[nodiscard]] int getMaxDofsPerNode() const;

    /**
     * per-bond dispatch: accumulate the residual / jacobian
     * contributions of every registered element. Output buffers
     * are zeroed inside.
     */
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
                                        MatrixXd &LocalK_IJ) const;

    /**
     * per-node dispatch: accumulate per-node residual / jacobian
     * contributions of every registered element.
     */
    void computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                         const int &NodeI,
                                         const VectorXd &U_I,
                                         const VectorXd &Uold_I,
                                         const double  &Volume,
                                         VectorXd &LocalR_node,
                                         MatrixXd &LocalK_node) const;

    /**
     * true iff there is at least one registered element and every one of
     * them is explicit (forward-Euler, Jacobian-free). Mixing explicit and
     * implicit kernels in one run is rejected by the driver.
     */
    [[nodiscard]] bool isExplicit() const;

    /**
     * common time-derivative order of the registered explicit kernels
     * (see ElementBase::getTimeOrder). All explicit elements in one run
     * must agree; the driver uses it to pick the forward-Euler (1) vs
     * central-difference (2) integrator. Returns 1 for an empty registry.
     */
    [[nodiscard]] int getTimeOrder() const;

    /**
     * per-DoF time-derivative order mask (length getMaxDofsPerNode()), taken
     * from the registered kernel (see ElementBase::getDofTimeOrders). Used by
     * the driver to pick the velocity-Verlet integrator for a mixed-order
     * (coupled) explicit kernel and to advance each DoF with its own rule.
     */
    [[nodiscard]] std::vector<int> getDofTimeOrders() const;

    /**
     * true iff the per-DoF order mask mixes first- and second-order DoFs, i.e.
     * the kernel needs the velocity-Verlet driver. False for a uniform-order
     * kernel (forward Euler or central difference) and an empty registry.
     */
    [[nodiscard]] bool hasMixedTimeOrders() const;

    /**
     * most restrictive positive stable-dt estimate over the registered kernels
     * (CFL-type bound for the fixed-dt explicit integrators); <=0 = none.
     */
    [[nodiscard]] double estimateStableDt(const PDMesh &Mesh) const;

    /**
     * first registered kernel advertising the named projection (nullptr if
     * none); the output paths call the provider, not blindly element[0].
     */
    [[nodiscard]] const ElementBase* findProjectionProvider(const std::string &name) const;

    /**
     * true iff every registered (explicit) kernel is quasi-static, i.e. wants
     * the ADR (Adaptive Dynamic Relaxation) driver instead of a time-marching
     * one. See ElementBase::isQuasiStatic. Returns false for an empty registry.
     */
    [[nodiscard]] bool isQuasiStatic() const;

    /**
     * per-bond / per-node dispatch of the EXPLICIT rate (du/dt = R): the
     * matrix-free counterparts used by the forward-Euler integrator. They
     * accumulate the rate contributions of every registered element; output
     * buffers are zeroed inside.
     */
    void computeBondResidual(const PDOperators &t_PDOperators,
                             const LocalElmtInfo &Info,
                             const int &NodeI,
                             const int &NodeJ,
                             const VectorXd &U_I,
                             const VectorXd &U_J,
                             const VectorXd &Uold_I,
                             const VectorXd &Uold_J,
                             const double  &Volume,
                             VectorXd &LocalR) const;

    void computeNodalResidual(const LocalElmtInfo &Info,
                              const int &NodeI,
                              const VectorXd &U_I,
                              const VectorXd &Uold_I,
                              const double  &Volume,
                              VectorXd &LocalR_node) const;

    /**
     * dispatch preprocessIteration() to every registered element.
     * Called by PDSystem at the start of each Newton iteration so
     * coupled kernels can refresh their frozen gradient / coefficient
     * caches before assembly.
     */
    void preprocessIteration(const PDMesh &Mesh,
                             PDOperators &ops,
                             const LocalElmtInfo &Info,
                             const int &DofsPerNode,
                             const VectorXd &U,
                             const VectorXd &Uold) const;

    void printElmtSystemInfo() const;

private:
    struct ElementEntry {
        std::string Name;
        std::unique_ptr<ElementBase> Element;
    };
    std::vector<ElementEntry> m_Elements;
};
