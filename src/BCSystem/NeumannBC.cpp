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
//+++ Function: homogeneous mirror condition implementation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/NeumannBC.h"

#include <unordered_set>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
[[nodiscard]] std::vector<int> resolveComponents(
    const std::vector<int> &requested,const int dofsPerNode) {
    if (!requested.empty()) return requested;
    std::vector<int> components(static_cast<std::size_t>(dofsPerNode));
    for (int i=0;i<dofsPerNode;++i) {
        components[static_cast<std::size_t>(i)]=i+1;
    }
    return components;
}

void validateComponents(const std::vector<int> &components,
                        const int dofsPerNode,const std::string &where) {
    if (dofsPerNode<1) {
        MessagePrinter::printErrorTxt(where+": DofsPerNode must be positive");
        MessagePrinter::exitPeriX();
    }
    std::unordered_set<int> unique;
    for (const int component : components) {
        if (component<1 || component>dofsPerNode) {
            MessagePrinter::printErrorTxt(
                where+": component "+std::to_string(component)+" is outside [1,"
                +std::to_string(dofsPerNode)+"]");
            MessagePrinter::exitPeriX();
        }
        if (!unique.insert(component).second) {
            MessagePrinter::printErrorTxt(
                where+": duplicate component "+std::to_string(component));
            MessagePrinter::exitPeriX();
        }
    }
}

[[nodiscard]] const std::vector<int>& mirrorMap(
    const PDMesh &Mesh,const char *where) {
    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    if (static_cast<int>(mirror.size())<Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            std::string(where)+": the homogeneous mirror condition needs "
            "GhostMirrorBulkID for the boundary ghost ring");
        MessagePrinter::exitPeriX();
    }
    return mirror;
}

[[nodiscard]] int checkedMirror(const PDMesh &Mesh,
                                const std::vector<int> &mirror,
                                const int ghostID,
                                const char *where) {
    if (ghostID<1 || ghostID>Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            std::string(where)+": node "+std::to_string(ghostID)
            +" is outside the PD mesh");
        MessagePrinter::exitPeriX();
    }
    const int bulkID=mirror[static_cast<std::size_t>(ghostID-1)];
    if (bulkID<1 || bulkID>Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            std::string(where)+": node "+std::to_string(ghostID)
            +" is not a boundary ghost with a mirror bulk");
        MessagePrinter::exitPeriX();
    }
    return bulkID;
}
}

void NeumannBC::presetSolution(const PDMesh &Mesh,
                               const std::vector<int> &NodeIDs,
                               const int &DofsPerNode,
                               VectorXd &U) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateComponents(components,DofsPerNode,"NeumannBC::presetSolution");
    const auto &mirror=mirrorMap(Mesh,"NeumannBC::presetSolution");

    for (const int ghostID : NodeIDs) {
        const int bulkID=checkedMirror(
            Mesh,mirror,ghostID,"NeumannBC::presetSolution");
        const int ghostBase=(ghostID-1)*DofsPerNode;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (const int component : components) {
            U(ghostBase+component)=U(bulkBase+component);
        }
    }
}

void NeumannBC::presetControlledRows(const PDMesh &Mesh,
                                     const std::vector<int> &NodeIDs,
                                     const int &DofsPerNode,
                                     std::vector<char> &mask) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateComponents(
        components,DofsPerNode,"NeumannBC::presetControlledRows");
    const auto &mirror=mirrorMap(Mesh,"NeumannBC::presetControlledRows");

    for (const int ghostID : NodeIDs) {
        (void)checkedMirror(
            Mesh,mirror,ghostID,"NeumannBC::presetControlledRows");
        const std::size_t base=static_cast<std::size_t>(ghostID-1)
                             *static_cast<std::size_t>(DofsPerNode);
        for (const int component : components) {
            mask[base+static_cast<std::size_t>(component-1)]=1;
        }
    }
}

void NeumannBC::apply(const PDMesh &Mesh,
                      const std::vector<int> &NodeIDs,
                      const int &DofsPerNode,
                      const VectorXd &U,
                      SparseMatrix &K,
                      VectorXd &RHS) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateComponents(components,DofsPerNode,"NeumannBC::apply");
    const auto &mirror=mirrorMap(Mesh,"NeumannBC::apply");

    for (const int ghostID : NodeIDs) {
        const int bulkID=checkedMirror(Mesh,mirror,ghostID,"NeumannBC::apply");
        const int ghostBase=(ghostID-1)*DofsPerNode;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (const int component : components) {
            const int ghostDof=ghostBase+component;
            const int bulkDof=bulkBase+component;
            const double scale=bcRowScale(K,ghostDof);
            K.zeroRowAndSetDiagonal(ghostDof,scale);
            K.insertValue(ghostDof,bulkDof,-scale);
            RHS.insertValue(
                ghostDof,scale*(U(bulkDof)-U(ghostDof)));
        }
    }
}
