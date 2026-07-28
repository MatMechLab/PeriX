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
//+++ Function: exact Dirichlet row replacement.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/DirichletBC.h"

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

void validateDefinition(const std::vector<int> &components,
                        const std::vector<double> &values,
                        const int dofsPerNode,
                        const std::string &where) {
    if (dofsPerNode<1) {
        MessagePrinter::printErrorTxt(where+": DofsPerNode must be positive");
        MessagePrinter::exitPeriX();
    }
    if (values.empty()) {
        MessagePrinter::printErrorTxt(where+": at least one prescribed value is required");
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
    if (values.size()!=1 && values.size()!=components.size()) {
        MessagePrinter::printErrorTxt(
            where+": value count must be one (broadcast) or match the number "
            "of constrained components");
        MessagePrinter::exitPeriX();
    }
}

[[nodiscard]] double selectedValue(const std::vector<double> &values,
                                   const std::size_t index) {
    return values.size()==1 ? values.front() : values[index];
}

[[nodiscard]] int mirrorBulk(const PDMesh &Mesh,const int nodeID,
                             const bool direct,const std::string &where) {
    if (nodeID<1 || nodeID>Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            where+": node "+std::to_string(nodeID)+" is outside the PD mesh");
        MessagePrinter::exitPeriX();
    }
    if (direct) return 0;

    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    if (static_cast<int>(mirror.size())<Mesh.getNodesNum()
        || mirror[static_cast<std::size_t>(nodeID-1)]<1) {
        MessagePrinter::printErrorTxt(
            where+": reflected Dirichlet data must act on ghost nodes with a "
            "mirror bulk; use direct=true only when the listed node itself is "
            "the prescribed boundary layer");
        MessagePrinter::exitPeriX();
    }
    return mirror[static_cast<std::size_t>(nodeID-1)];
}
}

void DirichletBC::presetSolution(const PDMesh &Mesh,
                                 const std::vector<int> &NodeIDs,
                                 const int &DofsPerNode,
                                 VectorXd &U) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,
                       "DirichletBC::presetSolution");

    for (const int nodeID : NodeIDs) {
        const int bulkID=mirrorBulk(Mesh,nodeID,m_Direct,
                                    "DirichletBC::presetSolution");
        const int ghostBase=(nodeID-1)*DofsPerNode;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (std::size_t k=0;k<components.size();++k) {
            const int component=components[k];
            const double value=selectedValue(m_Values,k);
            U(ghostBase+component)=m_Direct
                ? value
                : 2.0*value-U(bulkBase+component);
        }
    }
}

void DirichletBC::presetControlledRows(const PDMesh &Mesh,
                                       const std::vector<int> &NodeIDs,
                                       const int &DofsPerNode,
                                       std::vector<char> &mask) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,
                       "DirichletBC::presetControlledRows");
    for (const int nodeID : NodeIDs) {
        (void)mirrorBulk(Mesh,nodeID,m_Direct,
                         "DirichletBC::presetControlledRows");
        const std::size_t base=static_cast<std::size_t>(nodeID-1)
                             *static_cast<std::size_t>(DofsPerNode);
        for (const int component : components) {
            mask[base+static_cast<std::size_t>(component-1)]=1;
        }
    }
}

void DirichletBC::apply(const PDMesh &Mesh,
                        const std::vector<int> &NodeIDs,
                        const int &DofsPerNode,
                        const VectorXd &U,
                        SparseMatrix &K,
                        VectorXd &RHS) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,"DirichletBC::apply");

    for (const int nodeID : NodeIDs) {
        const int bulkID=mirrorBulk(Mesh,nodeID,m_Direct,"DirichletBC::apply");
        const int ghostBase=(nodeID-1)*DofsPerNode;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (std::size_t k=0;k<components.size();++k) {
            const int component=components[k];
            const int ghostDof=ghostBase+component;
            const double value=selectedValue(m_Values,k);
            const double scale=bcRowScale(K,ghostDof);

            K.zeroRowAndSetDiagonal(ghostDof,scale);
            if (m_Direct) {
                RHS.insertValue(ghostDof,scale*(value-U(ghostDof)));
                continue;
            }

            const int bulkDof=bulkBase+component;
            K.insertValue(ghostDof,bulkDof,scale);
            RHS.insertValue(
                ghostDof,scale*(2.0*value-U(ghostDof)-U(bulkDof)));
        }
    }
}
