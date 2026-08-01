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
//+++ Function: exact Dirichlet row replacement. The prescribed value
//+++           is imposed at the wall by reflection across it,
//+++           u_ghost = 2*g(t) - u_bulk; a node with no mirror bulk
//+++           (or direct=true) is pinned directly, u = g(t). The
//+++           value optionally ramps in time, g(t) = value +
//+++           velocity*t, and an optional axis-aligned box restricts
//+++           the condition to a subset of the bound group.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/DirichletBC.h"

#include <cstdio>
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

void checkNodeRange(const PDMesh &Mesh,const int nodeID,
                    const std::string &where) {
    if (nodeID<1 || nodeID>Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            where+": node "+std::to_string(nodeID)+" is outside the PD mesh");
        MessagePrinter::exitPeriX();
    }
}
}

bool DirichletBC::insideBox(const PDMesh &Mesh,const int NodeID) const {
    for (int axis=0;axis<3;++axis) {
        const double lo=m_Box[static_cast<std::size_t>(2*axis)];
        const double hi=m_Box[static_cast<std::size_t>(2*axis+1)];
        if (lo>hi) continue;
        const double x=Mesh.getIthNodeJthCoord(NodeID,axis+1);
        if (x<lo || x>hi) return false;
    }
    return true;
}

void DirichletBC::presetSolution(const PDMesh &Mesh,
                                 const std::vector<int> &NodeIDs,
                                 const int &DofsPerNode,
                                 VectorXd &U) const {
    presetSolutionAtTime(Mesh,NodeIDs,DofsPerNode,U,0.0);
}

void DirichletBC::presetSolutionAtTime(const PDMesh &Mesh,
                                       const std::vector<int> &NodeIDs,
                                       const int &DofsPerNode,
                                       VectorXd &U,
                                       const double &time) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,
                       "DirichletBC::presetSolution");
    if (!m_Velocity.empty()) {
        validateDefinition(components,m_Velocity,DofsPerNode,
                           "DirichletBC::presetSolution(velocity)");
    }

    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    const bool haveMirror=
        static_cast<int>(mirror.size())>=Mesh.getNodesNum();
    int matched=0;
    for (const int nodeID : NodeIDs) {
        checkNodeRange(Mesh,nodeID,"DirichletBC::presetSolution");
        if (m_HasBox && !insideBox(Mesh,nodeID)) continue;
        ++matched;
        const int ghostBase=(nodeID-1)*DofsPerNode;
        const int bulkID=haveMirror
            ? mirror[static_cast<std::size_t>(nodeID-1)]
            : 0;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (std::size_t k=0;k<components.size();++k) {
            const int component=components[k];
            // Ramped value g(t)=value+velocity*t (velocity 0 if not set).
            const double value=selectedValue(m_Values,k)
                +(m_Velocity.empty() ? 0.0 : selectedValue(m_Velocity,k))*time;
            // Reflection: the midpoint of ghost and bulk lies on the wall, so
            // the ghost is set so the wall value is exactly g(t). A direct
            // pin (or a node with no mirror bulk) writes the value itself.
            U(ghostBase+component)=(bulkID>=1 && !m_Direct)
                ? 2.0*value-U(bulkBase+component)
                : value;
        }
    }
    warnEmptySelection(matched,NodeIDs);
}

void DirichletBC::presetControlledRows(const PDMesh &Mesh,
                                       const std::vector<int> &NodeIDs,
                                       const int &DofsPerNode,
                                       std::vector<char> &mask) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,
                       "DirichletBC::presetControlledRows");
    for (const int nodeID : NodeIDs) {
        checkNodeRange(Mesh,nodeID,"DirichletBC::presetControlledRows");
        if (m_HasBox && !insideBox(Mesh,nodeID)) continue;
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
    applyAtTime(Mesh,NodeIDs,DofsPerNode,U,K,RHS,0.0);
}

void DirichletBC::applyAtTime(const PDMesh &Mesh,
                              const std::vector<int> &NodeIDs,
                              const int &DofsPerNode,
                              const VectorXd &U,
                              SparseMatrix &K,
                              VectorXd &RHS,
                              const double &time) const {
    const auto components=resolveComponents(m_Components,DofsPerNode);
    validateDefinition(components,m_Values,DofsPerNode,"DirichletBC::apply");
    if (!m_Velocity.empty()) {
        validateDefinition(components,m_Velocity,DofsPerNode,
                           "DirichletBC::apply(velocity)");
    }

    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    const bool haveMirror=
        static_cast<int>(mirror.size())>=Mesh.getNodesNum();
    int matched=0;
    for (const int nodeID : NodeIDs) {
        checkNodeRange(Mesh,nodeID,"DirichletBC::apply");
        // Same node selection as presetSolutionAtTime: the box (if any)
        // restricts which nodes of the group this BC acts on; the constraint
        // must agree with the preset or the Newton increment fights it.
        if (m_HasBox && !insideBox(Mesh,nodeID)) continue;
        ++matched;
        const int ghostBase=(nodeID-1)*DofsPerNode;
        const int bulkID=haveMirror
            ? mirror[static_cast<std::size_t>(nodeID-1)]
            : 0;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (std::size_t k=0;k<components.size();++k) {
            const int component=components[k];
            const int ghostDof=ghostBase+component;
            // Ramped value g(t)=value+velocity*t, identical to the preset.
            const double value=selectedValue(m_Values,k)
                +(m_Velocity.empty() ? 0.0 : selectedValue(m_Velocity,k))*time;
            const double scale=bcRowScale(K,ghostDof);

            K.zeroRowAndSetDiagonal(ghostDof,scale);
            if (bulkID>=1 && !m_Direct) {
                // Reflection constraint U(g)+U(b)=2*g(t) (wall value = g(t)):
                //   row g -> s*(dU(g)+dU(b)) = s*(2*g(t) - U(g) - U(b)).
                const int bulkDof=bulkBase+component;
                K.insertValue(ghostDof,bulkDof,scale);
                RHS.insertValue(
                    ghostDof,scale*(2.0*value-U(ghostDof)-U(bulkDof)));
            }
            else {
                // Direct pin (or a node with no mirror bulk): U -> g(t).
                RHS.insertValue(ghostDof,scale*(value-U(ghostDof)));
            }
        }
    }
    warnEmptySelection(matched,NodeIDs);
}

void DirichletBC::warnEmptySelection(const int matched,
                                     const std::vector<int> &NodeIDs) const {
    // A box restriction that selects ZERO group nodes pins nothing -- the
    // constraint is silently absent. For a rigid-body pin this leaves the
    // matrix singular. A centroid-based PD mesh has no node at an exact
    // location, so a too-small box catches nothing; warn ONCE.
    if (m_SelectionWarned || matched>0 || !m_HasBox || NodeIDs.empty()) {
        return;
    }
    m_SelectionWarned=true;
    char buffer[320];
    std::snprintf(buffer,sizeof(buffer),
        "dirichlet box [%.4g,%.4g]x[%.4g,%.4g]x[%.4g,%.4g] selected 0 of %d group nodes "
        "-- nothing constrained (a too-small box catches no cell centroid on a PD mesh). "
        "Enlarge the box so it captures at least one node.",
        m_Box[0],m_Box[1],m_Box[2],m_Box[3],m_Box[4],m_Box[5],
        static_cast<int>(NodeIDs.size()));
    MessagePrinter::printWarningTxt(buffer);
}
