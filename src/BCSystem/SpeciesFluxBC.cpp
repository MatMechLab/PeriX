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
//+++ Function: source-form species flux implementation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/SpeciesFluxBC.h"

#include <cmath>
#include <unordered_set>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
[[nodiscard]] std::vector<int> resolveComponents(
    const std::vector<int> &requested) {
    return requested.empty() ? std::vector<int>{1} : requested;
}

void validateDefinition(const std::vector<int> &components,
                        const std::vector<double> &values,
                        const int dofsPerNode) {
    if (values.empty()) {
        MessagePrinter::printErrorTxt(
            "SpeciesFluxBC::apply: at least one flux value is required");
        MessagePrinter::exitPeriX();
    }
    std::unordered_set<int> unique;
    for (const int component : components) {
        if (component<1 || component>dofsPerNode) {
            MessagePrinter::printErrorTxt(
                "SpeciesFluxBC::apply: component "+std::to_string(component)
                +" is outside [1,"+std::to_string(dofsPerNode)+"]");
            MessagePrinter::exitPeriX();
        }
        if (!unique.insert(component).second) {
            MessagePrinter::printErrorTxt(
                "SpeciesFluxBC::apply: duplicate component "
                +std::to_string(component));
            MessagePrinter::exitPeriX();
        }
    }
    if (values.size()!=1 && values.size()!=components.size()) {
        MessagePrinter::printErrorTxt(
            "SpeciesFluxBC::apply: flux value count must be one (broadcast) "
            "or match the number of target components");
        MessagePrinter::exitPeriX();
    }
}

[[nodiscard]] double selectedFlux(const std::vector<double> &values,
                                  const std::size_t index) {
    return values.size()==1 ? values.front() : values[index];
}
}

void SpeciesFluxBC::presetControlledRows(const PDMesh &Mesh,
                                         const std::vector<int> &NodeIDs,
                                         const int &DofsPerNode,
                                         std::vector<char> &mask) const {
    const auto components=resolveComponents(m_Components);
    validateDefinition(components,m_Values,DofsPerNode);

    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    if (static_cast<int>(mirror.size())<Mesh.getNodesNum()) return;

    for (const int ghostID : NodeIDs) {
        if (ghostID<1 || ghostID>Mesh.getNodesNum()) continue;
        if (mirror[static_cast<std::size_t>(ghostID-1)]<1) continue;
        const std::size_t base=static_cast<std::size_t>(ghostID-1)
                             *static_cast<std::size_t>(DofsPerNode);
        for (const int component : components) {
            mask[base+static_cast<std::size_t>(component-1)]=1;
        }
    }
}

void SpeciesFluxBC::apply(const PDMesh &Mesh,
                          const std::vector<int> &NodeIDs,
                          const int &DofsPerNode,
                          const VectorXd &U,
                          SparseMatrix &K,
                          VectorXd &RHS) const {
    const auto components=resolveComponents(m_Components);
    validateDefinition(components,m_Values,DofsPerNode);

    const auto &data=Mesh.getDataConstRef();
    const auto &mirror=data.GhostMirrorBulkID;
    if (static_cast<int>(mirror.size())<Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            "SpeciesFluxBC::apply: the source-form flux needs a ghost-ringed "
            "boundary with GhostMirrorBulkID");
        MessagePrinter::exitPeriX();
    }

    for (const int ghostID : NodeIDs) {
        if (ghostID<1 || ghostID>Mesh.getNodesNum()) {
            MessagePrinter::printErrorTxt(
                "SpeciesFluxBC::apply: node "+std::to_string(ghostID)
                +" is outside the PD mesh");
            MessagePrinter::exitPeriX();
        }
        const int bulkID=mirror[static_cast<std::size_t>(ghostID-1)];
        if (bulkID<1 || bulkID>Mesh.getNodesNum()) {
            MessagePrinter::printErrorTxt(
                "SpeciesFluxBC::apply: node "+std::to_string(ghostID)
                +" is not a boundary ghost with a mirror bulk");
            MessagePrinter::exitPeriX();
        }

        const std::size_t ghostIndex=static_cast<std::size_t>(ghostID-1);
        const bool innermost=data.GhostLayerIndex.empty()
            || ghostIndex>=data.GhostLayerIndex.size()
            || data.GhostLayerIndex[ghostIndex]<=1;

        double thickness=0.0;
        if (static_cast<int>(data.GhostFluxThickness.size())
            >=Mesh.getNodesNum()) {
            thickness=data.GhostFluxThickness[
                static_cast<std::size_t>(ghostID-1)];
        }
        if (thickness<=0.0) {
            const double dx=Mesh.getIthNodeJthCoord(ghostID,1)
                           -Mesh.getIthNodeJthCoord(bulkID,1);
            const double dy=Mesh.getIthNodeJthCoord(ghostID,2)
                           -Mesh.getIthNodeJthCoord(bulkID,2);
            const double dz=Mesh.getIthNodeJthCoord(ghostID,3)
                           -Mesh.getIthNodeJthCoord(bulkID,3);
            thickness=std::sqrt(dx*dx+dy*dy+dz*dz);
        }
        if (!(thickness>0.0) || !std::isfinite(thickness)) {
            MessagePrinter::printErrorTxt(
                "SpeciesFluxBC::apply: boundary ghost "
                +std::to_string(ghostID)
                +" has no positive conservative source thickness");
            MessagePrinter::exitPeriX();
        }

        const int ghostBase=(ghostID-1)*DofsPerNode;
        const int bulkBase=(bulkID-1)*DofsPerNode;
        for (std::size_t k=0;k<components.size();++k) {
            const int component=components[k];
            const int ghostDof=ghostBase+component;
            const int bulkDof=bulkBase+component;

            // R_c=(c-c_old)/dt-div(M grad(mu))-j/thickness. PeriX stores
            // RHS=R and K=-dR/dU, so a positive inward j contributes
            // -j/thickness and therefore increases the converged mass.
            if (innermost) {
                RHS.addValue(
                    bulkDof,-selectedFlux(m_Values,k)/thickness);
            }

            // The source supplies the species balance; the matching ghost
            // value is mirrored to impose zero normal composition gradient.
            const double scale=bcRowScale(K,ghostDof);
            K.zeroRowAndSetDiagonal(ghostDof,scale);
            K.insertValue(ghostDof,bulkDof,-scale);
            RHS.insertValue(
                ghostDof,scale*(U(bulkDof)-U(ghostDof)));
        }
    }
}
