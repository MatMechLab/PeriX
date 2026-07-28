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
//+++ Function: manuscript-scope BC registry and dispatch.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/BCSystem.h"

#include <algorithm>
#include <set>
#include <utility>

#include "BCSystem/DirichletBC.h"
#include "BCSystem/NeumannBC.h"
#include "BCSystem/PDTractionBC.h"
#include "BCSystem/SpeciesFluxBC.h"
#include "ElmtSystem/ElmtSystem.h"
#include "Utils/MessagePrinter.h"

void BCSystem::addBC(const std::string &name,
                     const std::string &phyName,
                     std::unique_ptr<BCBase> bc) {
    if (!bc) {
        MessagePrinter::printErrorTxt(
            "BCSystem::addBC: null BC for name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    if (name.empty()) {
        MessagePrinter::printErrorTxt("BCSystem::addBC: BC name is empty");
        MessagePrinter::exitPeriX();
    }
    if (hasBC(name)) {
        MessagePrinter::printErrorTxt(
            "BCSystem::addBC: duplicate BC name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    if (phyName.empty()) {
        MessagePrinter::printErrorTxt(
            "BCSystem::addBC: empty physical group for BC '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    m_BCs.push_back(BCEntry{name,phyName,std::move(bc)});
}

bool BCSystem::hasBC(const std::string &name) const {
    return std::any_of(
        m_BCs.begin(),m_BCs.end(),
        [&name](const BCEntry &entry) { return entry.Name==name; });
}

bool BCSystem::validateSpeciesDirichletOnGhosts(
    const PDMesh &Mesh,const ElmtSystem &ElmtSys) const {
    std::set<int> speciesSlots;
    for (int i=0;i<ElmtSys.getElementsNum();++i) {
        for (const int slot
             : ElmtSys.getElementByIndex(i).getGhostDropSpeciesDofSlots()) {
            speciesSlots.insert(slot);
        }
    }
    if (speciesSlots.empty()) return true;

    const auto &data=Mesh.getDataConstRef();
    if (static_cast<int>(data.GhostMirrorBulkID.size())<data.NodesNum) {
        return true;
    }

    for (const auto &entry : m_BCs) {
        const auto *dirichlet=
            dynamic_cast<const DirichletBC*>(entry.BC.get());
        if (!dirichlet) continue;

        const auto &components=dirichlet->getComponents();
        bool targetsSpecies=components.empty();
        for (const int component : components) {
            if (speciesSlots.contains(component-1)) {
                targetsSpecies=true;
                break;
            }
        }
        if (!targetsSpecies) continue;

        bool targetsGhost=false;
        for (const int nodeID : Mesh.getNodeIDsViaPhyName(entry.PhyName)) {
            if (nodeID>=1 && nodeID<=data.NodesNum
                && data.GhostMirrorBulkID[
                    static_cast<std::size_t>(nodeID-1)]>0) {
                targetsGhost=true;
                break;
            }
        }
        if (!targetsGhost) continue;

        MessagePrinter::printErrorTxt(
            "BCSystem: dirichlet BC '"+entry.Name
            +"' targets a Cahn-Hilliard species DoF on boundary ghosts. "
            "The conservative species operator excludes ghost bonds, so this "
            "value cannot drive boundary transport. Use the manuscript's "
            "source-form 'speciesflux' BC for a prescribed species influx.");
        return false;
    }
    return true;
}

void BCSystem::presetSolution(const PDMesh &Mesh,
                              const int &DofsPerNode,
                              VectorXd &U,
                              const double &time) const {
    (void)time;
    for (const auto &entry : m_BCs) {
        const auto &nodeIDs=Mesh.getNodeIDsViaPhyName(entry.PhyName);
        if (nodeIDs.empty()
            && m_WarnedEmptyGroups.insert(entry.Name).second) {
            MessagePrinter::printWarningTxt(
                "BCSystem: BC '"+entry.Name
                +"' references empty physical group '"+entry.PhyName
                +"' and is a no-op");
        }
        entry.BC->presetSolution(Mesh,nodeIDs,DofsPerNode,U);
    }
}

std::vector<char> BCSystem::collectPresetDofMask(
    const PDMesh &Mesh,const int &DofsPerNode) const {
    std::vector<char> mask(
        static_cast<std::size_t>(Mesh.getNodesNum())
        *static_cast<std::size_t>(DofsPerNode),0);
    for (const auto &entry : m_BCs) {
        const auto &nodeIDs=Mesh.getNodeIDsViaPhyName(entry.PhyName);
        entry.BC->presetControlledRows(
            Mesh,nodeIDs,DofsPerNode,mask);
    }
    return mask;
}

void BCSystem::applyBCs(const PDMesh &Mesh,
                        PDOperators &Operators,
                        const int &DofsPerNode,
                        const VectorXd &U,
                        const VectorXd &Uold,
                        const LocalElmtInfo &Info,
                        SparseMatrix &K,
                        VectorXd &RHS) const {
    (void)Uold;
    (void)Info;
    for (const auto &entry : m_BCs) {
        const auto &nodeIDs=Mesh.getNodeIDsViaPhyName(entry.PhyName);
        if (nodeIDs.empty()
            && m_WarnedEmptyGroups.insert(entry.Name).second) {
            MessagePrinter::printWarningTxt(
                "BCSystem: BC '"+entry.Name
                +"' references empty physical group '"+entry.PhyName
                +"' and is a no-op");
        }
        entry.BC->applyWithOperators(
            Mesh,Operators,nodeIDs,DofsPerNode,U,K,RHS);
    }
}

std::vector<std::string> BCSystem::getLinearSystemOnlyBCs() const {
    std::vector<std::string> names;
    for (const auto &entry : m_BCs) {
        if (entry.BC->requiresLinearSystem()) {
            names.push_back(
                entry.Name+" ("+entry.BC->getBCType()+")");
        }
    }
    return names;
}

void BCSystem::printBCSystemInfo() const {
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("BC system information summary:");
    MessagePrinter::printNormalTxt(
        "  registered BCs="+std::to_string(getBCsNum()));

    for (const auto &entry : m_BCs) {
        std::string line="  - name='"+entry.Name
                        +"', type='"+entry.BC->getBCType()
                        +"', phygroup='"+entry.PhyName+"'";
        const std::vector<int> *components=nullptr;
        if (const auto *bc=
            dynamic_cast<const DirichletBC*>(entry.BC.get())) {
            components=&bc->getComponents();
        }
        else if (const auto *bc=
                 dynamic_cast<const NeumannBC*>(entry.BC.get())) {
            components=&bc->getComponents();
        }
        else if (const auto *bc=
                 dynamic_cast<const SpeciesFluxBC*>(entry.BC.get())) {
            components=&bc->getComponents();
        }
        else if (const auto *bc=
                 dynamic_cast<const PDTractionBC*>(entry.BC.get())) {
            components=&bc->getDisplacementComponents();
            if (bc->hasChemicalExpansion()) {
                line+=", chemical_expansion=true";
            }
        }
        if (components && !components->empty()) {
            line+=", components=[";
            for (std::size_t i=0;i<components->size();++i) {
                if (i>0) line+=",";
                line+=std::to_string((*components)[i]);
            }
            line+="]";
        }
        MessagePrinter::printNormalTxt(line);
    }
    MessagePrinter::printStars();
}
