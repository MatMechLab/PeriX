//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "ICSystem/ICSystem.h"

#include <algorithm>
#include <utility>

#include "Utils/MessagePrinter.h"

namespace {
[[nodiscard]] const char* fieldName(const ICField &field) {
    return field==ICField::Velocity ? "velocity" : "solution";
}
}

void ICSystem::addIC(const std::string &name,
                     const std::string &phyName,
                     std::unique_ptr<ICBase> ic,
                     const ICField &field) {
    if (!ic) {
        MessagePrinter::printErrorTxt(
            "ICSystem::addIC: null IC passed for name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    if (name.empty()) {
        MessagePrinter::printErrorTxt(
            "ICSystem::addIC: the IC name must not be empty");
        MessagePrinter::exitPeriX();
    }
    if (hasIC(name)) {
        MessagePrinter::printErrorTxt(
            "ICSystem::addIC: duplicate IC name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    m_ICs.push_back(ICEntry{name,phyName,field,std::move(ic)});
}

bool ICSystem::hasIC(const std::string &name) const {
    return std::any_of(
        m_ICs.begin(),m_ICs.end(),
        [&name](const ICEntry &entry){ return entry.Name==name; });
}

bool ICSystem::hasVelocityIC() const {
    return std::any_of(
        m_ICs.begin(),m_ICs.end(),
        [](const ICEntry &entry){ return entry.Field==ICField::Velocity; });
}

void ICSystem::applyField(const PDMesh &Mesh,
                          const int &DofsPerNode,
                          const ICField &field,
                          VectorXd &Values) const {
    std::vector<int> allNodes;
    for (const auto &entry : m_ICs) {
        if (entry.Field!=field) continue;

        if (entry.PhyName.empty()) {
            if (allNodes.empty()) {
                allNodes.reserve(static_cast<std::size_t>(Mesh.getNodesNum()));
                for (int nodeID=1;nodeID<=Mesh.getNodesNum();++nodeID) {
                    allNodes.push_back(nodeID);
                }
            }
            entry.IC->apply(Mesh,allNodes,DofsPerNode,Values);
            continue;
        }

        const auto &nodeIDs=Mesh.getNodeIDsViaPhyName(entry.PhyName);
        if (nodeIDs.empty()) {
            MessagePrinter::printWarningTxt(
                "ICSystem: IC '"+entry.Name+"' references empty physical group '"
                +entry.PhyName+"'");
        }
        entry.IC->apply(Mesh,nodeIDs,DofsPerNode,Values);
    }
}

void ICSystem::applyInitialConditions(const PDMesh &Mesh,
                                      const int &DofsPerNode,
                                      VectorXd &Solution) const {
    if (hasVelocityIC()) {
        MessagePrinter::printErrorTxt(
            "ICSystem::applyInitialConditions: velocity ICs are registered; "
            "use the overload that also receives InitialVelocity");
        MessagePrinter::exitPeriX();
    }
    applyField(Mesh,DofsPerNode,ICField::Solution,Solution);
}

void ICSystem::applyInitialConditions(const PDMesh &Mesh,
                                      const int &DofsPerNode,
                                      VectorXd &Solution,
                                      VectorXd &InitialVelocity) const {
    const int expectedSize=Mesh.getNodesNum()*DofsPerNode;
    if (Solution.getSize()!=expectedSize
        || InitialVelocity.getSize()!=expectedSize) {
        MessagePrinter::printErrorTxt(
            "ICSystem::applyInitialConditions: solution and initial-velocity "
            "vectors must both have size "+std::to_string(expectedSize));
        MessagePrinter::exitPeriX();
    }
    applyField(Mesh,DofsPerNode,ICField::Solution,Solution);
    applyField(Mesh,DofsPerNode,ICField::Velocity,InitialVelocity);
}

void ICSystem::printICSystemInfo() const {
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("IC system info");
    MessagePrinter::printNormalTxt(
        "  registered ICs="+std::to_string(getICsNum()));
    for (const auto &entry : m_ICs) {
        std::string dofs;
        for (std::size_t i=0;i<entry.IC->getDofs().size();++i) {
            if (i>0) dofs+=",";
            dofs+=std::to_string(entry.IC->getDofs()[i]);
        }
        const std::string phyName=entry.PhyName.empty()
                                ? "<all-pd-nodes>" : entry.PhyName;
        MessagePrinter::printNormalTxt(
            "  - name='"+entry.Name
            +"', type='"+entry.IC->getICType()
            +"', field='"+fieldName(entry.Field)
            +"', phygroup='"+phyName
            +"', dofs=["+dofs+"]");
    }
    MessagePrinter::printStars();
}
