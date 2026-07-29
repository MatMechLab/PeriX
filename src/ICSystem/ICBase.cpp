//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "ICSystem/ICBase.h"

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

void ICBase::validateApplication(const PDMesh &Mesh,
                                 const std::vector<int> &NodeIDs,
                                 const int &DofsPerNode,
                                 const VectorXd &Values) const {
    if (DofsPerNode<1) {
        MessagePrinter::printErrorTxt(
            "ICBase::apply: DofsPerNode must be positive");
        MessagePrinter::exitPeriX();
    }

    const int expectedSize=Mesh.getNodesNum()*DofsPerNode;
    if (Values.getSize()!=expectedSize) {
        MessagePrinter::printErrorTxt(
            "ICBase::apply: target vector has size "+std::to_string(Values.getSize())
            +", expected "+std::to_string(expectedSize));
        MessagePrinter::exitPeriX();
    }

    if (m_Dofs.empty()) {
        MessagePrinter::printErrorTxt(
            "ICBase::apply: at least one target DoF is required");
        MessagePrinter::exitPeriX();
    }
    for (const int dof : m_Dofs) {
        if (dof<1 || dof>DofsPerNode) {
            MessagePrinter::printErrorTxt(
                "ICBase::apply: target DoF "+std::to_string(dof)
                +" is outside [1,"+std::to_string(DofsPerNode)+"]");
            MessagePrinter::exitPeriX();
        }
    }

    for (const int nodeID : NodeIDs) {
        if (nodeID<1 || nodeID>Mesh.getNodesNum()) {
            MessagePrinter::printErrorTxt(
                "ICBase::apply: PD node "+std::to_string(nodeID)
                +" is outside [1,"+std::to_string(Mesh.getNodesNum())+"]");
            MessagePrinter::exitPeriX();
        }
    }
}

void ICBase::apply(const PDMesh &Mesh,
                   const std::vector<int> &NodeIDs,
                   const int &DofsPerNode,
                   VectorXd &Values) const {
    validateApplication(Mesh,NodeIDs,DofsPerNode,Values);

    for (const int nodeID : NodeIDs) {
        const double x=Mesh.getIthNodeJthCoord(nodeID,1);
        const double y=Mesh.getIthNodeJthCoord(nodeID,2);
        const double z=Mesh.getIthNodeJthCoord(nodeID,3);
        const int rowBase=(nodeID-1)*DofsPerNode;
        for (std::size_t component=0;component<m_Dofs.size();++component) {
            Values(rowBase+m_Dofs[component])=
                computeValue(x,y,z,static_cast<int>(component));
        }
    }
}
