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
//+++ Date    : 2026.04.15
//+++ Function: residual / jacobian assembly. Per-bond and per-node
//+++           physics is delegated to ElmtSystem. Matrix layout
//+++           is (NodesNum * ndof), with global index
//+++             g = (NodeID-1)*ndof + a   (1-based)
//+++           for the a-th DoF of NodeID.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDSystem/PDSystem.h"

#include <cmath>

#include "Utils/MessagePrinter.h"

void PDSystem::formResidualAndJacobian(const PDMesh &Mesh,
                                       PDOperators &t_PDOperators,
                                       const ElmtSystem &t_ElmtSystem,
                                       const LocalElmtInfo &Info,
                                       const VectorXd &U,
                                       const VectorXd &Uold,
                                       SparseMatrix &K,
                                       VectorXd &RHS) {
    if (t_ElmtSystem.getElementsNum()==0) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobian: ElmtSystem has no element registered");
        MessagePrinter::exitPeriX();
    }

    const int ndof=t_ElmtSystem.getMaxDofsPerNode();
    const int NodesNum=Mesh.getNodesNum();

    if (RHS.getSize()!=NodesNum*ndof) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobian: RHS size mismatch ("
                                      +std::to_string(RHS.getSize())+" vs "
                                      +std::to_string(NodesNum*ndof)+")");
        MessagePrinter::exitPeriX();
    }
    if (Uold.getSize()!=U.getSize()) {
        MessagePrinter::printErrorTxt("PDSystem::formResidualAndJacobian: Uold size ("
                                      +std::to_string(Uold.getSize())+") does not match U ("
                                      +std::to_string(U.getSize())+")");
        MessagePrinter::exitPeriX();
    }

    if (m_LocalR.getSize()!=ndof) {
        m_LocalR.resize(ndof);
        m_LocalR_node.resize(ndof);
        m_LocalK_II.resize(ndof,ndof);
        m_LocalK_IJ.resize(ndof,ndof);
        m_LocalK_node.resize(ndof,ndof);
        m_U_I.resize(ndof);
        m_U_J.resize(ndof);
        m_Uold_I.resize(ndof);
        m_Uold_J.resize(ndof);
    }

    // Let coupled kernels refresh their per-iteration frozen caches
    // before we touch K and RHS. Uold is also passed so time-split
    // kernels can evaluate explicit terms on the previous-step field.
    t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,Info,ndof,U,Uold);

    K.setToZeros();
    RHS.setToZeros();

    for (int i=1;i<=NodesNum;i++) {
        t_PDOperators.calcAMatrix(i,Mesh.getDataConstRef());
        const double volume=Mesh.getIthNodeVolume(i);

        const int rowBaseI=(i-1)*ndof;
        for (int a=1;a<=ndof;a++) {
            m_U_I(a)=U(rowBaseI+a);
            m_Uold_I(a)=Uold(rowBaseI+a);
        }

        // make a per-call copy so we can mutate per-bond fields below
        LocalElmtInfo bondInfo=Info;
        // clear stale per-bond fields for the nodal call
        bondInfo.Xsi[0]=bondInfo.Xsi[1]=bondInfo.Xsi[2]=0.0;
        bondInfo.XsiNorm=0.0;

        // ---- nodal contribution (e.g., source / body force / time-derivative) ----
        t_ElmtSystem.computeNodalResidualAndJacobian(bondInfo,i,m_U_I,m_Uold_I,volume,
                                                    m_LocalR_node,m_LocalK_node);
        for (int a=1;a<=ndof;a++) {
            RHS.addValue(rowBaseI+a,m_LocalR_node(a));
            for (int b=1;b<=ndof;b++) {
                K.addValue(rowBaseI+a,rowBaseI+b,-m_LocalK_node(a,b));
            }
        }

        // ---- bond contributions ----
        const auto &meshData=Mesh.getDataConstRef();
        for (const auto &NodeJ : Mesh.getIthNodeNeighborNodeIDs(i)) {
            t_PDOperators.calcOperators(i,NodeJ,meshData);
            const int rowBaseJ=(NodeJ-1)*ndof;
            for (int a=1;a<=ndof;a++) {
                m_U_J(a)=U(rowBaseJ+a);
                m_Uold_J(a)=Uold(rowBaseJ+a);
            }

            // populate per-bond fields of bondInfo (reference vector)
            bondInfo.Xsi[0]=meshData.NodeCoords[(NodeJ-1)*3+0]-meshData.NodeCoords[(i-1)*3+0];
            bondInfo.Xsi[1]=meshData.NodeCoords[(NodeJ-1)*3+1]-meshData.NodeCoords[(i-1)*3+1];
            bondInfo.Xsi[2]=meshData.NodeCoords[(NodeJ-1)*3+2]-meshData.NodeCoords[(i-1)*3+2];
            bondInfo.XsiNorm=std::sqrt(bondInfo.Xsi[0]*bondInfo.Xsi[0]
                                      +bondInfo.Xsi[1]*bondInfo.Xsi[1]
                                      +bondInfo.Xsi[2]*bondInfo.Xsi[2]);

            t_ElmtSystem.computeBondResidualAndJacobian(
                t_PDOperators,bondInfo,i,NodeJ,m_U_I,m_U_J,m_Uold_I,m_Uold_J,volume,
                m_LocalR,m_LocalK_II,m_LocalK_IJ);

            // Newton: K = -dF/dU, RHS = +F.
            for (int a=1;a<=ndof;a++) {
                RHS.addValue(rowBaseI+a,m_LocalR(a));
                for (int b=1;b<=ndof;b++) {
                    K.addValue(rowBaseI+a,rowBaseI+b,-m_LocalK_II(a,b));
                    K.addValue(rowBaseI+a,rowBaseJ+b,-m_LocalK_IJ(a,b));
                }
            }
        }
    }
}
