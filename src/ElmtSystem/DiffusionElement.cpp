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
//+++ Function: transient diffusion kernel in strong form
//+++           (backward Euler). Strong form:
//+++             dc/dt - D*laplacian(c) - f = 0
//+++           Per-bond contribution (V_j already inside the operator):
//+++             r_i^bond = -D*laplacian(i,j) * (c_j - c_i)
//+++           Per-node contribution (dt>0):
//+++             r_i^node = (c_i - c_i_old)/dt - f
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/DiffusionElement.h"

#include <algorithm>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

void DiffusionElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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
                                                     MatrixXd &LocalK_IJ) const {
    (void)Info; (void)NodeI; (void)NodeJ; (void)Uold_I; (void)Uold_J; (void)Volume;

    const double coeff = -m_D * pdLaplacian(t_PDOperators);

    LocalR(1)      =  coeff * (U_J(1) - U_I(1));
    LocalK_II(1,1) = -coeff;
    LocalK_IJ(1,1) =  coeff;
}

void DiffusionElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                       const int &NodeI,
                                                       const VectorXd &U_I,
                                                       const VectorXd &Uold_I,
                                                       const double  &Volume,
                                                       VectorXd &LocalR_node,
                                                       MatrixXd &LocalK_node) const {
    (void)NodeI; (void)Volume;
    if (Info.Dt<=0.0) {
        MessagePrinter::printErrorTxt("DiffusionElement is transient-only; got dt="
                                      +std::to_string(Info.Dt)
                                      +". Use PoissonElement for static analysis.");
        MessagePrinter::exitPeriX();
    }
    LocalR_node(1)   = (U_I(1) - Uold_I(1)) / Info.Dt - m_F;
    LocalK_node(1,1) = 1.0 / Info.Dt;
}

std::vector<ProjectionInfo> DiffusionElement::getAvailableProjections() const {
    return {
        {"gradient", ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)},
        {"flux",     ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)}
    };
}

void DiffusionElement::computeNodalProjection(const std::string &name,
                                              const PDMesh &Mesh,
                                              PDOperators &ops,
                                              const VectorXd &U,
                                              const int &NodeI,
                                              const int &DofsPerNode,
                                              std::vector<double> &out) const {
    if (DofsPerNode!=1) {
        std::fill(out.begin(),out.end(),0.0);
        return;
    }

    // PDDO discrete derivative (V_j baked in by PDOperators):
    //   dc/dx_alpha = sum_j (c_j - c_i) * gfun_alpha_ij
    const auto &neighbors=Mesh.getIthNodeNeighborNodeIDs(NodeI);
    const auto &meshData=Mesh.getDataConstRef();
    const bool is3D=(m_Dim>=3);
    double dcdx=0.0,dcdy=0.0,dcdz=0.0;
    for (const int NodeJ : neighbors) {
        ops.calcOperators(NodeI,NodeJ,meshData);
        const double dc=U(NodeJ)-U(NodeI);
        dcdx+=dc*ops("d/dx");
        dcdy+=dc*ops("d/dy");
        if (is3D) dcdz+=dc*ops("d/dz");
    }

    if (out.size()<3) return;
    std::fill(out.begin(),out.begin()+3,0.0);

    if (name=="gradient") {
        out[0]=dcdx; out[1]=dcdy; out[2]=dcdz;
    }
    else if (name=="flux") {
        out[0]=-m_D*dcdx;
        out[1]=-m_D*dcdy;
        out[2]=is3D ? -m_D*dcdz : 0.0;
    }
}
