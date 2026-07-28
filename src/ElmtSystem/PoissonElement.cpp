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
//+++ Function: Poisson kernel in strong form
//+++             sigma * laplacian(u) + f = 0
//+++           Per-bond contribution (V_j already inside the operator):
//+++             r_i^bond = sigma * (d2/dx2 + d2/dy2)(i,j) * (u_j - u_i)
//+++           Per-node contribution:
//+++             r_i^node = f
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/PoissonElement.h"

#include <algorithm>

#include "PDMesh/PDMesh.h"

void PoissonElement::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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

    const double Lap   = pdLaplacian(t_PDOperators);
    const double coeff = m_Sigma * Lap;

    LocalR(1)       =  coeff * (U_J(1) - U_I(1));
    LocalK_II(1,1)  = -coeff;
    LocalK_IJ(1,1)  =  coeff;
}

void PoissonElement::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                     const int &NodeI,
                                                     const VectorXd &U_I,
                                                     const VectorXd &Uold_I,
                                                     const double  &Volume,
                                                     VectorXd &LocalR_node,
                                                     MatrixXd &LocalK_node) const {
    (void)Info; (void)NodeI; (void)U_I; (void)Uold_I; (void)Volume;
    LocalR_node(1)   = m_F;
    LocalK_node(1,1) = 0.0;
}

std::vector<ProjectionInfo> PoissonElement::getAvailableProjections() const {
    return {
        {"gradient", ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)},
        {"flux",     ProjectionType::Vector, projectionComponentsFor(ProjectionType::Vector)}
    };
}

void PoissonElement::computeNodalProjection(const std::string &name,
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

    // Discrete gradient via PDDO 1st-order operators (V_j baked in by PDOperators):
    //   du/dx_alpha = sum_j (u_j - u_i) * gfun_alpha_ij
    const auto &neighbors=Mesh.getIthNodeNeighborNodeIDs(NodeI);
    const auto &meshData=Mesh.getDataConstRef();
    const bool is3D=(m_Dim>=3);
    double dudx=0.0,dudy=0.0,dudz=0.0;
    for (const int NodeJ : neighbors) {
        ops.calcOperators(NodeI,NodeJ,meshData);
        const double du=U(NodeJ)-U(NodeI);
        dudx+=du*ops("d/dx");
        dudy+=du*ops("d/dy");
        if (is3D) dudz+=du*ops("d/dz");
    }

    if (out.size()<3) return;
    std::fill(out.begin(),out.begin()+3,0.0);

    if (name=="gradient") {
        out[0]=dudx; out[1]=dudy; out[2]=dudz;
    }
    else if (name=="flux") {
        // flux = -sigma * grad(u) (Fourier / Fick form)
        out[0]=-m_Sigma*dudx;
        out[1]=-m_Sigma*dudy;
        out[2]=-m_Sigma*dudz;
    }
}
