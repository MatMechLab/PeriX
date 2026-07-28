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
//+++ Function: poisson element kernel.
//+++           Strong form (2D):  sigma * laplacian(u) + f = 0
//+++           Per-bond contribution to F_i (V_j already inside the
//+++           operator value returned by PDOperators):
//+++             r_i^bond = sigma * (d2/dx2 + d2/dy2)(i,j) * (u_j - u_i)
//+++           Per-node contribution to F_i:
//+++             r_i^node = f
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "ElmtSystem/ElementBase.h"

class PoissonElement final : public ElementBase {
public:
    PoissonElement()=default;
    explicit PoissonElement(const double &sigma):m_Sigma(sigma) {}
    PoissonElement(const double &sigma,const double &f):m_Sigma(sigma),m_F(f) {}

    [[nodiscard]] int getDofsPerNode() const override { return 1; }
    [[nodiscard]] std::string getElementType() const override { return "poisson"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override { return {"u"}; }

    void setSigma(const double &sigma) { m_Sigma=sigma; }
    [[nodiscard]] double getSigma() const { return m_Sigma; }

    void setF(const double &f) { m_F=f; }
    [[nodiscard]] double getF() const { return m_F; }

    void computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
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
                                        MatrixXd &LocalK_IJ) const override;

    void computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                         const int &NodeI,
                                         const VectorXd &U_I,
                                         const VectorXd &Uold_I,
                                         const double  &Volume,
                                         VectorXd &LocalR_node,
                                         MatrixXd &LocalK_node) const override;

    [[nodiscard]] std::vector<ProjectionInfo> getAvailableProjections() const override;

    void computeNodalProjection(const std::string &name,
                                const PDMesh &Mesh,
                                PDOperators &ops,
                                const VectorXd &U,
                                const int &NodeI,
                                const int &DofsPerNode,
                                std::vector<double> &out) const override;

private:
    double m_Sigma=1.0;
    double m_F=0.0;
};
