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
//+++ Function: transient diffusion element kernel
//+++           (backward Euler, PDDO, strong form).
//+++           Strong form:
//+++             dc/dt - D * laplacian(c) - f = 0
//+++           Per-bond contribution to F_i (V_j already inside the
//+++           operator value returned by PDOperators):
//+++             r_i^bond = -D*laplacian(i,j) * (c_j - c_i)
//+++           Per-node contribution to F_i (transient only, dt>0):
//+++             r_i^node = (c_i - c_i_old)/dt - f
//+++           The kernel is transient-only; passing dt=0 is an error
//+++           (use PoissonElement for the static problem).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "ElmtSystem/ElementBase.h"

class DiffusionElement final : public ElementBase {
public:
    DiffusionElement()=default;
    DiffusionElement(const double &D,const double &f)
        :m_D(D),m_F(f) {}

    [[nodiscard]] int getDofsPerNode() const override { return 1; }
    [[nodiscard]] std::string getElementType() const override { return "diffusion"; }
    [[nodiscard]] std::vector<std::string> getDofNames() const override { return {"c"}; }

    void setD(const double &D) { m_D=D; }
    [[nodiscard]] double getD() const { return m_D; }
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
    double m_D=1.0;       /**< diffusivity */
    double m_F=0.0;       /**< volumetric source */
};
