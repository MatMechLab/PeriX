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
//+++ Date    : 2026.06.05
//+++ Function: matrix-free explicit-DYNAMICS rate assembly, serial or
//+++           OpenMP. For a second-order (elastodynamic) explicit kernel
//+++           the strong-form PDDO force and per-bond damage update
//+++           are evaluated once per step inside the element's
//+++           preprocessIteration(), which caches the resulting nodal
//+++           acceleration a_i=(L_i+b_i)/rho and itself fans out across
//+++           threads when Info.UseParallel is set. This routine then
//+++           performs a NODAL-only gather of that cached acceleration
//+++           into Rate via computeNodalResidual, so no PDDO moment matrix
//+++           is factorised and the family loop is not repeated here. The
//+++           gather is likewise split across OMP_NUM_THREADS threads when
//+++           parallel -- each node writes only its own rows, so there is
//+++           no shared write and the result is bitwise identical to the
//+++           serial gather. The central-difference integrator advances
//+++             u^{n+1} = 2 u^n - u^{n-1} + dt^2 * Rate(u^n).
//+++           Index layout matches the implicit assembler:
//+++             g = (NodeID-1)*ndof + a   (1-based).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDSystem/PDSystem.h"

#include "Utils/MessagePrinter.h"

void PDSystem::formExplicitDynamicsRate(const PDMesh &Mesh,
                                        PDOperators &t_PDOperators,
                                        const ElmtSystem &t_ElmtSystem,
                                        const LocalElmtInfo &Info,
                                        const VectorXd &U,
                                        const VectorXd &Uold,
                                        VectorXd &Rate) {
    if (t_ElmtSystem.getElementsNum()==0) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitDynamicsRate: ElmtSystem has no element registered");
        MessagePrinter::exitPeriX();
    }

    const int ndof=t_ElmtSystem.getMaxDofsPerNode();
    const int NodesNum=Mesh.getNodesNum();

    if (Rate.getSize()!=NodesNum*ndof) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitDynamicsRate: Rate size mismatch ("
                                      +std::to_string(Rate.getSize())+" vs "
                                      +std::to_string(NodesNum*ndof)+")");
        MessagePrinter::exitPeriX();
    }
    if (Uold.getSize()!=U.getSize()) {
        MessagePrinter::printErrorTxt("PDSystem::formExplicitDynamicsRate: Uold size ("
                                      +std::to_string(Uold.getSize())+") does not match U ("
                                      +std::to_string(U.getSize())+")");
        MessagePrinter::exitPeriX();
    }

    // The force and bond-damage update are computed here once per step, and
    // the per-node acceleration is cached inside the
    // kernel; the kernel parallelises that pass internally when Info.UseParallel
    // (this is the dominant cost of an explicit-dynamics step). U is the current
    // level u^n.
    t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,Info,ndof,U,Uold);

    Rate.setToZeros();

    LocalElmtInfo nodeInfo=Info;
    nodeInfo.Xsi[0]=nodeInfo.Xsi[1]=nodeInfo.Xsi[2]=0.0;
    nodeInfo.XsiNorm=0.0;

    // Single code path for serial and OpenMP. The gather is a pure per-node read
    // of the cached acceleration, so each node writes only its own rows of Rate;
    // with Info.UseParallel==false it runs on one thread (bitwise identical to
    // the original serial gather), with it true it fans out without contention.
#pragma omp parallel if(Info.UseParallel)
    {
        VectorXd U_I(ndof), Uold_I(ndof), localR_node(ndof);

#pragma omp for schedule(static)
        for (int i=1;i<=NodesNum;i++) {
            const double volume=Mesh.getIthNodeVolume(i);
            const int rowBaseI=(i-1)*ndof;
            for (int a=1;a<=ndof;a++) {
                U_I(a)=U(rowBaseI+a);
                Uold_I(a)=Uold(rowBaseI+a);
            }

            // gather the cached nodal acceleration a_i (du2/dt2 = a_i).
            t_ElmtSystem.computeNodalResidual(nodeInfo,i,U_I,Uold_I,volume,localR_node);
            for (int a=1;a<=ndof;a++) {
                Rate.addValue(rowBaseI+a,localR_node(a));
            }
        }
    }
}
