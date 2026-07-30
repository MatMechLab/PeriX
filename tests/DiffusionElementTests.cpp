//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/DiffusionElement.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDOperators/PDOperators.h"

namespace {
int failures=0;

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-12) {
    const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
    if (std::fabs(actual-expected)<=tolerance*scale) return;
    std::printf("FAIL %-36s actual=% .16e expected=% .16e\n",
                name.c_str(),actual,expected);
    ++failures;
}

void expectTrue(const std::string &name,const bool value) {
    if (value) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

PDOperators makeOperators(const double gxx,const double gyy) {
    PDOperators operators;
    operators.setDim(2);
    operators.setOrder(2);
    operators.setup();

    std::vector<double> values(
        static_cast<std::size_t>(operators.getOperatorsVecSize()),0.0);
    values[static_cast<std::size_t>(
        operators.getOperatorIndex("d2/dx2")-1)]=gxx;
    values[static_cast<std::size_t>(
        operators.getOperatorIndex("d2/dy2")-1)]=gyy;
    operators.loadOperatorValues(values.data());
    return operators;
}
}

int main() {
    constexpr double diffusivity=0.4;
    constexpr double source=-0.7;
    constexpr double dt=0.25;
    constexpr double cI=0.8;
    constexpr double cJ=1.3;
    constexpr double cOld=0.6;
    constexpr double gxx=0.9;
    constexpr double gyy=0.6;

    DiffusionElement element(diffusivity,source);
    const PDOperators operators=makeOperators(gxx,gyy);
    LocalElmtInfo info;
    info.Dt=dt;

    const VectorXd uI(1,cI);
    const VectorXd uJ(1,cJ);
    const VectorXd uOldI(1,cOld);
    const VectorXd uOldJ(1,0.0);
    VectorXd bondResidual(1,0.0);
    MatrixXd kii(1,1,0.0);
    MatrixXd kij(1,1,0.0);
    element.computeBondResidualAndJacobian(
        operators,info,1,2,uI,uJ,uOldI,uOldJ,1.0,
        bondResidual,kii,kij);

    const double coefficient=-diffusivity*(gxx+gyy);
    expectNear("diffusive bond residual",
               bondResidual(1),coefficient*(cJ-cI));
    expectNear("diffusive self Jacobian",kii(1,1),-coefficient);
    expectNear("diffusive neighbor Jacobian",kij(1,1),coefficient);

    VectorXd nodalResidual(1,0.0);
    MatrixXd nodalJacobian(1,1,0.0);
    element.computeNodalResidualAndJacobian(
        info,1,uI,uOldI,1.0,nodalResidual,nodalJacobian);
    expectNear("backward-Euler residual",
               nodalResidual(1),(cI-cOld)/dt-source);
    expectNear("backward-Euler Jacobian",nodalJacobian(1,1),1.0/dt);
    expectNear("assembled transient equation",
               bondResidual(1)+nodalResidual(1),
               (cI-cOld)/dt
                   -diffusivity*(gxx+gyy)*(cJ-cI)-source);

    expectTrue("element type",element.getElementType()=="diffusion");
    expectTrue("single concentration degree of freedom",
               element.getDofNames()==std::vector<std::string>{"c"});
    expectNear("diffusivity accessor",element.getD(),diffusivity);
    expectNear("source accessor",element.getF(),source);

    if (failures!=0) return 1;
    std::printf("Diffusion element tests passed\n");
    return 0;
}
