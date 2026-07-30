//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/PoissonElement.h"
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
    constexpr double sigma=2.5;
    constexpr double source=-3.0;
    constexpr double gxx=1.25;
    constexpr double gyy=-0.5;
    constexpr double ui=1.2;
    constexpr double uj=-0.4;

    PoissonElement element(sigma,source);
    const PDOperators operators=makeOperators(gxx,gyy);
    const LocalElmtInfo info{};
    const VectorXd uI(1,ui);
    const VectorXd uJ(1,uj);
    const VectorXd uOldI(1,0.0);
    const VectorXd uOldJ(1,0.0);
    VectorXd residual(1,0.0);
    MatrixXd kii(1,1,0.0);
    MatrixXd kij(1,1,0.0);

    element.computeBondResidualAndJacobian(
        operators,info,1,2,uI,uJ,uOldI,uOldJ,1.0,
        residual,kii,kij);

    const double coefficient=sigma*(gxx+gyy);
    expectNear("bond residual",residual(1),coefficient*(uj-ui));
    expectNear("self Jacobian",kii(1,1),-coefficient);
    expectNear("neighbor Jacobian",kij(1,1),coefficient);

    VectorXd nodalResidual(1,0.0);
    MatrixXd nodalJacobian(1,1,0.0);
    element.computeNodalResidualAndJacobian(
        info,1,uI,uOldI,1.0,nodalResidual,nodalJacobian);
    expectNear("source residual",nodalResidual(1),source);
    expectNear("source Jacobian",nodalJacobian(1,1),0.0);
    expectNear("assembled strong form",
               residual(1)+nodalResidual(1),
               sigma*(gxx+gyy)*(uj-ui)+source);

    expectTrue("element type",element.getElementType()=="poisson");
    expectTrue("single u degree of freedom",
               element.getDofNames()==std::vector<std::string>{"u"});

    if (failures!=0) return 1;
    std::printf("Poisson element tests passed\n");
    return 0;
}
