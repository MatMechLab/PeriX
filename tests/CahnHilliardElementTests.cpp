//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/CahnHilliardElement.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDOperators/PDOperators.h"

namespace {
int failures=0;

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-12) {
    const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
    if (std::fabs(actual-expected)<=tolerance*scale) return;
    std::printf("FAIL %-40s actual=% .16e expected=% .16e\n",
                name.c_str(),actual,expected);
    ++failures;
}

void expectTrue(const std::string &name,const bool value) {
    if (value) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

PDOperators makeOperators(const double laplacian) {
    PDOperators operators;
    operators.setDim(2);
    operators.setOrder(2);
    operators.setup();

    std::vector<double> values(
        static_cast<std::size_t>(operators.getOperatorsVecSize()),0.0);
    values[static_cast<std::size_t>(
        operators.getOperatorIndex("d2/dx2")-1)]=0.75*laplacian;
    values[static_cast<std::size_t>(
        operators.getOperatorIndex("d2/dy2")-1)]=0.25*laplacian;
    operators.loadOperatorValues(values.data());
    return operators;
}

double regularSolutionDerivative(const double c,const double chi) {
    return std::log(c/(1.0-c))+chi*(1.0-2.0*c);
}

double regularSolutionSecondDerivative(const double c,const double chi) {
    return 1.0/(c*(1.0-c))-2.0*chi;
}
}

int main() {
    constexpr double chi=2.6;
    constexpr double kappa=0.02;
    constexpr double mobilityPrefactor=3.0;
    constexpr double laplacian=0.8;
    constexpr double cI=0.25;
    constexpr double cJ=0.4;
    constexpr double muI=-0.2;
    constexpr double muJ=0.3;

    CahnHilliardElement element;
    element.setChi(chi);
    element.setKappa(kappa);
    element.setMobility(mobilityPrefactor);
    const PDOperators operators=makeOperators(laplacian);

    const LocalElmtInfo steadyInfo{};
    VectorXd uI(2,0.0);
    VectorXd uJ(2,0.0);
    uI(1)=cI;
    uI(2)=muI;
    uJ(1)=cJ;
    uJ(2)=muJ;
    const VectorXd uOldI(2,0.0);
    const VectorXd uOldJ(2,0.0);
    VectorXd residual(2,0.0);
    MatrixXd kii(2,2,0.0);
    MatrixXd kij(2,2,0.0);
    element.computeBondResidualAndJacobian(
        operators,steadyInfo,1,2,uI,uJ,uOldI,uOldJ,1.0,
        residual,kii,kij);

    const double mobility=mobilityPrefactor*cI*(1.0-cI);
    const double concentrationCoefficient=-mobility*laplacian;
    const double chemicalCoefficient=kappa*laplacian;
    expectNear("conservative chemical-potential flux",
               residual(1),concentrationCoefficient*(muJ-muI));
    expectNear("gradient-energy contribution",
               residual(2),chemicalCoefficient*(cJ-cI));
    expectNear("c-mu self Jacobian",kii(1,2),-concentrationCoefficient);
    expectNear("c-mu neighbor Jacobian",kij(1,2),concentrationCoefficient);
    expectNear("mu-c self Jacobian",kii(2,1),-chemicalCoefficient);
    expectNear("mu-c neighbor Jacobian",kij(2,1),chemicalCoefficient);

    LocalElmtInfo transientInfo;
    transientInfo.Dt=0.1;
    const double equilibriumMu=regularSolutionDerivative(cI,chi);
    uI(2)=equilibriumMu;
    VectorXd old(2,0.0);
    old(1)=0.2;
    VectorXd nodalResidual(2,0.0);
    MatrixXd nodalJacobian(2,2,0.0);
    element.computeNodalResidualAndJacobian(
        transientInfo,1,uI,old,1.0,nodalResidual,nodalJacobian);

    expectNear("concentration time derivative",
               nodalResidual(1),(cI-old(1))/transientInfo.Dt);
    expectNear("uniform chemical equilibrium",nodalResidual(2),0.0);
    expectNear("concentration time Jacobian",
               nodalJacobian(1,1),1.0/transientInfo.Dt);
    expectNear("regular-solution curvature",
               nodalJacobian(2,1),
               -regularSolutionSecondDerivative(cI,chi));
    expectNear("chemical-potential Jacobian",nodalJacobian(2,2),1.0);

    expectTrue("element type",element.getElementType()=="cahnhilliard");
    expectTrue("split c-mu degrees of freedom",
               element.getDofNames()
                   ==std::vector<std::string>({"c","mu"}));
    expectTrue("ghost-dropped conserved fields",
               element.getGhostDropSpeciesDofSlots()
                   ==std::vector<int>({0,1}));

    if (failures!=0) return 1;
    std::printf("Cahn-Hilliard element tests passed\n");
    return 0;
}
