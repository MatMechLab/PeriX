//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/PDDODynamicFracElement.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
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

PDOperators makeOperators(const double gxx,const double gyy,
                          const double gxy) {
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
    values[static_cast<std::size_t>(
        operators.getOperatorIndex("d2/dxdy")-1)]=gxy;
    operators.loadOperatorValues(values.data());
    return operators;
}

PDMesh makeReciprocalBondMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.NodesNum=2;
    data.BulkElmtsNum=2;
    data.HorizonRadius=1.0;
    data.NodesNeighNodesID={{2},{1}};
    data.NodesElmtID={1,2};
    data.NodeCoords={0.0,0.0,0.0, 1.0,0.0,0.0};
    data.NodeVolumes={1.0,1.0};
    return mesh;
}
}

int main() {
    constexpr double young=100.0;
    constexpr double nu=0.25;
    constexpr double rho=4.0;
    constexpr double gxx=0.5;
    constexpr double gyy=-0.25;
    constexpr double gxy=0.2;

    PDDODynamicFracElement element;
    element.setDim(2);
    element.setE(young);
    element.setNu(nu);
    element.setRho(rho);
    element.setState(
        PDDODynamicFracElement::StressState::PlaneStress);
    element.setBodyForce(1.0,-2.0,0.0);

    PDOperators operators=makeOperators(gxx,gyy,gxy);
    const LocalElmtInfo bondInfo{};
    VectorXd uI(2,0.0);
    VectorXd uJ(2,0.0);
    uI(1)=0.1;
    uI(2)=-0.3;
    uJ(1)=0.4;
    uJ(2)=-0.5;
    const VectorXd uOldI(2,0.0);
    const VectorXd uOldJ(2,0.0);
    VectorXd residual(2,0.0);
    MatrixXd kii(2,2,0.0);
    MatrixXd kij(2,2,0.0);
    element.computeBondResidualAndJacobian(
        operators,bondInfo,1,2,uI,uJ,uOldI,uOldJ,1.0,
        residual,kii,kij);

    const double c11=young/(1.0-nu*nu);
    const double c12=young*nu/(1.0-nu*nu);
    const double c66=young/(2.0*(1.0+nu));
    const double p11=c11*gxx+c66*gyy;
    const double p22=c66*gxx+c11*gyy;
    const double pxy=(c12+c66)*gxy;
    const double dux=uJ(1)-uI(1);
    const double duy=uJ(2)-uI(2);
    expectNear("Navier x residual",residual(1),p11*dux+pxy*duy);
    expectNear("Navier y residual",residual(2),pxy*dux+p22*duy);
    expectNear("xx self tangent",kii(1,1),-p11);
    expectNear("xy self tangent",kii(1,2),-pxy);
    expectNear("yx neighbor tangent",kij(2,1),pxy);
    expectNear("yy neighbor tangent",kij(2,2),p22);

    LocalElmtInfo transientInfo;
    transientInfo.Dt=0.5;
    uI(1)=0.7;
    uI(2)=-0.1;
    VectorXd oldNodal(2,0.0);
    oldNodal(1)=0.5;
    oldNodal(2)=-0.2;
    VectorXd nodalResidual(2,0.0);
    MatrixXd nodalJacobian(2,2,0.0);
    element.computeNodalResidualAndJacobian(
        transientInfo,1,uI,oldNodal,1.0,
        nodalResidual,nodalJacobian);
    const double inertia=rho/(transientInfo.Dt*transientInfo.Dt);
    expectNear("backward-Euler x inertia",
               nodalResidual(1),1.0-inertia*(uI(1)-oldNodal(1)));
    expectNear("backward-Euler y inertia",
               nodalResidual(2),-2.0-inertia*(uI(2)-oldNodal(2)));
    expectNear("x inertia tangent",nodalJacobian(1,1),-inertia);
    expectNear("y inertia tangent",nodalJacobian(2,2),-inertia);

    PDMesh mesh=makeReciprocalBondMesh();
    element.setG0(1.0e-3);
    VectorXd initial(4,0.0);
    LocalElmtInfo stepInfo;
    stepInfo.Dt=0.1;
    stepInfo.Timestep=1;
    element.preprocessIteration(
        mesh,operators,stepInfo,2,initial,initial);

    VectorXd opened(initial);
    opened(3)=0.05;
    stepInfo.Timestep=2;
    element.preprocessIteration(
        mesh,operators,stepInfo,2,opened,opened);

    uI.setToZeros();
    uJ.setToZeros();
    uJ(1)=0.05;
    residual.setToZeros();
    kii.setToZeros();
    kij.setToZeros();
    element.computeBondResidualAndJacobian(
        operators,bondInfo,1,2,uI,uJ,uOldI,uOldJ,1.0,
        residual,kii,kij);
    expectNear("broken bond x force",residual(1),0.0);
    expectNear("broken bond y force",residual(2),0.0);
    expectNear("broken bond tangent",kij(1,1),0.0);

    std::vector<double> damage(1,0.0);
    element.computeNodalProjection(
        "damage",mesh,operators,opened,1,2,damage);
    expectNear("irreversible local damage",damage[0],1.0);

    expectTrue("element type",
               element.getElementType()=="pddo_dynamic_frac");
    expectTrue("displacement degrees of freedom",
               element.getDofNames()
                   ==std::vector<std::string>({"ux","uy"}));
    expectTrue("implicit kernel",!element.isExplicit());
    expectTrue("plane-stress state",element.getStateName()=="plane_stress");

    if (failures!=0) return 1;
    std::printf("Implicit PDDO dynamic-fracture element tests passed\n");
    return 0;
}
