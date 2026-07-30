//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/ExplicitPDDOFracElement.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"

namespace {
int failures=0;

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-11) {
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

PDMesh makeNinePointMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.NodesNum=9;
    data.BulkElmtsNum=9;
    data.HorizonRadius=4.0;
    data.VolumeCorrection=false;
    data.NodeCoords.reserve(27);
    for (int j=-1;j<=1;++j) {
        for (int i=-1;i<=1;++i) {
            data.NodeCoords.push_back(static_cast<double>(i));
            data.NodeCoords.push_back(static_cast<double>(j));
            data.NodeCoords.push_back(0.0);
        }
    }
    data.NodeVolumes.assign(9,1.0);
    data.NodeSpacing.assign(9,1.0);
    data.NodesElmtID.resize(9);
    data.NodesNeighNodesID.resize(9);
    for (int i=1;i<=9;++i) {
        data.NodesElmtID[static_cast<std::size_t>(i-1)]=i;
        auto &neighbors=data.NodesNeighNodesID[static_cast<std::size_t>(i-1)];
        for (int j=1;j<=9;++j) {
            if (j!=i) neighbors.push_back(j);
        }
    }
    return mesh;
}

PDOperators makeOperators() {
    PDOperators operators;
    operators.setDim(2);
    operators.setOrder(2);
    operators.setup();
    return operators;
}
}

int main() {
    constexpr double young=100.0;
    constexpr double nu=0.25;
    constexpr double rho=4.0;

    PDMesh mesh=makeNinePointMesh();
    PDOperators operators=makeOperators();
    LocalElmtInfo info;
    info.Dt=1.0e-3;
    info.Timestep=1;

    ExplicitPDDOFracElement forced;
    forced.setDim(2);
    forced.setE(young);
    forced.setNu(nu);
    forced.setRho(rho);
    forced.setState(
        ExplicitPDDOFracElement::StressState::PlaneStress);
    forced.setDamageOn(false);
    forced.setBodyForce(8.0,-12.0,0.0);

    VectorXd zero(18,0.0);
    forced.preprocessIteration(
        mesh,operators,info,2,zero,zero);
    VectorXd localU(2,0.0);
    VectorXd acceleration(2,0.0);
    forced.computeNodalResidual(
        info,5,localU,localU,1.0,acceleration);
    expectNear("body-force x acceleration",acceleration(1),2.0);
    expectNear("body-force y acceleration",acceleration(2),-3.0);

    const double c11=young/(1.0-nu*nu);
    expectNear("CFL estimate",
               forced.estimateStableDt(mesh),1.0/std::sqrt(c11/rho));
    expectTrue("explicit kernel",forced.isExplicit());
    expectTrue("central-difference time order",forced.getTimeOrder()==2);
    expectTrue("displacement degrees of freedom",
               forced.getDofNames()
                   ==std::vector<std::string>({"ux","uy"}));

    ExplicitPDDOFracElement fractured;
    fractured.setDim(2);
    fractured.setE(young);
    fractured.setNu(nu);
    fractured.setRho(rho);
    fractured.setState(
        ExplicitPDDOFracElement::StressState::PlaneStress);
    fractured.setG0(1.0e-5);
    fractured.setDamageOn(true);
    fractured.setTensionOnly(true);

    constexpr double expansion=0.02;
    VectorXd expanded(18,0.0);
    for (int node=1;node<=mesh.getNodesNum();++node) {
        expanded((node-1)*2+1)=
            expansion*mesh.getIthNodeJthCoord(node,1);
        expanded((node-1)*2+2)=
            expansion*mesh.getIthNodeJthCoord(node,2);
    }
    fractured.preprocessIteration(
        mesh,operators,info,2,expanded,zero);

    std::vector<double> damage(1,0.0);
    fractured.computeNodalProjection(
        "damage",mesh,operators,expanded,5,2,damage);
    expectNear("critical-stretch bond failure",damage[0],1.0);

    info.Timestep=2;
    fractured.preprocessIteration(
        mesh,operators,info,2,zero,expanded);
    damage[0]=0.0;
    fractured.computeNodalProjection(
        "damage",mesh,operators,zero,5,2,damage);
    expectNear("irreversible explicit damage",damage[0],1.0);

    acceleration.setToZeros();
    fractured.computeNodalResidual(
        info,5,localU,localU,1.0,acceleration);
    expectNear("severed-family x acceleration",acceleration(1),0.0);
    expectNear("severed-family y acceleration",acceleration(2),0.0);

    if (failures!=0) return 1;
    std::printf("Explicit PDDO fracture element tests passed\n");
    return 0;
}
