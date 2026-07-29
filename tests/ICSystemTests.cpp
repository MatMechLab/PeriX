//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "ICSystem/BoxIC.h"
#include "ICSystem/CircleIC.h"
#include "ICSystem/ConstantIC.h"
#include "ICSystem/CosineIC.h"
#include "ICSystem/EllipseIC.h"
#include "ICSystem/GaussianIC.h"
#include "ICSystem/ICSystem.h"
#include "ICSystem/LinearIC.h"
#include "ICSystem/RandomIC.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"

namespace {
int failures=0;

void expectNear(const std::string &name,
                const double actual,
                const double expected,
                const double tolerance=1.0e-12) {
    const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
    if (std::fabs(actual-expected)<=tolerance*scale) return;
    std::printf("FAIL %-44s actual=% .16e expected=% .16e\n",
                name.c_str(),actual,expected);
    ++failures;
}

void expectTrue(const std::string &name,const bool value) {
    if (value) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

PDMesh makeProfileMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.DZ=1.0;
    data.NodesNum=5;
    data.BulkElmtsNum=5;
    data.NodeCoords={
        0.0,0.0,0.0,
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.5,0.5,0.0,
        0.0,0.0,1.0
    };
    data.NodeVolumes={1.0,1.0,1.0,1.0,1.0};
    data.PhyNameToNodeIDsMap["pair"]={2,4};
    return mesh;
}

PDMesh makeImpactLayerMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=0.001;
    data.DY=0.1/99.0;
    data.DZ=0.0;
    data.NodesNum=5;
    data.BulkElmtsNum=4;
    data.NodeCoords={
        0.1,0.1-0.5*data.DY,0.0,
        0.1,0.1-1.5*data.DY,0.0,
        0.1,0.1-2.5*data.DY,0.0,
        0.1,0.1-3.5*data.DY,0.0,
        0.1,0.1+0.5*data.DY,0.0
    };
    data.NodeVolumes={1.0,1.0,1.0,1.0,1.0};
    return mesh;
}

void testRegistryLayeringAndVelocityField() {
    PDMesh mesh=makeProfileMesh();
    VectorXd solution(mesh.getNodesNum()*2,0.0);
    VectorXd velocity(mesh.getNodesNum()*2,0.0);

    ICSystem system;

    auto background=std::make_unique<ConstantIC>(
        std::vector<double>{2.0,-3.0});
    background->setDofs({1,2});
    system.addIC("background","",std::move(background));

    auto overrideValue=std::make_unique<ConstantIC>(7.0);
    overrideValue->setDofs({1});
    system.addIC("pair_override","pair",std::move(overrideValue));

    auto initialSpeed=std::make_unique<ConstantIC>(-22.0);
    initialSpeed->setDofs({2});
    system.addIC("pair_velocity","pair",std::move(initialSpeed),
                 ICField::Velocity);

    system.applyInitialConditions(mesh,2,solution,velocity);

    expectTrue("registry name lookup",system.hasIC("pair_override"));
    expectTrue("registry count",system.getICsNum()==3);
    expectTrue("registry velocity flag",system.hasVelocityIC());
    expectNear("global constant dof 1",solution(1),2.0);
    expectNear("global constant dof 2",solution(2),-3.0);
    expectNear("later IC overrides selected node",solution(3),7.0);
    expectNear("later IC preserves other dof",solution(4),-3.0);
    expectNear("velocity separated from solution",velocity(4),-22.0);
    expectNear("velocity untouched outside group",velocity(2),0.0);
}

void testLinearBoxCircleAndEllipse() {
    LinearIC linear;
    linear.setOffsets({1.0,2.0});
    linear.setSlopeX({2.0});
    linear.setSlopeY({3.0,4.0});
    linear.setSlopeZ({0.5});
    expectNear("linear component 1",linear.computeValue(1.0,2.0,3.0,0),10.5);
    expectNear("linear component 2",linear.computeValue(1.0,2.0,3.0,1),13.5);

    BoxIC box;
    box.setMinCorner({0.0,0.0,0.0});
    box.setMaxCorner({1.0,1.0,0.0});
    box.setInsideValues({4.0});
    box.setOutsideValues({-1.0});
    expectNear("box inside and degenerate z",
               box.computeValue(0.5,0.5,99.0,0),4.0);
    expectNear("box outside",box.computeValue(1.5,0.5,0.0,0),-1.0);

    CircleIC circle;
    circle.setCenter({0.0,0.0,0.0});
    circle.setRadius(1.0);
    circle.setTransitionThickness(1.0);
    circle.setInsideValues({2.0});
    circle.setOutsideValues({0.0});
    expectNear("circle inside",circle.computeValue(0.0,0.0,1.0,0),2.0);
    expectNear("circle C1 midpoint",circle.computeValue(1.5,0.0,0.0,0),1.0);
    expectNear("circle outside",circle.computeValue(2.0,0.0,0.0,0),0.0);

    EllipseIC ellipse;
    ellipse.setCenter({0.0,0.0,0.0});
    ellipse.setSemiAxes({2.0,1.0,0.0});
    ellipse.setInsideValues({3.0});
    ellipse.setOutsideValues({-1.0});
    expectNear("ellipse hard inside",ellipse.computeValue(1.0,0.0,7.0,0),3.0);
    expectNear("ellipse includes boundary",ellipse.computeValue(2.0,0.0,0.0,0),3.0);
    expectNear("ellipse hard outside",ellipse.computeValue(2.1,0.0,0.0,0),-1.0);
    ellipse.setSmoothness(4.0);
    expectNear("ellipse smooth boundary midpoint",
               ellipse.computeValue(2.0,0.0,0.0,0),1.0);
}

void testGaussianAndCosine() {
    GaussianIC gaussian;
    gaussian.setCenter({0.0,0.0,0.0});
    gaussian.setSigma({1.0,2.0,0.0});
    gaussian.setAmplitudes({2.0});
    gaussian.setOffsets({1.0});
    expectNear("Gaussian center",gaussian.computeValue(0.0,0.0,50.0,0),3.0);
    expectNear("Gaussian anisotropic distance",
               gaussian.computeValue(1.0,2.0,50.0,0),
               1.0+2.0*std::exp(-1.0));

    CosineIC cosine;
    const double pi=std::acos(-1.0);
    cosine.setWaveNumbers({pi,0.0,0.0});
    cosine.setPhases({0.0,0.0,0.0});
    cosine.setAmplitudes({0.25});
    cosine.setOffsets({0.5});
    expectNear("cosine x mode at zero",cosine.computeValue(0.0,3.0,4.0,0),0.75);
    expectNear("cosine x mode at one",cosine.computeValue(1.0,3.0,4.0,0),0.25);
}

void testSeededRandomReproducibility() {
    PDMesh mesh=makeProfileMesh();
    VectorXd first(mesh.getNodesNum(),0.0);
    VectorXd second(mesh.getNodesNum(),0.0);
    VectorXd different(mesh.getNodesNum(),0.0);
    std::vector<int> nodes{1,2,3,4,5};

    RandomIC randomA;
    randomA.setMin(0.49);
    randomA.setMax(0.51);
    randomA.setSeed(12345);
    randomA.apply(mesh,nodes,1,first);

    RandomIC randomB;
    randomB.setMin(0.49);
    randomB.setMax(0.51);
    randomB.setSeed(12345);
    randomB.apply(mesh,nodes,1,second);

    RandomIC randomC;
    randomC.setMin(0.49);
    randomC.setMax(0.51);
    randomC.setSeed(12346);
    randomC.apply(mesh,nodes,1,different);

    bool differs=false;
    for (int i=1;i<=mesh.getNodesNum();++i) {
        expectNear("same seed reproduces sample "+std::to_string(i),
                   first(i),second(i),0.0);
        expectTrue("seeded sample is in [min,max)",
                   first(i)>=0.49 && first(i)<0.51);
        differs=differs || first(i)!=different(i);
    }
    expectTrue("different seed changes field",differs);
}

void testKalthoffFirstThreeLayers() {
    PDMesh mesh=makeImpactLayerMesh();
    VectorXd solution(mesh.getNodesNum()*2,0.0);
    VectorXd velocity(mesh.getNodesNum()*2,0.0);

    auto impact=std::make_unique<BoxIC>();
    impact->setDofs({1,2});
    impact->setMinCorner({0.075,0.096969696969697,0.0});
    impact->setMaxCorner({0.125,0.1,0.0});
    impact->setInsideValues({0.0,-22.0});
    impact->setOutsideValues({0.0,0.0});

    ICSystem system;
    system.addIC("impact_initial_velocity","",std::move(impact),
                 ICField::Velocity);
    system.applyInitialConditions(mesh,2,solution,velocity);

    expectNear("Kalthoff top layer velocity",velocity(2),-22.0);
    expectNear("Kalthoff second layer velocity",velocity(4),-22.0);
    expectNear("Kalthoff third layer velocity",velocity(6),-22.0);
    expectNear("Kalthoff fourth layer remains at rest",velocity(8),0.0);
    expectNear("Kalthoff impact ghost remains at rest",velocity(10),0.0);
}
}

int main() {
    testRegistryLayeringAndVelocityField();
    testLinearBoxCircleAndEllipse();
    testGaussianAndCosine();
    testSeededRandomReproducibility();
    testKalthoffFirstThreeLayers();

    if (failures==0) {
        std::printf("ICSystem tests passed\n");
        return 0;
    }
    std::printf("ICSystem tests failed: %d\n",failures);
    return 1;
}
