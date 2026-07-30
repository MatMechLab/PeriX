//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ElmtSystem/FracStressCahnHilliardElement.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"

namespace {
int failures=0;

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-10) {
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

VectorXd localValues(const VectorXd &global,const int node) {
    VectorXd local(4,0.0);
    const int base=(node-1)*4;
    for (int dof=1;dof<=4;++dof) local(dof)=global(base+dof);
    return local;
}

double regularSolutionDerivative(const double c,const double chi) {
    return std::log(c/(1.0-c))+chi*(1.0-2.0*c);
}
}

int main() {
    constexpr double young=120.0;
    constexpr double nu=0.25;
    constexpr double omega=0.3;
    constexpr double cref=0.1;
    constexpr double chi=2.6;
    constexpr double concentration=0.4;
    constexpr double expansion=(omega/3.0)*(concentration-cref);

    PDMesh mesh=makeNinePointMesh();
    PDOperators operators=makeOperators();
    FracStressCahnHilliardElement element;
    element.setDim(2);
    element.setE(young);
    element.setNu(nu);
    element.setState(
        FracStressCahnHilliardElement::StressState::PlaneStress);
    element.setOmega(omega);
    element.setDiffusivity(2.0);
    element.setCref(cref);
    element.setChi(chi);
    element.setKappa(0.01);
    element.setRho(0.0);
    element.setG0(1.0e-8);
    element.setDamageOn(true);
    element.setTensionOnly(true);

    VectorXd swollen(mesh.getNodesNum()*4,0.0);
    const double mu=regularSolutionDerivative(concentration,chi);
    for (int node=1;node<=mesh.getNodesNum();++node) {
        const int base=(node-1)*4;
        swollen(base+1)=concentration;
        swollen(base+2)=mu;
        swollen(base+3)=
            expansion*mesh.getIthNodeJthCoord(node,1);
        swollen(base+4)=
            expansion*mesh.getIthNodeJthCoord(node,2);
    }

    LocalElmtInfo info;
    info.Dt=0.1;
    info.Timestep=1;
    element.preprocessIteration(
        mesh,operators,info,4,swollen,swollen);
    info.Timestep=2;
    element.preprocessIteration(
        mesh,operators,info,4,swollen,swollen);

    std::vector<double> damage(1,0.0);
    element.computeNodalProjection(
        "damage",mesh,operators,swollen,5,4,damage);
    expectNear("stress-free swelling does not fracture",damage[0],0.0);

    operators.calcAMatrix(5,mesh.getDataConstRef());
    std::vector<double> stress(9,0.0);
    element.computeNodalProjection(
        "stress",mesh,operators,swollen,5,4,stress);
    expectNear("stress-free swelling sigma_xx",stress[0],0.0);
    expectNear("stress-free swelling sigma_yy",stress[4],0.0);
    expectNear("stress-free swelling sigma_xy",stress[1],0.0);

    VectorXd assembled(4,0.0);
    MatrixXd nodalJacobian(4,4,0.0);
    const VectorXd center=localValues(swollen,5);
    element.computeNodalResidualAndJacobian(
        info,5,center,center,1.0,assembled,nodalJacobian);
    operators.calcAMatrix(5,mesh.getDataConstRef());
    for (const int neighbor : mesh.getIthNodeNeighborNodeIDs(5)) {
        operators.calcOperators(5,neighbor,mesh.getDataConstRef());
        const VectorXd other=localValues(swollen,neighbor);
        VectorXd bondResidual(4,0.0);
        MatrixXd kii(4,4,0.0);
        MatrixXd kij(4,4,0.0);
        element.computeBondResidualAndJacobian(
            operators,info,5,neighbor,center,other,center,other,1.0,
            bondResidual,kii,kij);
        for (int dof=1;dof<=4;++dof) assembled(dof)+=bondResidual(dof);
    }
    expectNear("uniform-state species balance",assembled(1),0.0);
    expectNear("stress-coupled chemical equilibrium",assembled(2),0.0);
    expectNear("uniform-state x equilibrium",assembled(3),0.0);
    expectNear("uniform-state y equilibrium",assembled(4),0.0);

    VectorXd transport(swollen);
    for (int node=1;node<=mesh.getNodesNum();++node) {
        const int base=(node-1)*4;
        const double x=mesh.getIthNodeJthCoord(node,1);
        const double y=mesh.getIthNodeJthCoord(node,2);
        transport(base+1)=0.35+0.02*x-0.01*y;
        transport(base+2)=x*x+0.3*y;
        transport(base+3)=0.0;
        transport(base+4)=0.0;
    }
    element.preprocessIteration(
        mesh,operators,info,4,transport,transport);

    double totalBondSpeciesResidual=0.0;
    for (int node=1;node<=mesh.getNodesNum();++node) {
        operators.calcAMatrix(node,mesh.getDataConstRef());
        const VectorXd uI=localValues(transport,node);
        for (const int neighbor : mesh.getIthNodeNeighborNodeIDs(node)) {
            operators.calcOperators(node,neighbor,mesh.getDataConstRef());
            const VectorXd uJ=localValues(transport,neighbor);
            VectorXd bondResidual(4,0.0);
            MatrixXd kii(4,4,0.0);
            MatrixXd kij(4,4,0.0);
            element.computeBondResidualAndJacobian(
                operators,info,node,neighbor,uI,uJ,uI,uJ,
                mesh.getIthNodeVolume(neighbor),
                bondResidual,kii,kij);
            totalBondSpeciesResidual+=
                mesh.getIthNodeVolume(node)*bondResidual(1);
        }
    }
    expectNear("volume-weighted species conservation",
               totalBondSpeciesResidual,0.0);

    expectTrue("element type",
               element.getElementType()=="frac_stress_cahnhilliard");
    expectTrue("coupled degrees of freedom",
               element.getDofNames()
                   ==std::vector<std::string>({"c","mu","ux","uy"}));
    expectTrue("plane-stress state",element.getStateName()=="plane_stress");
    expectTrue("ghost-dropped conserved fields",
               element.getGhostDropSpeciesDofSlots()
                   ==std::vector<int>({0,1}));

    if (failures!=0) return 1;
    std::printf("Fracture stress Cahn-Hilliard element tests passed\n");
    return 0;
}
