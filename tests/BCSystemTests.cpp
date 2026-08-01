//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "BCSystem/BCSystem.h"
#include "BCSystem/DirichletBC.h"
#include "BCSystem/NeumannBC.h"
#include "BCSystem/PDTractionBC.h"
#include "BCSystem/SpeciesFluxBC.h"
#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"

namespace {
int failures=0;

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-11) {
    const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
    if (std::fabs(actual-expected)<=tolerance*scale) return;
    std::printf("FAIL %-42s actual=% .16e expected=% .16e\n",
                name.c_str(),actual,expected);
    ++failures;
}

void expectTrue(const std::string &name,const bool value) {
    if (value) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

double matrixValue(const SparseMatrix &matrix,const int row,const int column) {
    const auto rows=matrix.getCSRRowsIndexCopy();
    const auto columns=matrix.getCSRColsIndexCopy();
    const auto values=matrix.getCSRValuesCopy();
    for (int k=rows[static_cast<std::size_t>(row-1)];
         k<rows[static_cast<std::size_t>(row)];++k) {
        if (columns[static_cast<std::size_t>(k)]==column-1) {
            return values[static_cast<std::size_t>(k)];
        }
    }
    return 0.0;
}

void seedDiagonal(SparseMatrix &matrix,const double value) {
    matrix.setToZeros();
    for (int row=1;row<=matrix.getSize();++row) {
        matrix.insertValue(row,row,value);
    }
}

PDMesh makeMirrorMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.DZ=0.0;
    data.NodesNum=2;
    data.BulkElmtsNum=1;
    data.NodesNeighNodesID={{2},{1}};
    data.NodesElmtID={1,1};
    data.GhostMirrorBulkID={0,1};
    data.NodeCoords={0.0,0.0,0.0, 1.0,0.0,0.0};
    data.NodeVolumes={1.0,1.0};
    data.NodeNormal={{0.0,0.0,0.0},{1.0,0.0,0.0}};
    data.PhyNameToNodeIDsMap["wallnodes_ghost"]={2};
    return mesh;
}

PDMesh makeFluxMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.DZ=0.0;
    data.NodesNum=3;
    data.BulkElmtsNum=1;
    data.NodesNeighNodesID={{2,3},{1,3},{1,2}};
    data.NodesElmtID={1,1,1};
    data.GhostMirrorBulkID={0,1,1};
    data.GhostLayerIndex={0,1,2};
    data.GhostFluxThickness={0.0,0.5,0.0};
    data.NodeCoords={
        0.0,0.0,0.0,
        1.0,0.0,0.0,
        2.0,0.0,0.0
    };
    data.NodeVolumes={1.0,1.0,1.0};
    data.NodeNormal={
        {0.0,0.0,0.0},
        {1.0,0.0,0.0},
        {1.0,0.0,0.0}
    };
    data.PhyNameToNodeIDsMap["wallnodes_ghost"]={2,3};
    return mesh;
}

PDMesh makeTractionMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=1.0;
    data.DZ=0.0;
    data.NodesNum=4;
    data.BulkElmtsNum=3;
    data.NodesNeighNodesID={
        {2,3,4},
        {1,3,4},
        {1,2,4},
        {1,2,3}
    };
    data.NodesElmtID={1,2,3,1};
    data.GhostMirrorBulkID={0,0,0,1};
    data.NodeCoords={
        0.0, 0.0,0.0,
        0.0, 1.0,0.0,
        0.0,-1.0,0.0,
        1.0, 0.0,0.0
    };
    data.NodeVolumes={1.0,1.0,1.0,1.0};
    data.NodeNormal={
        {0.0,0.0,0.0},
        {0.0,0.0,0.0},
        {0.0,0.0,0.0},
        {1.0,0.0,0.0}
    };
    data.HorizonRadius=2.0;
    data.VolumeCorrection=false;
    data.PhyNameToNodeIDsMap["rightnodes_ghost"]={4};
    return mesh;
}

PDOperators makeFirstOrderOperators() {
    PDOperators operators;
    operators.setDim(2);
    operators.setOrder(1);
    operators.setup();
    return operators;
}

void testDirichletReflectionAndDirect() {
    PDMesh mesh=makeMirrorMesh();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),1);
    VectorXd rhs(2,0.0);
    VectorXd solution(2,0.0);
    solution(1)=2.0;

    DirichletBC reflected(5.0);
    reflected.presetSolution(mesh,{2},1,solution);
    expectNear("Dirichlet reflected preset",solution(2),8.0);

    solution(2)=7.0;
    seedDiagonal(matrix,4.0);
    reflected.apply(mesh,{2},1,solution,matrix,rhs);
    expectNear("Dirichlet reflected diagonal",matrixValue(matrix,2,2),4.0);
    expectNear("Dirichlet reflected bulk coefficient",
               matrixValue(matrix,2,1),4.0);
    expectNear("Dirichlet reflected RHS",rhs(2),4.0);

    std::vector<char> mask(2,0);
    reflected.presetControlledRows(mesh,{2},1,mask);
    expectTrue("Dirichlet preset mask",mask[1]==1);

    DirichletBC direct(3.0);
    direct.setDirect(true);
    solution(2)=0.0;
    direct.presetSolution(mesh,{2},1,solution);
    expectNear("Dirichlet direct preset",solution(2),3.0);

    solution(2)=1.0;
    rhs.setToZeros();
    seedDiagonal(matrix,4.0);
    direct.apply(mesh,{2},1,solution,matrix,rhs);
    expectNear("Dirichlet direct bulk coefficient",
               matrixValue(matrix,2,1),0.0);
    expectNear("Dirichlet direct RHS",rhs(2),8.0);
}

void testDirichletVelocityBoxAndBulkPin() {
    PDMesh mesh=makeMirrorMesh();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),1);
    VectorXd rhs(2,0.0);
    VectorXd solution(2,0.0);

    // g(t)=value+velocity*t ramps the reflected wall value in time.
    DirichletBC ramped(2.0);
    ramped.setVelocity({3.0});
    solution(1)=1.0;
    ramped.presetSolutionAtTime(mesh,{2},1,solution,0.5);
    expectNear("Dirichlet ramped preset",solution(2),6.0);

    solution(2)=5.0;
    seedDiagonal(matrix,4.0);
    ramped.applyAtTime(mesh,{2},1,solution,matrix,rhs,0.5);
    expectNear("Dirichlet ramped bulk coefficient",
               matrixValue(matrix,2,1),4.0);
    expectNear("Dirichlet ramped RHS",rhs(2),4.0);

    // At t=0 the ramp reduces to the plain value.
    solution(1)=1.0;
    ramped.presetSolution(mesh,{2},1,solution);
    expectNear("Dirichlet ramp at rest",solution(2),3.0);

    // A listed bulk node has no mirror ghost and is pinned directly.
    DirichletBC pin(3.0);
    solution(1)=1.0;
    pin.presetSolution(mesh,{1},1,solution);
    expectNear("Dirichlet bulk preset pins directly",solution(1),3.0);

    rhs.setToZeros();
    seedDiagonal(matrix,4.0);
    solution(1)=1.0;
    pin.apply(mesh,{1},1,solution,matrix,rhs);
    expectNear("Dirichlet bulk pin off-diagonal",
               matrixValue(matrix,1,2),0.0);
    expectNear("Dirichlet bulk pin RHS",rhs(1),8.0);

    // The coordinate box restricts the condition to the covered nodes.
    DirichletBC boxed(9.0);
    boxed.setBox({0.5,1.5,-1.0,1.0,1.0,-1.0});
    solution(1)=1.0;
    solution(2)=0.0;
    boxed.presetSolution(mesh,{1,2},1,solution);
    expectNear("Dirichlet box skips outside node",solution(1),1.0);
    expectNear("Dirichlet box reflects inside node",solution(2),17.0);

    std::vector<char> mask(2,0);
    boxed.presetControlledRows(mesh,{1,2},1,mask);
    expectTrue("Dirichlet box mask",mask[0]==0 && mask[1]==1);
}

void testHomogeneousMirror() {
    PDMesh mesh=makeMirrorMesh();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),1);
    VectorXd rhs(2,0.0);
    VectorXd solution(2,0.0);
    solution(1)=2.0;
    solution(2)=9.0;

    NeumannBC mirror;
    mirror.presetSolution(mesh,{2},1,solution);
    expectNear("Neumann homogeneous preset",solution(2),2.0);

    solution(2)=5.0;
    seedDiagonal(matrix,4.0);
    mirror.apply(mesh,{2},1,solution,matrix,rhs);
    expectNear("Neumann diagonal",matrixValue(matrix,2,2),4.0);
    expectNear("Neumann bulk coefficient",matrixValue(matrix,2,1),-4.0);
    expectNear("Neumann RHS",rhs(2),-12.0);
}

void testSpeciesFluxSourceAndLayerSelection() {
    PDMesh mesh=makeFluxMesh();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),1);
    seedDiagonal(matrix,4.0);
    matrix.insertValue(1,1,-10.0); // K=-dR/dc for a backward-Euler c row
    VectorXd rhs(3,0.0);
    VectorXd solution(3,0.2);

    SpeciesFluxBC flux(2.0);
    flux.apply(mesh,{2,3},1,solution,matrix,rhs);

    // j=2 and V/face thickness=0.5 gives one bulk source -j/t=-4.
    // The deeper ghost must not inject the flux a second time.
    expectNear("Species flux conservative source",rhs(1),-4.0);
    expectTrue("Positive species flux raises concentration",
               rhs(1)/matrixValue(matrix,1,1)>0.0);
    expectNear("Species flux wall-ghost tie",matrixValue(matrix,2,1),-4.0);
    expectNear("Species flux deep-ghost tie",matrixValue(matrix,3,1),-4.0);
    expectTrue("Species flux requires assembled system",
               flux.requiresLinearSystem());
}

void testStrongMechanicalTraction() {
    PDMesh mesh=makeTractionMesh();
    PDOperators operators=makeFirstOrderOperators();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),2);
    seedDiagonal(matrix,4.0);
    VectorXd rhs(8,0.0);
    VectorXd solution(8,0.0);

    constexpr double E=100.0;
    constexpr double nu=0.25;
    constexpr double alpha=0.1;
    constexpr double beta=-0.05;
    const double C11=E/(1.0-nu*nu);
    const double C12=E*nu/(1.0-nu*nu);
    const double sigmaXX=C11*alpha+C12*beta;

    for (int node=1;node<=mesh.getNodesNum();++node) {
        const double x=mesh.getIthNodeJthCoord(node,1);
        const double y=mesh.getIthNodeJthCoord(node,2);
        solution((node-1)*2+1)=alpha*x;
        solution((node-1)*2+2)=beta*y;
    }

    PDTractionBC traction(
        E,nu,PDTractionBC::StressState::PlaneStress,
        {sigmaXX,0.0,0.0});
    traction.applyWithOperators(
        mesh,operators,{1},2,solution,matrix,rhs,0.0);

    expectNear("Strong traction bulk-group ux residual",rhs(7),0.0,1.0e-10);
    expectNear("Strong traction bulk-group uy residual",rhs(8),0.0,1.0e-10);
    expectTrue("Strong traction row is nonzero",
               std::fabs(matrixValue(matrix,7,1))>0.0);
    expectTrue("Strong traction requires assembled system",
               traction.requiresLinearSystem());
}

void testChemicalEigenstressTraction() {
    PDMesh mesh=makeTractionMesh();
    PDOperators operators=makeFirstOrderOperators();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),4);
    seedDiagonal(matrix,4.0);
    VectorXd rhs(16,0.0);
    VectorXd solution(16,0.0);

    for (int node=1;node<=mesh.getNodesNum();++node) {
        solution((node-1)*4+1)=0.2;
    }

    constexpr double E=100.0;
    constexpr double nu=0.25;
    constexpr double Omega=0.3;
    constexpr double cref=0.1;
    const double chemicalStress=E*Omega/(3.0*(1.0-nu));

    PDTractionBC traction(
        E,nu,PDTractionBC::StressState::PlaneStress,
        {0.0,0.0,0.0});
    traction.setDisplacementComponents({3,4});
    traction.setChemicalExpansion(Omega,cref,1);
    traction.applyWithOperators(
        mesh,operators,{4},4,solution,matrix,rhs,0.0);

    const int uxRow=(4-1)*4+3;
    const int uyRow=(4-1)*4+4;
    const int cColumn=(4-1)*4+1;
    expectNear("Chemical traction ux residual",rhs(uxRow),
               chemicalStress*(0.2-cref),1.0e-10);
    expectNear("Chemical traction uy residual",rhs(uyRow),0.0,1.0e-10);
    expectNear("Chemical traction consistent c coupling",
               matrixValue(matrix,uxRow,cColumn),-chemicalStress,1.0e-10);
}

void testRegistryAndDispatch() {
    PDMesh mesh=makeMirrorMesh();
    SparseMatrix matrix;
    matrix.initFromPDMesh(mesh.getDataConstRef(),1);
    seedDiagonal(matrix,4.0);
    VectorXd rhs(2,0.0);
    VectorXd solution(2,0.0);
    VectorXd oldSolution(2,0.0);
    solution(1)=2.0;
    solution(2)=9.0;
    oldSolution=solution;
    PDOperators operators=makeFirstOrderOperators();
    LocalElmtInfo info;

    BCSystem system;
    system.addBC(
        "sealed","wallnodes_ghost",std::make_unique<NeumannBC>());
    expectTrue("BC registry lookup",system.hasBC("sealed"));
    expectTrue("BC registry count",system.getBCsNum()==1);

    system.presetSolution(mesh,1,solution);
    expectNear("BC registry preset dispatch",solution(2),2.0);
    const auto mask=system.collectPresetDofMask(mesh,1);
    expectTrue("BC registry mask dispatch",
               mask.size()==2 && mask[0]==0 && mask[1]==1);

    solution(2)=5.0;
    system.applyBCs(
        mesh,operators,1,solution,oldSolution,info,matrix,rhs);
    expectNear("BC registry assembled dispatch",rhs(2),-12.0);
    expectTrue("Mirror BC is matrix-free compatible",
               system.getLinearSystemOnlyBCs().empty());

    BCSystem sourceSystem;
    sourceSystem.addBC(
        "influx","wallnodes_ghost",std::make_unique<SpeciesFluxBC>(1.0));
    const auto assembledOnly=sourceSystem.getLinearSystemOnlyBCs();
    expectTrue("Source BC assembled-only listing",
               assembledOnly.size()==1
               && assembledOnly.front()=="influx (speciesflux)");
}
}

int main() {
    testDirichletReflectionAndDirect();
    testDirichletVelocityBoxAndBulkPin();
    testHomogeneousMirror();
    testSpeciesFluxSourceAndLayerSelection();
    testStrongMechanicalTraction();
    testChemicalEigenstressTraction();
    testRegistryAndDispatch();

    if (failures==0) {
        std::puts("BCSystemTests: PASS");
        return 0;
    }
    std::printf("BCSystemTests: %d failure(s)\n",failures);
    return 1;
}
