//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cstdio>
#include <string>

#include "Mesh/MeshData.h"
#include "MeshModify/MeshModify.h"
#include "PDMesh/PDMesh.h"

namespace {
int failures=0;

void expect(const char *name,const bool condition) {
    if (condition) return;
    std::printf("FAIL %s\n",name);
    ++failures;
}
}

int main() {
    MeshModify modifier;
    expect("empty modifier",!modifier.hasOperations());
    expect("force-only default",
           modifier.getCrackTreatment()==MeshModify::CrackTreatment::ForceOnly);

    modifier.addCrackSegment(0.5,0.0,0.5,1.0,"vertical_notch");
    modifier.addCrackSegment(0.0,0.8,0.25,0.8);
    expect("two registered segments",modifier.getCracksNum()==2);
    expect("label retained",
           modifier.getCrackSegments().front().label=="vertical_notch");

    MeshData mesh;
    mesh.MeshDim=2;
    PDMesh pdMesh;
    modifier.apply(mesh,pdMesh);

    expect("force-only transferred",pdMesh.getForceOnlyCracks());
    expect("two applied segments",modifier.getLastAppliedCracksNum()==2);
    expect("two PD crack traces",pdMesh.getDataConstRef().Cracks.size()==2);

    PDMeshData &data=pdMesh.getDataRef();
    data.NodesNum=4;
    data.NodeCoords={
        0.25,0.50,0.0,
        0.75,0.50,0.0,
        0.25,0.25,0.0,
        0.25,0.75,0.0
    };
    expect("crossing bond identified",pdMesh.initialCrackCutsBond(1,2));
    expect("same-side bond retained",!pdMesh.initialCrackCutsBond(3,4));

    modifier.clear();
    expect("clear removes operations",!modifier.hasOperations());
    expect("clear resets applied count",modifier.getLastAppliedCracksNum()==0);

    MeshModify throughThickness;
    throughThickness.addCrackSegment(0.5,0.0,0.5,1.0,"3d_notch");
    mesh.MeshDim=3;
    PDMesh pdMesh3d;
    throughThickness.apply(mesh,pdMesh3d);
    PDMeshData &data3d=pdMesh3d.getDataRef();
    data3d.NodesNum=2;
    data3d.NodeCoords={
        0.25,0.50,-0.5,
        0.75,0.50, 0.5
    };
    expect("segment is through-thickness in 3D",
           pdMesh3d.initialCrackCutsBond(1,2));

    if (failures==0) {
        std::printf("MeshModify tests passed\n");
        return 0;
    }
    std::printf("MeshModify tests failed: %d\n",failures);
    return 1;
}
