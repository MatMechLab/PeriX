//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cmath>
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

    // --- geometric presets -------------------------------------------------
    MeshData plate;
    plate.MeshDim=2;
    plate.Xmin=0.0; plate.Xmax=1.0;
    plate.Ymin=0.0; plate.Ymax=1.0;

    MeshModify centered;
    MeshModify::CenterCrackPreset center;
    center.centerX=0.5;
    center.centerY=0.5;
    center.length=0.4;
    center.angleRadians=0.0;
    center.label="center";
    centered.addCenterCrackPreset(center);
    expect("preset counts as an operation",centered.hasOperations());
    expect("center preset registered",centered.getCenterPresetsNum()==1);

    PDMesh centeredMesh;
    centered.apply(plate,centeredMesh);
    expect("center preset resolves to one segment",
           centered.getLastAppliedCracksNum()==1);
    const auto &centerSeg=centered.getLastAppliedCracks().front();
    expect("center preset x1",std::abs(centerSeg.x1-0.3)<1.0e-12);
    expect("center preset x2",std::abs(centerSeg.x2-0.7)<1.0e-12);
    expect("center preset stays on the mid line",
           std::abs(centerSeg.y1-0.5)<1.0e-12
           && std::abs(centerSeg.y2-0.5)<1.0e-12);
    expect("center preset makes no 3D plane on a 2D mesh",
           centered.getLastAppliedPlanesNum()==0);

    MeshModify edged;
    MeshModify::EdgeCrackPreset edge;
    edge.side="left";
    edge.position=0.5;
    edge.length=0.3;
    edge.label="edge";
    edged.addEdgeCrackPreset(edge);
    PDMesh edgedMesh;
    edged.apply(plate,edgedMesh);
    expect("edge preset resolves to one segment",
           edged.getLastAppliedCracksNum()==1);
    const auto &edgeSeg=edged.getLastAppliedCracks().front();
    expect("edge preset starts on the left wall",
           std::abs(edgeSeg.x1-0.0)<1.0e-12
           && std::abs(edgeSeg.y1-0.5)<1.0e-12);
    expect("edge preset grows inward by default",
           std::abs(edgeSeg.x2-0.3)<1.0e-12
           && std::abs(edgeSeg.y2-0.5)<1.0e-12);

    // On a 3D mesh the same center preset becomes a bounded crack plane.
    MeshData block;
    block.MeshDim=3;
    block.Xmin=0.0; block.Xmax=1.0;
    block.Ymin=0.0; block.Ymax=1.0;
    block.Zmin=0.0; block.Zmax=0.2;

    MeshModify orientedPlane;
    MeshModify::CenterCrackPreset oriented;
    oriented.centerX=0.5;
    oriented.centerY=0.5;
    oriented.hasCenterZ=true;
    oriented.centerZ=0.1;
    oriented.length=0.4;
    oriented.hasWidth=true;
    oriented.width=0.2;
    oriented.hasNormal=true;
    oriented.normal={1.0,0.0,0.0};
    oriented.label="oriented";
    orientedPlane.addCenterCrackPreset(oriented);
    PDMesh orientedMesh;
    orientedPlane.apply(block,orientedMesh);
    expect("oriented preset resolves to one plane",
           orientedPlane.getLastAppliedPlanesNum()==1);
    expect("oriented preset adds no 2D segment",
           orientedPlane.getLastAppliedCracks().empty());
    const auto &plane=orientedPlane.getLastAppliedPlanes().front();
    bool planeInNormalPlane=true;
    for (const auto &corner:plane.corners) {
        // normal = +x means every corner keeps x = centerX
        if (std::abs(corner[0]-0.5)>1.0e-12) planeInNormalPlane=false;
    }
    expect("oriented plane is normal to x",planeInNormalPlane);
    expect("oriented plane reached PDMesh",
           orientedMesh.getDataConstRef().CrackPlanes.size()==1);

    // With no 3D orientation fields the preset extrudes through the thickness.
    MeshModify extruded;
    MeshModify::CenterCrackPreset plain;
    plain.centerX=0.5;
    plain.centerY=0.5;
    plain.length=0.4;
    plain.label="extruded";
    extruded.addCenterCrackPreset(plain);
    PDMesh extrudedMesh;
    extruded.apply(block,extrudedMesh);
    expect("plain preset extrudes to a plane",
           extruded.getLastAppliedPlanesNum()==1);
    const auto &through=extruded.getLastAppliedPlanes().front();
    expect("extrusion spans the full thickness",
           std::abs(through.corners[0][2]-0.0)<1.0e-12
           && std::abs(through.corners[2][2]-0.2)<1.0e-12);

    MeshModify mixed;
    mixed.addCrackSegment(0.1,0.1,0.9,0.1,"explicit");
    mixed.addCenterCrackPreset(center);
    PDMesh mixedMesh;
    mixed.apply(plate,mixedMesh);
    expect("explicit cracks and presets combine",
           mixed.getLastAppliedCracksNum()==2);

    if (failures==0) {
        std::printf("MeshModify tests passed\n");
        return 0;
    }
    std::printf("MeshModify tests failed: %d\n",failures);
    return 1;
}
