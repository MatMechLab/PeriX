//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Mesh/Circle2DLatticeGenerator.h"
#include "Mesh/MeshData.h"
#include "PDMesh/PDMesh.h"

namespace {
int failures=0;

void expectTrue(const std::string &name,const bool value) {
    if (value) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

void expectNear(const std::string &name,const double actual,
                const double expected,const double tolerance=1.0e-13) {
    const double scale=std::max({1.0,std::fabs(actual),std::fabs(expected)});
    if (std::fabs(actual-expected)<=tolerance*scale) return;
    std::printf("FAIL %-36s actual=% .16e expected=% .16e\n",
                name.c_str(),actual,expected);
    ++failures;
}
}

int main() {
    PDLatticeGeneratorBase::Params parameters;
    parameters.R=1.0;
    parameters.N=4;
    parameters.center={1.0,-2.0,0.0};
    parameters.makeBoundaryGroup=true;
    parameters.boundaryName="outer";

    Circle2DLatticeGenerator generator;
    MeshData meshData;
    expectTrue("circle generation",generator.generate(parameters,meshData));
    expectTrue("three physical groups",meshData.PhyGroupNum==3);

    const auto anchorBulkIt=
        meshData.PhyGroupName2BulkElmtIDVecMap.find("anchornodes");
    expectTrue("anchor bulk group exists",
               anchorBulkIt
                   !=meshData.PhyGroupName2BulkElmtIDVecMap.end());
    if (anchorBulkIt
        !=meshData.PhyGroupName2BulkElmtIDVecMap.end()) {
        expectTrue("innermost shell has three points",
                   anchorBulkIt->second==std::vector<int>({1,2,3}));
    }

    const double innerRadius=0.5*parameters.R/
        static_cast<double>(parameters.N);
    for (int point=1;point<=3;++point) {
        const double dx=
            meshData.BulkElmtCenters[static_cast<std::size_t>(3*(point-1))]
            -parameters.center[0];
        const double dy=
            meshData.BulkElmtCenters[
                static_cast<std::size_t>(3*(point-1)+1)]
            -parameters.center[1];
        expectNear("anchor radius",std::hypot(dx,dy),innerRadius);
    }

    const int bulkPoints=meshData.BulkElmtsNum;
    const std::vector<double> bulkCenters=meshData.BulkElmtCenters;
    const std::vector<double> bulkVolumes=meshData.BulkElmtVolumes;

    PDMesh pdMesh;
    pdMesh.createPDMesh(meshData);
    const auto &pdData=pdMesh.getDataConstRef();
    const auto anchorNodeIt=pdData.PhyNameToNodeIDsMap.find("anchornodes");
    expectTrue("PD anchor group exists",
               anchorNodeIt!=pdData.PhyNameToNodeIDsMap.end());
    if (anchorNodeIt!=pdData.PhyNameToNodeIDsMap.end()) {
        expectTrue("anchor maps to material points only",
                   anchorNodeIt->second==std::vector<int>({1,2,3}));
        for (const int node : anchorNodeIt->second) {
            expectTrue("anchor is not a ghost",
                       pdData.GhostMirrorBulkID[
                           static_cast<std::size_t>(node-1)]==0);
        }
    }
    expectTrue("bulk point count unchanged",
               pdMesh.getBulkElmtsNum()==bulkPoints);
    expectTrue("bulk centers unchanged",
               meshData.BulkElmtCenters==bulkCenters);
    expectTrue("bulk volumes unchanged",
               meshData.BulkElmtVolumes==bulkVolumes);
    expectTrue("outer ghost group retained",
               pdData.PhyNameToNodeIDsMap.contains("outernodes_ghost"));

    if (failures!=0) return 1;
    std::printf("Circle anchor-group tests passed\n");
    return 0;
}
