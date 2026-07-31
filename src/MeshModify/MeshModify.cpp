//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "MeshModify/MeshModify.h"

#include <string>

#include "Mesh/MeshData.h"
#include "MeshModify/MeshModifyGeometry.h"
#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

void MeshModify::clear() noexcept {
    m_Segments.clear();
    m_CenterPresets.clear();
    m_EdgePresets.clear();
    m_LastAppliedCracks.clear();
    m_LastAppliedPlanes.clear();
    m_Treatment=CrackTreatment::ForceOnly;
    m_LastApplied=0;
}

void MeshModify::addCrackSegment(const double x1,const double y1,
                                 const double x2,const double y2,
                                 const std::string &name) {
    m_Segments.push_back({x1,y1,x2,y2,name});
}

void MeshModify::addCenterCrackPreset(const CenterCrackPreset &preset) {
    m_CenterPresets.push_back(preset);
}

void MeshModify::addEdgeCrackPreset(const EdgeCrackPreset &preset) {
    m_EdgePresets.push_back(preset);
}

void MeshModify::apply(MeshData &meshData,
                       PDMesh &pdMesh) {
    m_LastApplied=0;
    m_LastAppliedCracks.clear();
    m_LastAppliedPlanes.clear();
    if (!hasOperations()) return;

    if (meshData.MeshDim!=2 && meshData.MeshDim!=3) {
        MessagePrinter::printErrorTxt(
            "MeshModify: pre-existing cracks require a 2D or 3D mesh");
        MessagePrinter::exitPeriX();
    }
    if (!pdMesh.getDataConstRef().NodesNeighNodesID.empty()) {
        MessagePrinter::printErrorTxt(
            "MeshModify: cracks must be registered before neighbor construction");
        MessagePrinter::exitPeriX();
    }

    const bool is3D=(meshData.MeshDim==3);

    // In 3D a center_crack / edge_crack preset resolves to a bounded crack
    // PLANE; in 2D it resolves to a line segment. Explicit "Cracks" entries are
    // always 2D segments (through-thickness on a 3D mesh).
    std::vector<CrackSegment> resolvedSegments=m_Segments;
    std::vector<CrackPlaneSpec> resolvedPlanes;
    resolvedSegments.reserve(m_Segments.size()+m_CenterPresets.size()
                             +m_EdgePresets.size());
    resolvedPlanes.reserve(m_CenterPresets.size()+m_EdgePresets.size());

    for (const auto &preset:m_CenterPresets) {
        if (is3D) resolvedPlanes.push_back(resolveCenterPreset3D(preset,meshData));
        else      resolvedSegments.push_back(resolveCenterPreset(preset));
    }
    for (const auto &preset:m_EdgePresets) {
        if (is3D) resolvedPlanes.push_back(resolveEdgePreset3D(preset,meshData));
        else      resolvedSegments.push_back(resolveEdgePreset(preset,meshData));
    }

    pdMesh.clearCracks();
    pdMesh.setForceOnlyCracks(true);

    int index=0;
    for (const auto &segment:resolvedSegments) {
        index+=1;
        validateSegment(segment,meshData,index);
        pdMesh.addCrack(segment.x1,segment.y1,segment.x2,segment.y2);
    }
    for (const auto &plane:resolvedPlanes) {
        index+=1;
        validatePlane(plane,meshData,index);
        pdMesh.addCrackPlane(plane.corners[0],plane.corners[1],
                             plane.corners[2],plane.corners[3]);
    }

    m_LastAppliedCracks=resolvedSegments;
    m_LastAppliedPlanes=resolvedPlanes;
    m_LastApplied=static_cast<int>(resolvedSegments.size()
                                   +resolvedPlanes.size());
}

void MeshModify::printMeshModifyInfo() const
{
    if (!hasOperations()) return;

    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Mesh modification");
    MessagePrinter::printNormalTxt("  type: pre_existing_cracks");
    MessagePrinter::printNormalTxt("  treatment: force_only");
    MessagePrinter::printNormalTxt("  explicit crack segments: "
                                   +std::to_string(m_Segments.size()));
    if (!m_CenterPresets.empty()) {
        MessagePrinter::printNormalTxt("  center_crack presets: "
                                       +std::to_string(m_CenterPresets.size()));
    }
    if (!m_EdgePresets.empty()) {
        MessagePrinter::printNormalTxt("  edge_crack presets: "
                                       +std::to_string(m_EdgePresets.size()));
    }
    if (m_LastApplied>0) {
        MessagePrinter::printNormalTxt("  applied segments: "
            +std::to_string(m_LastAppliedCracks.size())
            +", applied planes: "
            +std::to_string(m_LastAppliedPlanes.size()));
    }
    MessagePrinter::printStars();
}
