//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "MeshModify/MeshModify.h"

#include <cmath>

#include "Mesh/MeshData.h"
#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

namespace {
void stopWithError(const std::string &message) {
    MessagePrinter::printErrorTxt(message);
    MessagePrinter::exitPeriX();
}
}

void MeshModify::clear() noexcept {
    m_Segments.clear();
    m_Treatment=CrackTreatment::ForceOnly;
    m_LastApplied=0;
}

void MeshModify::addCrackSegment(const double x1,const double y1,
                                 const double x2,const double y2,
                                 const std::string &name) {
    m_Segments.push_back({x1,y1,x2,y2,name});
}

void MeshModify::apply(MeshData &meshData,
                       PDMesh &pdMesh) {
    m_LastApplied=0;
    if (m_Segments.empty()) return;

    if (meshData.MeshDim!=2 && meshData.MeshDim!=3) {
        stopWithError("MeshModify: pre-existing cracks require a 2D or 3D mesh");
        return;
    }
    const auto &existingFamilies=pdMesh.getDataConstRef().NodesNeighNodesID;
    if (!existingFamilies.empty()) {
        stopWithError("MeshModify: cracks must be registered before neighbor construction");
        return;
    }

    for (std::size_t i=0;i<m_Segments.size();++i) {
        const CrackSegment &segment=m_Segments[i];
        const bool finite=std::isfinite(segment.x1) && std::isfinite(segment.y1)
            && std::isfinite(segment.x2) && std::isfinite(segment.y2);
        const double deltaX=segment.x2-segment.x1;
        const double deltaY=segment.y2-segment.y1;
        if (!finite || deltaX*deltaX+deltaY*deltaY<=0.0) {
            stopWithError("MeshModify: invalid crack segment "
                          +std::to_string(i+1));
            return;
        }
    }

    pdMesh.clearCracks();
    pdMesh.setForceOnlyCracks(true);
    for (const CrackSegment &segment:m_Segments) {
        pdMesh.addCrack(segment.x1,segment.y1,
                        segment.x2,segment.y2);
    }
    m_LastApplied=static_cast<int>(m_Segments.size());
}

void MeshModify::printMeshModifyInfo() const
{
    if (m_Segments.empty()) return;

    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Mesh modification");
    MessagePrinter::printNormalTxt("  type: pre_existing_cracks");
    MessagePrinter::printNormalTxt("  treatment: force_only");
    MessagePrinter::printNormalTxt("  crack segments: "
                                   +std::to_string(m_Segments.size()));
    MessagePrinter::printStars();
}
