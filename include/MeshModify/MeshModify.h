//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <string>
#include <vector>

struct MeshData;
class PDMesh;

/**
 * Registers the pre-existing cracks used by the published fracture examples
 * and transfers them to PDMesh before neighbor construction.
 *
 * Cracks can be given explicitly (2D segments through "Cracks") or through
 * geometric presets ("Presets") that are resolved against the mesh bounding
 * box. In 3D a preset becomes a bounded planar quadrilateral, so the same
 * deck description works for a plate and for its extruded counterpart.
 */
class MeshModify {
public:
    enum class CrackTreatment {
        ForceOnly
    };

    struct CrackSegment {
        double x1=0.0;
        double y1=0.0;
        double x2=0.0;
        double y2=0.0;
        std::string label;
    };

    /** A resolved 3D crack plane: the planar quadrilateral c0->c1->c2->c3
     *  (looping order) handed to PDMesh::addCrackPlane. */
    struct CrackPlaneSpec {
        std::array<std::array<double,3>,4> corners{};
        std::string label;
    };

    /** Crack centred on a point. In 2D it is the segment of the given length
     *  and angle through (centerX,centerY). In 3D, when none of the optional
     *  orientation fields is set, that same segment is extruded through the
     *  full thickness; otherwise a bounded oriented rectangle is built from
     *  center + normal + length (primary in-plane axis) + width. */
    struct CenterCrackPreset {
        double centerX=0.0;
        double centerY=0.0;
        double length=0.0;
        double angleRadians=0.0;
        std::string label;
        bool hasCenterZ=false;
        double centerZ=0.0;
        bool hasNormal=false;
        std::array<double,3> normal{0.0,1.0,0.0};
        bool hasAxis=false;
        std::array<double,3> axis{1.0,0.0,0.0};
        bool hasWidth=false;
        double width=0.0;
    };

    /** Crack starting on one wall of the mesh bounding box. In 3D the segment
     *  is extruded over z; width>0 limits it to a band centred on the mid
     *  plane, otherwise it spans the full thickness. */
    struct EdgeCrackPreset {
        std::string side;
        double position=0.0;
        double length=0.0;
        bool hasAngle=false;
        double angleRadians=0.0;
        std::string label;
        bool hasWidth=false;
        double width=0.0;
    };

    MeshModify()=default;

    void clear() noexcept;
    void setCrackTreatment(CrackTreatment treatment) noexcept {
        m_Treatment=treatment;
    }
    [[nodiscard]] CrackTreatment getCrackTreatment() const noexcept {
        return m_Treatment;
    }

    void addCrackSegment(double x1,double y1,double x2,double y2,
                         const std::string &name={});
    void addCenterCrackPreset(const CenterCrackPreset &preset);
    void addEdgeCrackPreset(const EdgeCrackPreset &preset);

    [[nodiscard]] bool hasOperations() const noexcept {
        return !m_Segments.empty() || !m_CenterPresets.empty()
            || !m_EdgePresets.empty();
    }
    [[nodiscard]] int getCracksNum() const noexcept {
        return static_cast<int>(m_Segments.size());
    }
    [[nodiscard]] int getCenterPresetsNum() const noexcept {
        return static_cast<int>(m_CenterPresets.size());
    }
    [[nodiscard]] int getEdgePresetsNum() const noexcept {
        return static_cast<int>(m_EdgePresets.size());
    }
    [[nodiscard]] int getLastAppliedCracksNum() const noexcept {
        return m_LastApplied;
    }
    [[nodiscard]] int getLastAppliedPlanesNum() const noexcept {
        return static_cast<int>(m_LastAppliedPlanes.size());
    }
    [[nodiscard]] const std::vector<CrackSegment>& getCrackSegments() const noexcept {
        return m_Segments;
    }
    [[nodiscard]] const std::vector<CrackSegment>& getLastAppliedCracks() const noexcept {
        return m_LastAppliedCracks;
    }
    [[nodiscard]] const std::vector<CrackPlaneSpec>& getLastAppliedPlanes() const noexcept {
        return m_LastAppliedPlanes;
    }

    void apply(MeshData &meshData,PDMesh &pdMesh);
    void printMeshModifyInfo() const;

private:
    [[nodiscard]] CrackSegment resolveCenterPreset(
        const CenterCrackPreset &preset) const;
    [[nodiscard]] CrackSegment resolveEdgePreset(
        const EdgeCrackPreset &preset,const MeshData &meshData) const;
    [[nodiscard]] CrackPlaneSpec resolveCenterPreset3D(
        const CenterCrackPreset &preset,const MeshData &meshData) const;
    [[nodiscard]] CrackPlaneSpec resolveEdgePreset3D(
        const EdgeCrackPreset &preset,const MeshData &meshData) const;
    void validateSegment(const CrackSegment &segment,
                         const MeshData &meshData,int index) const;
    void validatePlane(const CrackPlaneSpec &plane,
                       const MeshData &meshData,int index) const;

private:
    std::vector<CrackSegment> m_Segments;
    std::vector<CenterCrackPreset> m_CenterPresets;
    std::vector<EdgeCrackPreset> m_EdgePresets;
    std::vector<CrackSegment> m_LastAppliedCracks;
    std::vector<CrackPlaneSpec> m_LastAppliedPlanes;
    CrackTreatment m_Treatment=CrackTreatment::ForceOnly;
    int m_LastApplied=0;
};
