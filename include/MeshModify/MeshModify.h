//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>
#include <vector>

struct MeshData;
class PDMesh;

/**
 * Registers the straight pre-existing cracks used by the published fracture
 * examples and transfers them to PDMesh before neighbor construction.
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

    [[nodiscard]] bool hasOperations() const noexcept {
        return !m_Segments.empty();
    }
    [[nodiscard]] int getCracksNum() const noexcept {
        return static_cast<int>(m_Segments.size());
    }
    [[nodiscard]] int getLastAppliedCracksNum() const noexcept {
        return m_LastApplied;
    }
    [[nodiscard]] const std::vector<CrackSegment>& getCrackSegments() const noexcept {
        return m_Segments;
    }

    void apply(MeshData &meshData,PDMesh &pdMesh);
    void printMeshModifyInfo() const;

private:
    std::vector<CrackSegment> m_Segments;
    CrackTreatment m_Treatment=CrackTreatment::ForceOnly;
    int m_LastApplied=0;
};
