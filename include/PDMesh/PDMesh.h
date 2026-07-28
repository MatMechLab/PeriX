//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Author  : Yang Bai
//+++ Date    : 2026.04.14
//+++ Function: the pd mesh class of PeriX
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "Mesh/MeshData.h"
#include "PDMesh/PDMeshData.h"

class PDMesh {
public:
    PDMesh();
    ~PDMesh();

    // Both radius setters fold in the relative inclusion margin so a bond
    // landing EXACTLY on the horizon (e.g. the axis-aligned m-cell bonds of a
    // factor-m horizon on a uniform lattice) is a family member regardless of
    // mesh-file round-off; see PDMeshData::HorizonInclusionMargin for the full
    // rationale. Every membership test consumes the stored radius, so the
    // margin fixes them consistently at this single source.
    void setHorizonRadius(const double &radius) {
        m_Data.HorizonRadius = radius*(1.0+PDMeshData::HorizonInclusionMargin);
        m_Data.HorizonRadiusFactor = 0.0;
        m_Data.HorizonBaseDX = 0.0;
    }

    void setHorizonRadiusFromFactor(const double &factor,const double &dx) {
        m_Data.HorizonRadiusFactor = factor;
        m_Data.HorizonBaseDX = dx;
        m_Data.HorizonRadius = factor*dx*(1.0+PDMeshData::HorizonInclusionMargin);
    }

    void addCrack(const double &x1,const double &y1,
                  const double &x2,const double &y2) {
        m_Data.Cracks.push_back({x1, y1, x2, y2});
    }

    /** Register a bounded 3D crack plane as the planar quadrilateral with the
     *  four corners c0->c1->c2->c3 (looping order). A bond that strictly
     *  pierces the quad interior is severed by initialCrackCutsBond, the same
     *  ghost-aware rule that cuts 2D segment cracks. */
    void addCrackPlane(const std::array<double,3> &c0,const std::array<double,3> &c1,
                       const std::array<double,3> &c2,const std::array<double,3> &c3) {
        m_Data.CrackPlanes.push_back({c0[0],c0[1],c0[2], c1[0],c1[1],c1[2],
                                      c2[0],c2[1],c2[2], c3[0],c3[1],c3[2]});
    }

    void clearCracks() {
        m_Data.Cracks.clear();
        m_Data.CrackPlanes.clear();
    }

    /** false (default): crack-crossing bonds are deleted from the family;
     *  true: they are kept (force-only crack, enforced by the element). */
    void setForceOnlyCracks(const bool &v) { m_Data.ForceOnlyCracks = v; }
    [[nodiscard]] bool getForceOnlyCracks() const { return m_Data.ForceOnlyCracks; }

    /** opt-in per-node (variable) horizon for non-uniform imported meshes. */
    void setVariableHorizon(const bool &v) { m_Data.VariableHorizon = v; }
    [[nodiscard]] bool getVariableHorizon() const { return m_Data.VariableHorizon; }

    /** number of mirror ghost layers per boundary wall (PDMesh "ghost_layer").
     *  >=1; clamped to 1 below. >1 completes the near-boundary PD family on the
     *  outside of the wall so a symmetry/reflection boundary has adequate
     *  operator support. Default 1 preserves the single-ring mesh. */
    void setGhostLayer(const int &v) { m_Data.GhostLayer = (v<1) ? 1 : v; }
    [[nodiscard]] int getGhostLayer() const { return m_Data.GhostLayer; }
    /** per-boundary-group override of ghost_layer (others use the global value).
     *  Put extra mirror layers on planar symmetry/reflection walls; imported
     *  curved groups are limited to one layer. */
    void setGhostLayerGroup(const std::string &name,const int &v) { m_Data.GhostLayerGroups[name]=(v<1)?1:v; }

    /** partial-volume rim correction in the PDDO operators: true (default)
     *  applies vc=(delta-|xi|+dx/2)/dx in the horizon rim; false sums plain
     *  cell volumes over |xi|<=delta (matches the reference Matlab drivers). */
    void setVolumeCorrection(const bool &v) { m_Data.VolumeCorrection = v; }
    [[nodiscard]] bool getVolumeCorrection() const { return m_Data.VolumeCorrection; }

    /** geometry-operator cache toggle (PDMesh.op_cache, default true): compute
     *  the per-bond PDDO operators once at startup and replay them each call
     *  (they depend only on the frozen reference geometry). */
    void setOpCache(const bool &v) { m_Data.OpCache = v; }
    [[nodiscard]] bool getOpCache() const { return m_Data.OpCache; }

    void createPDMesh(MeshData &t_MeshData);
    /** Fill NodeHorizon/NodeSpacing with the per-node (variable) horizon scaled
     *  to each node's local cell size. Must be called AFTER createPDMesh (needs
     *  NodeVolumes) and BEFORE createNeighborNodes. A no-op unless
     *  VariableHorizon is set. */
    void computeVariableHorizon();
    void createNeighborNodes();
    void savePDMesh(const string &inputfilename);

    /**
     * True iff the bond (i,j) is severed by a pre-existing crack (initial
     * slit). This is THE single crack-crossing rule shared by the geometric
     * family cut (createNeighborNodes, treatment=delete_crossing_bonds) and
     * by the force-only bond-health seeding inside the fracture kernels, so
     * the two treatments always agree on which bonds an initial notch kills.
     *
     * A ghost PD point is the boundary-condition image of its mirror bulk
     * point: every BC row (reflection Dirichlet, mirror Neumann, traction,
     * pdtraction) algebraically slaves u_ghost to u_mirrorbulk, so a bond
     * touching a ghost physically couples the MIRROR positions, not the
     * literal ghost coordinates. Each endpoint is therefore resolved to its
     * mirror bulk (GhostMirrorBulkID) before the strict segment-crossing
     * test. This (a) severs the ghost-ghost / through-the-mouth bonds that
     * bridge a wall-touching notch outside the wall, and (b) keeps a ghost
     * bonded to its OWN side of an imported interior notch even when the
     * slit is thinner than a cell and the ghost geometrically lands on the
     * far side. Bulk-bulk bonds resolve to themselves, so their behavior is
     * unchanged. i and j are 1-based PD node ids.
     */
    [[nodiscard]] bool initialCrackCutsBond(const int &i,const int &j) const;

    int getNodesNum()const {
        return m_Data.NodesNum;
    }

    const vector<int>& getIthNodeNeighborNodeIDs(const int &i)const {
        return m_Data.NodesNeighNodesID[i-1];
    }

    double getIthNodeJthCoord(const int &i,const int &j) const {
        return m_Data.NodeCoords[(i-1)*3+j-1];
    }

    double getIthNodeVolume(const int &i)const {
        return m_Data.NodeVolumes[i-1];
    }

    /** Outward unit normal of node i (1-based). Non-zero only for boundary
     *  ghosts (= the outward normal of the wall they image); {0,0,0} for bulk
     *  nodes and ghosts with no mirror bulk. See PDMeshData::NodeNormal. */
    const std::array<double,3>& getIthNodeNormal(const int &i)const {
        return m_Data.NodeNormal[i-1];
    }

    const vector<int>& getNodeIDsViaPhyName(const string &name)const {
        if (m_Data.PhyNameToNodeIDsMap.find(name) != m_Data.PhyNameToNodeIDsMap.end()) {
            return m_Data.PhyNameToNodeIDsMap.at(name);
        }
        static const std::vector<int> empty;
        return empty;
    }

    const PDMeshData& getDataConstRef()const {
        return m_Data;
    }

    PDMeshData& getDataRef() {
        return m_Data;
    }

    void printPDMeshInfo()const;

private:
    /** Fill NodeNormal with the outward unit normal at every PD node: for each
     *  ghost it is (x_ghost - x_mirrorbulk)/|.| (the outward normal of the wall
     *  it images, equal to the boundary FE element normal by the mirror
     *  construction); bulk nodes and unmapped ghosts get {0,0,0}. Called at the
     *  end of createPDMesh (both the structured and imported branches), AFTER
     *  NodeCoords and GhostMirrorBulkID are final. */
    void computeNodeNormals();

    PDMeshData m_Data;/**< pd mesh data strcuture */
};
