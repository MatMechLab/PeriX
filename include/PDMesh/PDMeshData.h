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
//+++ Function: the pd mesh or pd point data structure
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

/**
 * data structure for the pd point
 */
struct PDMeshData {
    /**
     * Relative horizon INCLUSION MARGIN, folded into HorizonRadius (and hence
     * into every per-node delta_i) at the single place the radius is set.
     *
     * The PD family is the closed ball |xi| <= delta, so a bond landing EXACTLY
     * on the horizon must be a member. That case is common by construction:
     * factor-m horizons on a uniform lattice put the axis-aligned m-cell bonds
     * at exactly m*dx. Whether such a bond passes an "|xi| <= delta" test is
     * then decided by round-off -- and for an IMPORTED mesh the noise is not
     * machine eps but the mesh-FILE precision: cell volumes parsed from a gmsh
     * file wiggle at ~1e-12 relative (single-precision files: ~1e-7), so under
     * VariableHorizon the per-node delta_i inherit that wiggle and the on-horizon
     * bonds flip in/out PSEUDO-RANDOMLY node to node. Observed as a visible
     * speckle in the solution field along a uniformly-loaded boundary (family
     * sizes, and so the PDDO operators, differ node to node); with a uniform
     * horizon the same noise enters globally through the median cell size and
     * silently flips the WHOLE m-cell ring in or out depending on the mesh file.
     *
     * The margin makes the closed-ball test robust: delta carries +1e-6
     * relative, which dominates any double/single-precision file round-off
     * while sitting far below a meaningful mesh grading step (per-cell volume
     * ratios in a genuinely graded mesh differ by >~1e-3). Applying it at the
     * radius SOURCE (PDMesh::setHorizonRadius*) instead of at each comparison
     * fixes every consumer at once -- neighbour-list build and PDDO operator
     * family cuts (CPU and CUDA) -- and keeps CPU/GPU parity by construction,
     * since both read the same
     * stored values. The 1e-6 shift of the weight/volume-correction arguments
     * is physically invisible.
     */
    static constexpr double HorizonInclusionMargin=1.0e-6;

    double DX,DY,DZ;/**< grid size of x/y/z-axis */

    int NodesNum;/**< the total nodes num */
    int BulkElmtsNum;/**< the bulk elements num */
    vector<vector<int>> NodesNeighNodesID;/**< the neighboring nodes id of each PD node */
    vector<int> NodesElmtID;/**< pd node's parent element id */
    vector<int> GhostMirrorBulkID;/**< for each ghost node (1-based), the bulk PD node id it mirrors across the boundary; 0 for bulks or unmapped ghosts */

    /**
     * Per (boundary) ghost node (1-based, size NodesNum when populated), the
     * CONSERVATIVE flux-smearing thickness  t_g = V_bulk / face_measure  of the
     * boundary face that spawned the wall-adjacent ghost: V_bulk is the mirror
     * bulk cell's measure (area in 2D, volume in 3D) and face_measure is the
     * boundary element's measure (length in 2D, area in 3D). A surface flux
     * density j applied at the ghost is turned into the volumetric source
     * S = j / t_g on the mirror bulk row, so the discrete injection
     *     sum_faces S * V_bulk = j * sum_faces face_measure = j * Area
     * recovers the surface integral EXACTLY on any cell shape. This is the only
     * thickness for which that holds on a triangle/tet mesh: there the cell
     * centroid sits at 1/dim of the cell height from the face, so the purely
     * geometric ghost<->bulk distance |x_g - x_bulk| = 2*signedDistance
     * over/under-smears (e.g. 2H/3 instead of H/2 on a triangle => a uniform
     * 25% flux error). 0 for bulk nodes, deeper-layer ghosts, and ghosts with no
     * face measure; EMPTY when no boundary face measure was available (the flux
     * BC then falls back to the geometric |x_g - x_bulk| thickness, which is
     * exact for an axis-aligned quad/hex background mesh). See SpeciesFluxBC.
     */
    vector<double> GhostFluxThickness;

    /**
     * Number of mirror ghost layers spawned on each boundary wall (PDMesh
     * "ghost_layer", default 1). Layer 1 is the single half-cell ghost the
     * mesh always had (the exact mirror of the adjacent bulk cell). With
     * ghost_layer = N > 1 each boundary face additionally mirrors the next
     * N-1 bulk cells inward along its normal, so a near-boundary bulk node's
     * PD family is filled out to the full horizon on the OUTSIDE of the wall
     * instead of staying one-sided. This restores operator support near a
     * symmetry or reflection wall. Default 1 preserves the single-ring mesh.
     */
    int GhostLayer=1;

    /**
     * Per-boundary-group override of GhostLayer (PDMesh "ghost_layer" given as an
     * object {"<group>": N}); a group not listed uses the global GhostLayer.
     * Multi-layer ghosts belong on REFLECTION walls -- a symmetry plane (Dirichlet
     * on the normal component + zero-flux/Neumann on the rest), or any wall whose
     * BCs reflect the field. Curved imported boundaries remain single-layer
     * because a unique reflection plane is not available.
     * For imported meshes, a group with non-collinear boundary normals is
     * automatically clamped to one ghost layer because the deeper-layer algorithm
     * assumes a single reflection plane per physical group.
     */
    unordered_map<string,int> GhostLayerGroups;

    /**
     * Per PD node (1-based, size NodesNum when multi-layer ghosts are built),
     * the reflection-layer index of a ghost: 1 for the innermost (wall-adjacent)
     * ghost, 2..GhostLayer for the deeper mirror layers, and 0 for every bulk
     * node. EMPTY when ghost_layer == 1 (so every consumer that checks empty()
     * keeps the original single-layer path). The depth-sensitive SURFACE-load
     * source-form species flux and traction conditions act only on the
     * innermost layer (index <= 1) -- a surface load belongs to the wall
     * cell once, not once per mirror layer -- while every layer still
     * mirror-reflects its own bulk so the family stays complete.
     */
    vector<int> GhostLayerIndex;

    /**
     * Outward unit normal at every PD node (1-based, size NodesNum). For a
     * boundary GHOST node it is the outward normal of the wall it images,
     * which is IDENTICAL to the outward normal of the (d-1)-D boundary finite
     * element that spawned the ghost: an axis-aligned wall gives +/-e_axis and
     * an imported wall gives the sign-fixed MeshImport::boundaryNormal(). It is
     * computed once, at mesh-build time (PDMesh::computeNodeNormals, called from
     * createPDMesh), as
     *     n_g = (x_ghost - x_mirrorbulk)/|x_ghost - x_mirrorbulk|
     * and this equals that boundary-element normal because the ghost is placed
     * as the EXACT planar mirror of its adjacent bulk centroid across the wall
     * (see createPDMesh). Computing it with the same arithmetic the natural-BC
     * rows used to do inline makes the cached value bit-for-bit identical to the
     * old on-the-fly (x_ghost-x_bulk) normalisation. BULK nodes and ghosts with
     * no mirror bulk carry {0,0,0}. Consumers (e.g. PDTractionBC's sigma.n row)
     * read this single, inspectable source of truth instead of re-deriving the
     * wall orientation from coordinates.
     */
    vector<array<double,3>> NodeNormal;

    unordered_map<string,vector<int>> PhyNameToNodeIDsMap;/**< physical name to node ids map */

    vector<double> NodeCoords;/**< coordinates of the PD nodes */
    vector<double> NodeVolumes;/**< volume of each pd node */

    double HorizonRadius=0.0;/**< the radius of the pd horizon (uniform/global value) */
    double HorizonRadiusFactor=0.0;/**< HorizonRadiusFactor from input metadata */
    double HorizonBaseDX=0.0;/**< x-cell spacing used to convert HorizonRadiusFactor to HorizonRadius */
    int MaxNeighbors=0;/**< the maximum neighbor nodes */
    int MinNeighbors=0;/**< the minimum neighbor ndoes */

    /**
     * Variable (per-node) horizon for non-uniform / graded imported meshes.
     * OPT-IN via PDMesh.VariableHorizon (default false). When ON, each PD node
     * i carries its own horizon delta_i scaled to its local cell size, so a
     * single global horizon does not under-/over-fill families in a graded mesh.
     * NodeHorizon/NodeSpacing are EMPTY when the feature is off; every consumer
     * (the neighbour-family build and the PDDO operators) checks `empty()` and
     * falls back to the global HorizonRadius / max(DX,DY,DZ), so the uniform
     * path is unchanged.
     */
    bool VariableHorizon=false;/**< opt-in per-node horizon (off => global) */
    vector<double> NodeHorizon;/**< per-node horizon delta_i (empty => use HorizonRadius) */
    vector<double> NodeSpacing;/**< per-node spacing for the partial-volume rim (empty => use max(DX,DY,DZ)) */

    /**
     * partial-volume (rim) correction toggle for the PDDO moment matrix and
     * per-bond operators: when ON (default) a bond whose neighbour cell
     * straddles the horizon edge carries the linear rim factor
     * vc = (delta - |xi| + dx/2)/dx; when OFF every in-horizon bond carries
     * its full cell volume (vc = 1). OFF reproduces discretizations that sum
     * plain cell volumes over |xi| <= delta -- e.g. the reference
     * Kalthoff--Winkler Matlab drivers -- and is selected per run via
     * PDMesh.VolumeCorrection in the input file.
     */
    bool VolumeCorrection=true;

    /**
     * geometry-operator cache toggle (input key PDMesh.op_cache, default true).
     * When true the implicit drivers build the per-bond PDDO operator cache at
     * startup (PDOperators::buildGeometryCache) and replay it on every
     * assembly; the operators depend only on the frozen reference geometry, so
     * the replay is bitwise identical to a live recomputation.
     */
    bool OpCache=true;

    /**
     * pre-existing crack segments (initial slits). Each entry is a
     * 2D line segment {x1, y1, x2, y2} in the same coordinate system
     * as NodeCoords. Bonds whose connecting segment crosses any
     * crack are EXCLUDED from NodesNeighNodesID by
     * createNeighborNodes(), which is the standard peridynamic way
     * to model an initial discontinuity (matches Madenci's slits[]
     * approach in Fortran example 6.10).
     *
     * The crossing test is GHOST-AWARE (PDMesh::initialCrackCutsBond): a
     * bond endpoint that is a ghost PD point is resolved to its mirror
     * bulk (GhostMirrorBulkID) before the segment test, because a ghost is
     * the boundary-condition image of that bulk -- its value is slaved to
     * the bulk's by the BC rows. This severs the ghost-side bonds that
     * would otherwise bridge a wall-touching notch outside the wall, and
     * keeps a flank ghost of an imported (thin-slit) notch bonded to its
     * OWN crack face even when it geometrically lands on the far side.
     */
    vector<array<double,4>> Cracks;

    /**
     * pre-existing 3D crack PLANES (initial planar slits). Each entry is a
     * planar quadrilateral given by its four corners c0,c1,c2,c3 stored as
     * 12 doubles {c0x,c0y,c0z, c1x,c1y,c1z, c2x,c2y,c2z, c3x,c3y,c3z} and
     * ordered around the loop (c0->c1->c2->c3), in the same coordinate
     * system as NodeCoords. A bond whose connecting segment STRICTLY pierces
     * the quad interior is severed, exactly as a `Cracks` segment severs a
     * bond it strictly crosses.
     *
     * Unlike a `Cracks` segment -- which in 3D is only the xy trace of a
     * plane extruded over ALL z (a through-thickness crack whose normal must
     * lie in the xy plane) -- a CrackPlane is a BOUNDED, arbitrarily oriented
     * rectangle. It therefore also models an embedded/part-through crack, a
     * finite-thickness slit, or a plane inclined out of the z direction (e.g.
     * a delamination with a z normal), which the extruded-segment form cannot
     * represent. The SAME ghost-aware endpoint resolution (GhostMirrorBulkID)
     * and the "endpoint touching = non-crossing" convention are applied by
     * initialCrackCutsBond before the pierce test.
     */
    vector<array<double,12>> CrackPlanes;

    /** true iff any pre-existing initial crack (2D segment or 3D plane) is
     *  configured; the single guard the geometric family cut and the
     *  force-only bond-health seeding share. */
    [[nodiscard]] bool hasInitialCracks() const {
        return !Cracks.empty() || !CrackPlanes.empty();
    }

    /**
     * crack treatment. false (default): a crack-crossing bond is DELETED from
     * the family at mesh-build time (the geometric, kinematics-respecting slit
     * used by frac_mechanics). true: the crossing bonds are KEPT in the family
     * -- the discontinuity is enforced later, in the force only, by the element
     * zeroing those bonds' health while still using them to build the shape
     * tensor / deformation gradient. The force-only treatment reproduces the
     * reference Matlab Kalthoff--Winkler driver, whose fixed PDDO operator spans
     * the slit, so the near-tip gradient is "smeared" across the crack and the
     * resulting stress concentration drives crack propagation.
     */
    bool ForceOnlyCracks=false;
};

/**
 * True iff the bond (i,j) is severed by a pre-existing crack (initial slit).
 * The single ghost-aware crossing rule shared by the geometric family cut,
 * the force-only bond-health seeding, and the crack-respecting PDDO operator
 * mode (PDOperators::setRespectInitialCracks): a ghost endpoint is resolved
 * to its mirror bulk before the strict xy segment-crossing test, and the
 * orientation signs carry a relative epsilon so endpoint-touching bonds get
 * a deterministic non-crossing verdict. See PDMesh::initialCrackCutsBond
 * (which forwards here) for the full rationale. i,j are 1-based node ids.
 */
[[nodiscard]] bool initialCrackCutsBond(const PDMeshData &Data,const int &i,const int &j);
