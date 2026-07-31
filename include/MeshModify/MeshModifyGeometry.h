//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <string>

#include "MeshModify/MeshModify.h"

struct MeshData;

/**
 * Small geometry helpers shared by the MeshModify preset resolution and the
 * crack registration itself.
 */
namespace meshmodify {

inline constexpr double kPi=3.14159265358979323846;

using Vec3=std::array<double,3>;

struct MeshBox {
    double xmin=0.0;
    double xmax=0.0;
    double ymin=0.0;
    double ymax=0.0;
    double zmin=0.0;
    double zmax=0.0;
};

[[nodiscard]] Vec3 vsub(const Vec3 &a,const Vec3 &b);
[[nodiscard]] double vdot(const Vec3 &a,const Vec3 &b);
[[nodiscard]] Vec3 vcross(const Vec3 &a,const Vec3 &b);
[[nodiscard]] double vnorm(const Vec3 &a);
[[nodiscard]] Vec3 inPlaneHelper(const Vec3 &n);

/** Mesh bounding box, taken from the node coordinates when they are available
 *  and from the recorded axis extents otherwise. */
[[nodiscard]] MeshBox getMeshBox(const MeshData &meshData);

[[nodiscard]] double orient2D(double ax,double ay,double bx,double by,
                              double cx,double cy);
[[nodiscard]] bool pointInBox(double x,double y,const MeshBox &box,double tol);

/** False when the mesh extents have not been populated yet (a degenerate box):
 *  the overlap checks then have nothing meaningful to test against. */
[[nodiscard]] bool isUsableBox(const MeshBox &box);
[[nodiscard]] bool onSegment(double ax,double ay,double bx,double by,
                             double px,double py,double tol);
[[nodiscard]] bool closedSegmentsIntersect(double ax,double ay,
                                           double bx,double by,
                                           double cx,double cy,
                                           double dx,double dy,double tol);
[[nodiscard]] bool segmentOverlapsBox(const MeshModify::CrackSegment &s,
                                      const MeshBox &box);
[[nodiscard]] bool planeOverlapsBox(const MeshModify::CrackPlaneSpec &s,
                                    const MeshBox &box);

/** Direction an edge crack grows in when the deck does not give an angle:
 *  inward, normal to the wall it starts from. */
[[nodiscard]] double defaultEdgeAngle(const std::string &side);

[[nodiscard]] MeshModify::CrackPlaneSpec extrudeSegment(
    double x1,double y1,double x2,double y2,
    double zlo,double zhi,const std::string &label);

} // namespace meshmodify
