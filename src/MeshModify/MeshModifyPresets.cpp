//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "MeshModify/MeshModifyGeometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "Mesh/MeshData.h"
#include "Utils/MessagePrinter.h"

namespace meshmodify {

Vec3 vsub(const Vec3 &a,const Vec3 &b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}

double vdot(const Vec3 &a,const Vec3 &b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

Vec3 vcross(const Vec3 &a,const Vec3 &b) {
    return {a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}

double vnorm(const Vec3 &a) {
    return std::sqrt(vdot(a,a));
}

// A default vector inside the crack plane (unit normal n) used as the primary
// (length) in-plane axis when the user does not pin one: the projection onto
// the plane of the FIRST coordinate axis, in x->y->z order, that is not
// (nearly) parallel to n. This makes "length" prefer a horizontal in-plane
// direction, so the secondary axis (n x a1) -- the "width" -- naturally points
// along the thickness for the common vertical mode-I crack.
Vec3 inPlaneHelper(const Vec3 &n) {
    const Vec3 axes[3]={{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    for (const auto &e:axes) {
        const double d=vdot(e,n);
        const Vec3 p{e[0]-d*n[0],e[1]-d*n[1],e[2]-d*n[2]};
        if (vnorm(p)>1.0e-6) return p;
    }
    return {1.0,0.0,0.0};
}

MeshBox getMeshBox(const MeshData &meshData) {
    MeshBox box{meshData.Xmin,meshData.Xmax,meshData.Ymin,meshData.Ymax,
                meshData.Zmin,meshData.Zmax};
    if (meshData.NodesNum>0
        && static_cast<int>(meshData.NodeCoords.size())>=3*meshData.NodesNum) {
        box.xmin=std::numeric_limits<double>::max();
        box.ymin=std::numeric_limits<double>::max();
        box.zmin=std::numeric_limits<double>::max();
        box.xmax=-std::numeric_limits<double>::max();
        box.ymax=-std::numeric_limits<double>::max();
        box.zmax=-std::numeric_limits<double>::max();
        for (int i=0;i<meshData.NodesNum;i++) {
            const double x=meshData.NodeCoords[3*static_cast<std::size_t>(i)+0];
            const double y=meshData.NodeCoords[3*static_cast<std::size_t>(i)+1];
            const double z=meshData.NodeCoords[3*static_cast<std::size_t>(i)+2];
            box.xmin=std::min(box.xmin,x);
            box.xmax=std::max(box.xmax,x);
            box.ymin=std::min(box.ymin,y);
            box.ymax=std::max(box.ymax,y);
            box.zmin=std::min(box.zmin,z);
            box.zmax=std::max(box.zmax,z);
        }
    }
    return box;
}

double orient2D(double ax,double ay,double bx,double by,double cx,double cy) {
    return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
}

bool isUsableBox(const MeshBox &box) {
    return box.xmax>box.xmin || box.ymax>box.ymin || box.zmax>box.zmin;
}

bool pointInBox(double x,double y,const MeshBox &box,double tol) {
    return x>=box.xmin-tol && x<=box.xmax+tol
        && y>=box.ymin-tol && y<=box.ymax+tol;
}

bool onSegment(double ax,double ay,double bx,double by,
               double px,double py,double tol) {
    return std::abs(orient2D(ax,ay,bx,by,px,py))<=tol
        && px>=std::min(ax,bx)-tol && px<=std::max(ax,bx)+tol
        && py>=std::min(ay,by)-tol && py<=std::max(ay,by)+tol;
}

bool closedSegmentsIntersect(double ax,double ay,double bx,double by,
                             double cx,double cy,double dx,double dy,
                             double tol) {
    const double o1=orient2D(ax,ay,bx,by,cx,cy);
    const double o2=orient2D(ax,ay,bx,by,dx,dy);
    const double o3=orient2D(cx,cy,dx,dy,ax,ay);
    const double o4=orient2D(cx,cy,dx,dy,bx,by);
    if (((o1>tol && o2<-tol)||(o1<-tol && o2>tol)) &&
        ((o3>tol && o4<-tol)||(o3<-tol && o4>tol))) {
        return true;
    }
    return onSegment(ax,ay,bx,by,cx,cy,tol)
        || onSegment(ax,ay,bx,by,dx,dy,tol)
        || onSegment(cx,cy,dx,dy,ax,ay,tol)
        || onSegment(cx,cy,dx,dy,bx,by,tol);
}

bool segmentOverlapsBox(const MeshModify::CrackSegment &s,const MeshBox &box) {
    if (!isUsableBox(box)) return true;
    const double scale=std::max({1.0,std::abs(box.xmin),std::abs(box.xmax),
                                 std::abs(box.ymin),std::abs(box.ymax)});
    const double tol=1.0e-12*scale;
    if (pointInBox(s.x1,s.y1,box,tol) || pointInBox(s.x2,s.y2,box,tol)) {
        return true;
    }
    return closedSegmentsIntersect(s.x1,s.y1,s.x2,s.y2,
                                   box.xmin,box.ymin,box.xmax,box.ymin,tol)
        || closedSegmentsIntersect(s.x1,s.y1,s.x2,s.y2,
                                   box.xmax,box.ymin,box.xmax,box.ymax,tol)
        || closedSegmentsIntersect(s.x1,s.y1,s.x2,s.y2,
                                   box.xmax,box.ymax,box.xmin,box.ymax,tol)
        || closedSegmentsIntersect(s.x1,s.y1,s.x2,s.y2,
                                   box.xmin,box.ymax,box.xmin,box.ymin,tol);
}

// Loose AABB-vs-AABB overlap between the plane's corner box and the mesh
// bounding box (the 3D counterpart of segmentOverlapsBox).
bool planeOverlapsBox(const MeshModify::CrackPlaneSpec &s,const MeshBox &box) {
    if (!isUsableBox(box)) return true;
    double pxmin=s.corners[0][0],pxmax=s.corners[0][0];
    double pymin=s.corners[0][1],pymax=s.corners[0][1];
    double pzmin=s.corners[0][2],pzmax=s.corners[0][2];
    for (const auto &c:s.corners) {
        pxmin=std::min(pxmin,c[0]); pxmax=std::max(pxmax,c[0]);
        pymin=std::min(pymin,c[1]); pymax=std::max(pymax,c[1]);
        pzmin=std::min(pzmin,c[2]); pzmax=std::max(pzmax,c[2]);
    }
    const double scale=std::max({1.0,std::abs(box.xmin),std::abs(box.xmax),
                                 std::abs(box.ymin),std::abs(box.ymax),
                                 std::abs(box.zmin),std::abs(box.zmax)});
    const double tol=1.0e-12*scale;
    return pxmax>=box.xmin-tol && pxmin<=box.xmax+tol
        && pymax>=box.ymin-tol && pymin<=box.ymax+tol
        && pzmax>=box.zmin-tol && pzmin<=box.zmax+tol;
}

double defaultEdgeAngle(const std::string &side) {
    if (side=="left")   return 0.0;
    if (side=="right")  return kPi;
    if (side=="bottom") return 0.5*kPi;
    if (side=="top")    return -0.5*kPi;
    MessagePrinter::printErrorTxt(
        "MeshModify edge_crack preset: side='"+side
        +"' is invalid (supported: left, right, bottom, top)");
    MessagePrinter::exitPeriX();
    return 0.0;
}

// Extrude an xy segment (x1,y1)-(x2,y2) over [zlo,zhi] into a rectangular
// crack plane. Corners loop c0->c1 along the segment at zlo, up to zhi, back
// along the segment, and down -- a valid planar quad whose normal is
// horizontal (perpendicular to the segment, in the xy plane).
MeshModify::CrackPlaneSpec extrudeSegment(
    double x1,double y1,double x2,double y2,
    double zlo,double zhi,const std::string &label) {
    MeshModify::CrackPlaneSpec s;
    s.corners[0]={x1,y1,zlo};
    s.corners[1]={x2,y2,zlo};
    s.corners[2]={x2,y2,zhi};
    s.corners[3]={x1,y1,zhi};
    s.label=label;
    return s;
}

} // namespace meshmodify

using namespace meshmodify;

MeshModify::CrackSegment
MeshModify::resolveCenterPreset(const CenterCrackPreset &preset) const {
    const double dx=0.5*preset.length*std::cos(preset.angleRadians);
    const double dy=0.5*preset.length*std::sin(preset.angleRadians);
    return {preset.centerX-dx,preset.centerY-dy,
            preset.centerX+dx,preset.centerY+dy,
            preset.label};
}

MeshModify::CrackSegment
MeshModify::resolveEdgePreset(const EdgeCrackPreset &preset,
                              const MeshData &meshData) const {
    const MeshBox box=getMeshBox(meshData);
    std::string side=preset.side;
    std::transform(side.begin(),side.end(),side.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    double x0=0.0,y0=0.0;
    if (side=="left")        { x0=box.xmin; y0=preset.position; }
    else if (side=="right")  { x0=box.xmax; y0=preset.position; }
    else if (side=="bottom") { x0=preset.position; y0=box.ymin; }
    else if (side=="top")    { x0=preset.position; y0=box.ymax; }
    else {
        MessagePrinter::printErrorTxt(
            "MeshModify edge_crack preset: side='"+preset.side
            +"' is invalid (supported: left, right, bottom, top)");
        MessagePrinter::exitPeriX();
    }

    const bool xSide=(side=="left" || side=="right");
    const double minPos=xSide ? box.ymin : box.xmin;
    const double maxPos=xSide ? box.ymax : box.xmax;
    const double scale=std::max({1.0,std::abs(minPos),std::abs(maxPos)});
    const double tol=1.0e-12*scale;
    if (preset.position<minPos-tol || preset.position>maxPos+tol) {
        MessagePrinter::printErrorTxt(
            "MeshModify edge_crack preset '"+preset.label
            +"': position="+std::to_string(preset.position)
            +" is outside the selected side range ["
            +std::to_string(minPos)+", "+std::to_string(maxPos)+"]");
        MessagePrinter::exitPeriX();
    }

    const double angle=preset.hasAngle ? preset.angleRadians
                                       : defaultEdgeAngle(side);
    const double x1=x0+preset.length*std::cos(angle);
    const double y1=y0+preset.length*std::sin(angle);
    return {x0,y0,x1,y1,preset.label};
}

MeshModify::CrackPlaneSpec
MeshModify::resolveCenterPreset3D(const CenterCrackPreset &preset,
                                  const MeshData &meshData) const {
    const MeshBox box=getMeshBox(meshData);
    const double cz=preset.hasCenterZ ? preset.centerZ
                                      : 0.5*(box.zmin+box.zmax);

    // Seamless generalization of the 2D center crack: with no 3D orientation
    // fields, reuse the (centerX,centerY,length,angle) segment and extrude it
    // through the full thickness -> a through-thickness crack plane.
    if (!preset.hasNormal && !preset.hasAxis && !preset.hasWidth) {
        const double dxh=0.5*preset.length*std::cos(preset.angleRadians);
        const double dyh=0.5*preset.length*std::sin(preset.angleRadians);
        return extrudeSegment(preset.centerX-dxh,preset.centerY-dyh,
                              preset.centerX+dxh,preset.centerY+dyh,
                              box.zmin,box.zmax,preset.label);
    }

    // General bounded oriented rectangle: center + normal + length (primary
    // in-plane axis a1) + width (secondary in-plane axis a2 = n x a1).
    Vec3 n=preset.normal;
    const double nlen=vnorm(n);
    if (nlen<=0.0) {
        MessagePrinter::printErrorTxt(
            "MeshModify center_crack (3D) preset '"+preset.label
            +"': 'normal' must be a non-zero vector");
        MessagePrinter::exitPeriX();
    }
    n={n[0]/nlen,n[1]/nlen,n[2]/nlen};

    Vec3 a1;
    if (preset.hasAxis) {
        const double d=vdot(preset.axis,n);
        a1={preset.axis[0]-d*n[0],preset.axis[1]-d*n[1],preset.axis[2]-d*n[2]};
        if (vnorm(a1)<=1.0e-14) a1=inPlaneHelper(n);
    }
    else {
        a1=inPlaneHelper(n);
    }
    const double a1len=vnorm(a1);
    if (a1len<=0.0) {
        MessagePrinter::printErrorTxt(
            "MeshModify center_crack (3D) preset '"+preset.label
            +"': could not build an in-plane axis (degenerate normal)");
        MessagePrinter::exitPeriX();
    }
    a1={a1[0]/a1len,a1[1]/a1len,a1[2]/a1len};
    const Vec3 a2=vcross(n,a1);

    const double L=preset.length;
    const double W=preset.hasWidth ? preset.width : (box.zmax-box.zmin);
    const Vec3 c{preset.centerX,preset.centerY,cz};
    auto corner=[&](const double sL,const double sW)->Vec3{
        return {c[0]+sL*0.5*L*a1[0]+sW*0.5*W*a2[0],
                c[1]+sL*0.5*L*a1[1]+sW*0.5*W*a2[1],
                c[2]+sL*0.5*L*a1[2]+sW*0.5*W*a2[2]};
    };
    CrackPlaneSpec s;
    s.corners[0]=corner(-1.0,-1.0);
    s.corners[1]=corner(+1.0,-1.0);
    s.corners[2]=corner(+1.0,+1.0);
    s.corners[3]=corner(-1.0,+1.0);
    s.label=preset.label;
    return s;
}

MeshModify::CrackPlaneSpec
MeshModify::resolveEdgePreset3D(const EdgeCrackPreset &preset,
                                const MeshData &meshData) const {
    // Reuse the 2D edge segment (it handles side/position/length/angle and all
    // range checks), then extrude it through the thickness.
    const CrackSegment seg=resolveEdgePreset(preset,meshData);
    const MeshBox box=getMeshBox(meshData);
    double zlo=box.zmin,zhi=box.zmax;
    if (preset.hasWidth && preset.width>0.0) {
        const double zc=0.5*(box.zmin+box.zmax);
        zlo=zc-0.5*preset.width;
        zhi=zc+0.5*preset.width;
    }
    return extrudeSegment(seg.x1,seg.y1,seg.x2,seg.y2,zlo,zhi,preset.label);
}

void MeshModify::validateSegment(const CrackSegment &segment,
                                 const MeshData &meshData,
                                 const int index) const {
    if (meshData.MeshDim<2 || meshData.MeshDim>3) {
        MessagePrinter::printErrorTxt(
            "MeshModify: pre-existing cracks require a 2D or 3D mesh");
        MessagePrinter::exitPeriX();
    }
    const bool finite=std::isfinite(segment.x1) && std::isfinite(segment.y1)
        && std::isfinite(segment.x2) && std::isfinite(segment.y2);
    if (!finite) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack segment "+std::to_string(index)
            +" has a non-finite coordinate");
        MessagePrinter::exitPeriX();
    }
    const double dx=segment.x2-segment.x1;
    const double dy=segment.y2-segment.y1;
    if (dx*dx+dy*dy<=0.0) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack segment "+std::to_string(index)
            +" has zero length");
        MessagePrinter::exitPeriX();
    }
    if (!segmentOverlapsBox(segment,getMeshBox(meshData))) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack segment "+std::to_string(index)
            +" does not overlap the mesh bounding box");
        MessagePrinter::exitPeriX();
    }
}

void MeshModify::validatePlane(const CrackPlaneSpec &plane,
                               const MeshData &meshData,
                               const int index) const {
    if (meshData.MeshDim!=3) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack plane "+std::to_string(index)
            +" requires a 3D mesh");
        MessagePrinter::exitPeriX();
    }
    for (const auto &c:plane.corners) {
        if (!std::isfinite(c[0])||!std::isfinite(c[1])||!std::isfinite(c[2])) {
            MessagePrinter::printErrorTxt(
                "MeshModify: crack plane "+std::to_string(index)
                +" has a non-finite corner coordinate");
            MessagePrinter::exitPeriX();
        }
    }
    const Vec3 u=vsub(plane.corners[1],plane.corners[0]);
    const Vec3 v=vsub(plane.corners[3],plane.corners[0]);
    if (vnorm(vcross(u,v))<=0.0) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack plane "+std::to_string(index)
            +" is degenerate (zero area / collinear corners)");
        MessagePrinter::exitPeriX();
    }
    const Vec3 n=vcross(u,v);
    const Vec3 w=vsub(plane.corners[2],plane.corners[0]);
    const double scale=vnorm(u)*vnorm(v);
    if (std::abs(vdot(n,w))>1.0e-9*scale*std::max(1.0,vnorm(w))) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack plane "+std::to_string(index)
            +" is not planar (the four corners are not coplanar)");
        MessagePrinter::exitPeriX();
    }
    if (!planeOverlapsBox(plane,getMeshBox(meshData))) {
        MessagePrinter::printErrorTxt(
            "MeshModify: crack plane "+std::to_string(index)
            +" does not overlap the mesh bounding box");
        MessagePrinter::exitPeriX();
    }
}
