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

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

#include <cmath>

namespace {
    // Signed side of point u relative to direction v with a RELATIVE epsilon:
    // +1 / -1 for a clear left/right, 0 when the cross product is below the
    // floating-point noise floor of its own terms. The slit convention is
    // "endpoint touching = non-crossing"; without the epsilon a bond passing
    // EXACTLY through a crack endpoint gets an arbitrary ulp-level sign that
    // can differ between two algebraically equal coordinate paths (e.g. a
    // ghost's literal position vs its mirror bulk's), making the cut
    // non-deterministic. 1e-12 is ~4 orders above the ~1e-16 rounding noise
    // and many orders below any real off-tip crossing (|det| ~ cell^2).
    [[nodiscard]] int orientSide(const double ux,const double uy,
                                 const double vx,const double vy) {
        const double det=ux*vy-uy*vx;
        const double scale=std::fabs(ux)*std::fabs(vy)+std::fabs(uy)*std::fabs(vx);
        if (det> 1.0e-12*scale) return +1;
        if (det<-1.0e-12*scale) return -1;
        return 0;
    }

    // Signed side of a point relative to the plane through the quad corner c0
    // with normal n (=(c1-c0)x(c3-c0)). Same relative-epsilon philosophy as
    // orientSide: a dot below the floating-point noise floor of its own terms
    // is "on the plane" (0), so an endpoint sitting exactly on the crack plane
    // yields a deterministic non-crossing verdict ("touching = non-crossing",
    // the Madenci slit convention; a notch must sit BETWEEN node columns).
    [[nodiscard]] int planeSide(const double px,const double py,const double pz,
                                const double c0x,const double c0y,const double c0z,
                                const double nx,const double ny,const double nz) {
        const double wx=px-c0x, wy=py-c0y, wz=pz-c0z;
        const double d=wx*nx+wy*ny+wz*nz;
        const double scale=std::fabs(wx*nx)+std::fabs(wy*ny)+std::fabs(wz*nz);
        if (d> 1.0e-12*scale) return +1;
        if (d<-1.0e-12*scale) return -1;
        return 0;
    }

    // True iff the bond A->B strictly pierces the bounded planar quadrilateral
    // {c0,c1,c2,c3} (looping/CCW order, assumed convex). Step 1: the two
    // endpoints must sit on strictly opposite sides of the quad's plane
    // (planeSide). Step 2: the segment/plane intersection point Q must fall
    // STRICTLY inside the quad, tested against all four directed edges with the
    // quad's own normal. The interior test is STRICT (a Q on the rim is NOT a
    // pierce) so the rim carries the same "touching = non-crossing" convention
    // as the 2D segment crack (orientSide): a bond crossing exactly at a crack
    // TIP/edge stays alive, and a bounded crack -- like a slit -- must sit
    // BETWEEN node columns. This makes a through-thickness crack plane cut the
    // exact same bonds as the equivalent 2D `Cracks` segment.
    [[nodiscard]] bool bondPiercesQuad(const double ax,const double ay,const double az,
                                       const double bx,const double by,const double bz,
                                       const std::array<double,12> &q) {
        const double cx[4]={q[0],q[3],q[6],q[9]};
        const double cy[4]={q[1],q[4],q[7],q[10]};
        const double cz[4]={q[2],q[5],q[8],q[11]};
        const double ux=cx[1]-cx[0], uy=cy[1]-cy[0], uz=cz[1]-cz[0]; // edge c0->c1
        const double vx=cx[3]-cx[0], vy=cy[3]-cy[0], vz=cz[3]-cz[0]; // edge c0->c3
        const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx; // quad normal
        const double nlen=std::sqrt(nx*nx+ny*ny+nz*nz);
        if (nlen<=0.0) return false; // degenerate quad

        const int sA=planeSide(ax,ay,az,cx[0],cy[0],cz[0],nx,ny,nz);
        const int sB=planeSide(bx,by,bz,cx[0],cy[0],cz[0],nx,ny,nz);
        if (sA*sB>=0) return false;  // same side, or an endpoint on the plane

        const double dA=(ax-cx[0])*nx+(ay-cy[0])*ny+(az-cz[0])*nz;
        const double dB=(bx-cx[0])*nx+(by-cy[0])*ny+(bz-cz[0])*nz;
        const double t=dA/(dA-dB);   // sA*sB<0 => dA-dB is safely non-zero
        const double Qx=ax+t*(bx-ax), Qy=ay+t*(by-ay), Qz=az+t*(bz-az);

        const double unx=nx/nlen, uny=ny/nlen, unz=nz/nlen;
        const double tol=1.0e-9*nlen; // twice-area units
        for (int k=0;k<4;k++) {
            const int m=(k+1)%4;
            const double ex=cx[m]-cx[k], ey=cy[m]-cy[k], ez=cz[m]-cz[k]; // edge k
            const double gx=Qx-cx[k], gy=Qy-cy[k], gz=Qz-cz[k];
            const double wx=ey*gz-ez*gy, wy=ez*gx-ex*gz, wz=ex*gy-ey*gx;
            if (wx*unx+wy*uny+wz*unz <= tol) return false; // on or outside edge k
        }
        return true;
    }
}

PDMesh::PDMesh()=default;
PDMesh::~PDMesh()=default;

bool initialCrackCutsBond(const PDMeshData &Data,const int &i,const int &j) {
    if (!Data.hasInitialCracks()) return false;

    // resolve a ghost endpoint to its mirror bulk (u_ghost is slaved to
    // u_mirrorbulk by the BC rows, so that is the point the bond couples).
    const bool haveMirror=
        static_cast<int>(Data.GhostMirrorBulkID.size())>=Data.NodesNum;
    auto resolve=[&](const int id)->int{
        if (haveMirror && id>=1 && id<=Data.NodesNum) {
            const int B=Data.GhostMirrorBulkID[static_cast<std::size_t>(id-1)];
            if (B>=1 && B<=Data.NodesNum) return B;
        }
        return id;
    };
    const int a=resolve(i);
    const int b=resolve(j);
    // a ghost and its own mirror bulk (or two images of the same bulk) sit on
    // the same side by definition; the pairing bond carries the BC and is
    // never cut by an initial slit.
    if (a==b) return false;

    const double ax=Data.NodeCoords[(a-1)*3+0];
    const double ay=Data.NodeCoords[(a-1)*3+1];
    const double az=Data.NodeCoords[(a-1)*3+2];
    const double bx=Data.NodeCoords[(b-1)*3+0];
    const double by=Data.NodeCoords[(b-1)*3+1];
    const double bz=Data.NodeCoords[(b-1)*3+2];

    // (1) 2D segment cracks. In 2D a crack is a line segment; in 3D it is the
    // through-thickness plane obtained by extruding that segment along z, so
    // the SAME xy segment-crossing test cuts the bond at any z. Strict
    // cross-product orientation test with a relative epsilon (see orientSide);
    // endpoint touching counts as non-crossing (Madenci slit convention -> a
    // notch must sit BETWEEN node columns).
    for (const auto &crack : Data.Cracks) {
        const double cx=crack[0];
        const double cy=crack[1];
        const double dx=crack[2];
        const double dy=crack[3];
        const int s1=orientSide(dx-cx,dy-cy,ax-cx,ay-cy);
        const int s2=orientSide(dx-cx,dy-cy,bx-cx,by-cy);
        if (s1*s2>=0) continue;
        const int s3=orientSide(bx-ax,by-ay,cx-ax,cy-ay);
        const int s4=orientSide(bx-ax,by-ay,dx-ax,dy-ay);
        if (s3*s4<0) return true;
    }

    // (2) Bounded 3D crack planes: a bond that strictly pierces the quad
    // interior is severed (see bondPiercesQuad). This is a genuine planar
    // patch, so unlike a `Cracks` segment it can be finite in z, inclined, or
    // parallel to the xy plane.
    for (const auto &quad : Data.CrackPlanes) {
        if (bondPiercesQuad(ax,ay,az,bx,by,bz,quad)) return true;
    }
    return false;
}

bool PDMesh::initialCrackCutsBond(const int &i,const int &j) const {
    return ::initialCrackCutsBond(m_Data,i,j);
}

void PDMesh::printPDMeshInfo() const {
    MessagePrinter::printNormalTxt("PD mesh information summary:");
    const int size=78;
    char buff[size];
    snprintf(buff,size,"  PD nodes=%8d, max/min neighbors=%3d/%3d, radius=%10.3e",m_Data.NodesNum,m_Data.MaxNeighbors,m_Data.MinNeighbors,m_Data.HorizonRadius);
    MessagePrinter::printNormalTxt(buff);
    if (m_Data.HorizonRadiusFactor>0.0) {
        snprintf(buff,size,"  horizon factor=%10.3e, base dx=%10.3e",
                 m_Data.HorizonRadiusFactor,m_Data.HorizonBaseDX);
        MessagePrinter::printNormalTxt(buff);
    }
    for (const auto &it:m_Data.PhyNameToNodeIDsMap) {
        snprintf(buff,size,"  phyname= %12s ===> pd nodes=%6d",it.first.c_str(),static_cast<int>(it.second.size()));
        MessagePrinter::printNormalTxt(buff);
    }
    MessagePrinter::printStars();
}
