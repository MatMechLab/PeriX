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
//+++ Date    : 2026.07.08
//+++ Function: common driver for the concentric-shell PD lattice
//+++           generators (fills MeshData as an imported point cloud)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/PDLatticeGeneratorBase.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Utils/MessagePrinter.h"

bool PDLatticeGeneratorBase::generate(const Params &params,MeshData &t_MeshData) const {
    const int D=dim();
    if (params.R<=0.0) {
        MessagePrinter::printErrorTxt(std::string(shapeName())+" lattice: radius R must be > 0");
        return false;
    }
    if (params.N<1) {
        MessagePrinter::printErrorTxt(std::string(shapeName())+" lattice: number of layers N must be >= 1");
        return false;
    }

    const double dx=params.R/static_cast<double>(params.N);
    const std::array<double,3> c=params.center;

    // ---- 1. walk the shells, collect points, volumes and the outer shell ----
    std::vector<std::array<double,3>> points;
    std::vector<double> volumes;
    std::vector<int> outerShellPointIDs; // 1-based ids of the outermost layer

    for (int k=0;k<params.N;k++) {
        const double r_eff=(static_cast<double>(k)+0.5)*dx;
        const int n_k=pointsOnShell(k,r_eff,dx);
        const double measure=shellMeasure(k,dx);
        const double volPerPoint=measure/static_cast<double>(n_k);

        const std::size_t before=points.size();
        placeShellPoints(k,r_eff,n_k,c,points);
        const int placed=static_cast<int>(points.size()-before);
        if (placed!=n_k) {
            MessagePrinter::printErrorTxt(std::string(shapeName())
                +" lattice: a shell placed a wrong number of points");
            return false;
        }
        for (int p=0;p<n_k;p++) volumes.push_back(volPerPoint);
        if (k==params.N-1) {
            for (int p=0;p<n_k;p++)
                outerShellPointIDs.push_back(static_cast<int>(before)+p+1); // 1-based
        }
    }

    const int nBulk=static_cast<int>(points.size());
    if (nBulk<=0) {
        MessagePrinter::printErrorTxt(std::string(shapeName())+" lattice: produced no points");
        return false;
    }

    // ---- 2. fill the imported-style point cloud in MeshData ----
    t_MeshData=MeshData{};   // start from a clean slate
    t_MeshData.MeshDim=D;
    t_MeshData.IsImported=true;
    t_MeshData.MeshOrder=1;
    t_MeshData.CharacteristicLength=dx;

    t_MeshData.BulkElmtsNum=nBulk;
    t_MeshData.NodesNumPerBulkElmt=1;
    t_MeshData.BulkElmtVTKCellType=1; // VTK_VERTEX
    t_MeshData.BulkElmtMeshType=MeshType::POINT;
    t_MeshData.BulkMeshTypeName="point";

    t_MeshData.BulkElmtCenters.resize(static_cast<std::size_t>(3*nBulk));
    t_MeshData.BulkElmtVolumes.resize(static_cast<std::size_t>(nBulk));
    t_MeshData.BulkElmtConn.resize(static_cast<std::size_t>(nBulk));
    for (int i=0;i<nBulk;i++) {
        t_MeshData.BulkElmtCenters[3*i+0]=points[i][0];
        t_MeshData.BulkElmtCenters[3*i+1]=points[i][1];
        t_MeshData.BulkElmtCenters[3*i+2]=(D==3)?points[i][2]:0.0;
        t_MeshData.BulkElmtVolumes[i]=volumes[i];
        t_MeshData.BulkElmtConn[i]=std::vector<int>{i+1}; // 1-based single-node cell
    }

    // ---- 3. build the outer boundary faces (single-node radial faces) ----
    // Each outermost-shell point p (radius r_eff=(N-0.5)dx) gets a boundary
    // face node at radius R along its own radial direction, with an outward
    // radial normal. createPDMesh reflects the point across that face plane to
    // an exterior ghost at radius R+dx/2 -- exactly the half-cell mirror the
    // structured generators use, but fanned out radially around the boundary.
    std::vector<std::array<double,3>> boundaryNodes;
    std::vector<std::vector<int>> boundaryConn;
    std::vector<int> boundaryBulkID;
    std::vector<std::array<double,3>> boundaryNormals;
    std::vector<double> boundaryMeasures;

    const bool makeBoundary=params.makeBoundaryGroup && !outerShellPointIDs.empty();
    if (makeBoundary) {
        const int nOuter=static_cast<int>(outerShellPointIDs.size());
        const double faceMeasure=boundarySurface(params.R)/static_cast<double>(nOuter);
        boundaryNodes.reserve(static_cast<std::size_t>(nOuter));
        boundaryConn.reserve(static_cast<std::size_t>(nOuter));
        boundaryBulkID.reserve(static_cast<std::size_t>(nOuter));
        boundaryNormals.reserve(static_cast<std::size_t>(nOuter));
        boundaryMeasures.reserve(static_cast<std::size_t>(nOuter));
        for (const int bulkID:outerShellPointIDs) {
            const double px=points[bulkID-1][0]-c[0];
            const double py=points[bulkID-1][1]-c[1];
            const double pz=(D==3)?(points[bulkID-1][2]-c[2]):0.0;
            const double r=std::sqrt(px*px+py*py+pz*pz);
            if (r<=0.0) continue; // a point at the centre has no radial direction
            const double s=params.R/r; // scale the point out to radius R
            const int nodeID=nBulk+static_cast<int>(boundaryNodes.size())+1; // 1-based
            boundaryNodes.push_back({c[0]+s*px,c[1]+s*py,(D==3)?(c[2]+s*pz):0.0});
            boundaryConn.push_back(std::vector<int>{nodeID});
            boundaryBulkID.push_back(bulkID);
            boundaryNormals.push_back({px/r,py/r,(D==3)?(pz/r):0.0});
            boundaryMeasures.push_back(faceMeasure);
        }
    }

    // ---- 4. node coordinates (bulk points first, then boundary face nodes) --
    const int nNodes=nBulk+static_cast<int>(boundaryNodes.size());
    t_MeshData.NodesNum=nNodes;
    t_MeshData.NodeCoords.resize(static_cast<std::size_t>(3*nNodes),0.0);
    for (int i=0;i<nBulk;i++) {
        t_MeshData.NodeCoords[3*i+0]=points[i][0];
        t_MeshData.NodeCoords[3*i+1]=points[i][1];
        t_MeshData.NodeCoords[3*i+2]=(D==3)?points[i][2]:0.0;
    }
    for (std::size_t b=0;b<boundaryNodes.size();b++) {
        const std::size_t idx=static_cast<std::size_t>(nBulk)+b;
        t_MeshData.NodeCoords[3*idx+0]=boundaryNodes[b][0];
        t_MeshData.NodeCoords[3*idx+1]=boundaryNodes[b][1];
        t_MeshData.NodeCoords[3*idx+2]=boundaryNodes[b][2];
    }

    // ---- 5. bounding box ----
    double xmin=std::numeric_limits<double>::max(),xmax=-xmin;
    double ymin=xmin,ymax=xmax,zmin=xmin,zmax=xmax;
    for (int i=0;i<nNodes;i++) {
        const double x=t_MeshData.NodeCoords[3*i+0];
        const double y=t_MeshData.NodeCoords[3*i+1];
        const double z=t_MeshData.NodeCoords[3*i+2];
        xmin=std::min(xmin,x); xmax=std::max(xmax,x);
        ymin=std::min(ymin,y); ymax=std::max(ymax,y);
        zmin=std::min(zmin,z); zmax=std::max(zmax,z);
    }
    t_MeshData.Xmin=xmin; t_MeshData.Xmax=xmax;
    t_MeshData.Ymin=ymin; t_MeshData.Ymax=ymax;
    t_MeshData.Zmin=(D==3)?zmin:0.0; t_MeshData.Zmax=(D==3)?zmax:0.0;

    // ---- 6. physical groups: "alldomain" (bulk) + optional "outer" boundary --
    const int boundaryDim=D-1;
    const bool haveBoundary=makeBoundary && !boundaryConn.empty();
    t_MeshData.PhyGroupNum=haveBoundary?2:1;
    t_MeshData.PhyGroupDimVec={D};
    t_MeshData.PhyGroupIDVec={0};
    t_MeshData.PhyGroupNameVec={"alldomain"};
    t_MeshData.PhyGroupElmtsNumVec={nBulk};
    t_MeshData.PhyGroupID2NameMap[0]="alldomain";
    t_MeshData.PhyGroupName2IDMap["alldomain"]=0;
    t_MeshData.PhyGroupName2ElmtConnMap["alldomain"]=t_MeshData.BulkElmtConn;
    t_MeshData.PhyGroupID2ElmtConnMap[0]=t_MeshData.BulkElmtConn;

    if (haveBoundary) {
        const std::string &name=params.boundaryName;
        t_MeshData.PhyGroupDimVec.push_back(boundaryDim);
        t_MeshData.PhyGroupIDVec.push_back(1);
        t_MeshData.PhyGroupNameVec.push_back(name);
        t_MeshData.PhyGroupElmtsNumVec.push_back(static_cast<int>(boundaryConn.size()));
        t_MeshData.PhyGroupID2NameMap[1]=name;
        t_MeshData.PhyGroupName2IDMap[name]=1;
        t_MeshData.PhyGroupName2ElmtConnMap[name]=boundaryConn;
        t_MeshData.PhyGroupID2ElmtConnMap[1]=boundaryConn;
        t_MeshData.PhyGroupName2BoundaryBulkElmtIDVecMap[name]=boundaryBulkID;
        t_MeshData.PhyGroupName2BoundaryNormalVecMap[name]=boundaryNormals;
        t_MeshData.PhyGroupName2BoundaryMeasureVecMap[name]=boundaryMeasures;
    }

    char buff[128];
    snprintf(buff,sizeof(buff),
             "%s lattice: N=%d layers, dx=%.4e, %d points, %d boundary faces",
             shapeName(),params.N,dx,nBulk,static_cast<int>(boundaryConn.size()));
    MessagePrinter::printNormalTxt(buff);
    return true;
}
