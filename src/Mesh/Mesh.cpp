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
//+++ Function: the mesh class of xAsFem
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/Mesh.h"
#include "Utils/MessagePrinter.h"

Mesh::Mesh() {
    resetMeshData();
}
void Mesh::resetMeshData() {
    m_Data.Nx=0;
    m_Data.Ny=0;
    m_Data.Nz=0;
    m_Data.Xmin=0.0;
    m_Data.Xmax=0.0;
    m_Data.Ymin=0.0;
    m_Data.Ymax=0.0;
    m_Data.Zmin=0.0;
    m_Data.Zmax=0.0;

    m_Data.NodesNum=0;
    m_Data.NodesNumPerBulkElmt=0;
    m_Data.NodesNumPerSurfElmt=0;
    m_Data.NodesNumPerLineElmt=0;

    m_Data.ElmtsNum=0;
    m_Data.BulkElmtsNum=0;
    m_Data.SurfElmtsNum=0;
    m_Data.LineElmtsNum=0;
    m_Data.PointElmtsNum=0;

    m_Data.ElmtConn.clear();
    m_Data.BulkElmtConn.clear();
    m_Data.NodeCoords.clear();
    m_Data.BulkElmtPDNodeID.clear();
    m_Data.BulkElmtVolumes.clear();
    m_Data.BulkElmtCenters.clear();
    m_Data.MeshOrder=-1;
    m_Data.MeshDim=0;

    m_Data.BulkElmtVTKCellType=0;
    m_Data.SurfElmtVTKCellType=0;
    m_Data.LineElmtVTKCellType=0;
    m_Data.BulkElmtMeshType=MeshType::NULLTYPE;
    m_Data.SurfElmtMeshType=MeshType::NULLTYPE;
    m_Data.LineElmtMeshType=MeshType::NULLTYPE;
    m_Data.BulkMeshTypeName.clear();

    m_Data.IsImported=false;
    m_Data.ImportedFileName.clear();
    m_Data.CharacteristicLength=0.0;

    m_Data.PhyGroupNum=0;
    m_Data.PhyGroupDimVec.clear();
    m_Data.PhyGroupIDVec.clear();
    m_Data.PhyGroupNameVec.clear();
    m_Data.PhyGroupElmtsNumVec.clear();
    m_Data.PhyGroupID2NameMap.clear();
    m_Data.PhyGroupName2IDMap.clear();
    m_Data.PhyGroupID2ElmtConnMap.clear();
    m_Data.PhyGroupName2ElmtConnMap.clear();
    m_Data.PhyGroupName2BulkElmtIDVecMap.clear();
    m_Data.PhyGroupID2BulkElmtIDVecMap.clear();
    m_Data.PhyGroupName2BoundaryBulkElmtIDVecMap.clear();
    m_Data.PhyGroupName2BoundaryNormalVecMap.clear();
    m_Data.PhyGroupName2BoundaryMeasureVecMap.clear();

}
//****************************************************
void Mesh::setMeshInfo(const int &nx,const double &xmin,const double &xmax,const MeshType &meshtype){
    m_Data.MeshDim=1;
    m_Data.Nx=nx;
    m_Data.Xmin=xmin;
    m_Data.Xmax=xmax;
    m_Data.BulkElmtMeshType=meshtype;
}
void Mesh::setMeshInfo(const int &nx,const int &ny,
                       const double &xmin,const double &xmax,
                       const double &ymin,const double &ymax,const MeshType &meshtype){
    m_Data.MeshDim=2;
    m_Data.Nx=nx;
    m_Data.Ny=ny;
    m_Data.Xmin=xmin;
    m_Data.Xmax=xmax;
    m_Data.Ymin=ymin;
    m_Data.Ymax=ymax;
    m_Data.BulkElmtMeshType=meshtype;
}
// setup 3d mesh info
void Mesh::setMeshInfo(const int &nx,const int &ny,const int &nz,
                       const double &xmin,const double &xmax,
                       const double &ymin,const double &ymax,
                       const double &zmin,const double &zmax,const MeshType &meshtype){
    m_Data.MeshDim=3;
    m_Data.Nx=nx;
    m_Data.Ny=ny;
    m_Data.Nz=nz;
    m_Data.Xmin=xmin;
    m_Data.Xmax=xmax;
    m_Data.Ymin=ymin;
    m_Data.Ymax=ymax;
    m_Data.Zmin=zmin;
    m_Data.Zmax=zmax;
    m_Data.BulkElmtMeshType=meshtype;
}
//****************************************************
void Mesh::printInfo() const {
    const int size=70;
    char buff[size];
    MessagePrinter::printNormalTxt("Mesh information summary:");

    snprintf(buff,size,"  dim= %1d, nodes=%6d, elements=%6d, order= %1d",getMeshDim(),getNodesNum(),getElmtsNum(),getMeshOrder());
    MessagePrinter::printNormalTxt(buff);
    if (m_Data.IsImported) {
        snprintf(buff,size,"  imported mesh, characteristic dx=%10.3e",m_Data.CharacteristicLength);
        MessagePrinter::printNormalTxt(buff);
    }

    snprintf(buff,size,"  bulk  elements=%6d, nodes per bulk  elmt=%2d",getBulkElmtsNum(),getNodesNumPerBulkElmt());
    MessagePrinter::printNormalTxt(buff);

    snprintf(buff,size,"  point elements=%6d, nodes per point elmt=%2d",getPointElmtsNum(),1);
    MessagePrinter::printNormalTxt(buff);

    snprintf(buff,size,"  line  elements=%6d, nodes per line  elmt=%2d",getLineElmtsNum(),getNodesNumPerLineElmt());
    MessagePrinter::printNormalTxt(buff);

    snprintf(buff,size,"  surf  elements=%6d, nodes per surf  elmt=%2d",getSurfElmtsNum(),getNodesNumPerSurfElmt());
    MessagePrinter::printNormalTxt(buff);

    MessagePrinter::printDashLine();
    snprintf(buff,size,"  physical groups=%6d",getPhyGroupNum());
    MessagePrinter::printNormalTxt(buff);

    MessagePrinter::printNormalTxt("  phyid                   phyname       dim     elmts");
    int phyid,dim,datasize;
    string phyname;
    for (int i=1;i<=getPhyGroupNum();i++) {
        phyid=getIthPhyGroupID(i);
        phyname=getIthPhyGroupName(i);
        dim=getIthPhyGroupDim(i);
        datasize=getElmtsNumViaPhyName(phyname);
        snprintf(buff,size,"  %5d %25s      %2d    %8d",phyid,phyname.c_str(),dim,datasize);
        MessagePrinter::printNormalTxt(buff);
    }

    MessagePrinter::printStars();
}
