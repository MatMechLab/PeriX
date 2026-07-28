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
//+++ Function: the mesh data structure
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <array>
#include <vector>
#include <string>
#include <map>

#include "Mesh/MeshType.h"

using namespace std;

/**
 * data structure for the finite element mesh
 */
struct MeshData {
    int Nx=0,Ny=0,Nz=0;/**< the elements number along x,y,z axis */
    double Xmin=0.0,Xmax=0.0,Ymin=0.0,Ymax=0.0,Zmin=0.0,Zmax=0.0;/**< the axis length along each direction */

    int NodesNum=0;/**< the total nodes num */
    int NodesNumPerBulkElmt=0;/**< the nodes num per bulk element */
    int NodesNumPerSurfElmt=0;/**< the nodes num per surface element */
    int NodesNumPerLineElmt=0;/**< the nodes num per line element */

    int ElmtsNum=0;/**< total elements num */
    int BulkElmtsNum=0;/**< the bulk elements num */
    int SurfElmtsNum=0;/**< the surface elements num */
    int LineElmtsNum=0;/**< the line elements num */
    int PointElmtsNum=0;/**< the point elements num */

    vector<vector<int>> ElmtConn;/**< the connectivity of all elements */
    vector<vector<int>> BulkElmtConn;/**< the connectivity of bulk elements */
    vector<double> NodeCoords;/**< the coordinates of nodes */
    int MeshOrder=-1;/**< the mesh order */
    int MeshDim=0;/**< the dim of your mesh */

    vector<int> BulkElmtPDNodeID;
    vector<double> BulkElmtVolumes;/**< area/volume of each bulk element */
    vector<double> BulkElmtCenters;/**< xyz center of each bulk element */

    int BulkElmtVTKCellType=0;/**< the vtk cell type of bulk element */
    int SurfElmtVTKCellType=0;/**< the vtk cell type of surf element */
    int LineElmtVTKCellType=0;/**< the vtk cell type of line element */

    string BulkMeshTypeName;/**< mesh type name of the bulk element */

    MeshType BulkElmtMeshType=MeshType::NULLTYPE;/**< the mesh type of bulk element */
    MeshType SurfElmtMeshType=MeshType::NULLTYPE;/**< the mesh type of surface element */
    MeshType LineElmtMeshType=MeshType::NULLTYPE;/**< the mesh type of line element */

    bool IsImported=false;/**< true when the mesh came from an external file */
    string ImportedFileName;/**< imported mesh file name */
    double CharacteristicLength=0.0;/**< robust mesh length used as imported dx */

    // for physical group info
    int            PhyGroupNum=0;/**< the physical group number */
    vector<int>    PhyGroupDimVec;/**< the dim vector */
    vector<int>    PhyGroupIDVec;/**< the id vector */
    vector<string> PhyGroupNameVec;/**< the phy name vector */
    vector<int>    PhyGroupElmtsNumVec;/**< the physical group elements number vector */

    map<int,string> PhyGroupID2NameMap;/**< the phyid to phyname map */
    map<string,int> PhyGroupName2IDMap;/**< the phyname to phyid map */

    map<int,vector<vector<int>>>    PhyGroupID2ElmtConnMap;/**< the phy id to element conn map */
    map<string,vector<vector<int>>> PhyGroupName2ElmtConnMap;/**< the phy name to element conn map */

    map<string,vector<int>> PhyGroupName2BulkElmtIDVecMap;/**< the phy name to bulk element id map */
    map<int,vector<int>>    PhyGroupID2BulkElmtIDVecMap;/**< the phy name to bulk element id map */

    map<string,vector<int>> PhyGroupName2BoundaryBulkElmtIDVecMap;/**< boundary elmt to adjacent bulk id */
    map<string,vector<array<double,3>>> PhyGroupName2BoundaryNormalVecMap;/**< outward unit normal per boundary elmt */
    map<string,vector<double>> PhyGroupName2BoundaryMeasureVecMap;/**< boundary length/area per boundary elmt */

};
