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
//+++ Function: the 2D quad4 lagrange mesh generator
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/Lagrange2DQuad4MeshGenerator.h"

Lagrange2DQuad4MeshGenerator::Lagrange2DQuad4MeshGenerator(){
    m_MeshGenerated=false;
}
Lagrange2DQuad4MeshGenerator::~Lagrange2DQuad4MeshGenerator(){
    m_MeshGenerated=false;
}
//*********************************************************
bool Lagrange2DQuad4MeshGenerator::generateMesh(MeshData &Data){
    Data.MeshDim=2;

    vector<vector<int>> leftconn,rightconn;
    vector<vector<int>> bottomconn,topconn;
    vector<int> leftnodes,rightnodes;
    vector<int> bottomnodes,topnodes;

    // only create mesh on the master rank, then distributed them into different ranks !
    double dx,dy;
    int i,j,k,e;
    int i1,i2,i3,i4;

    // generate the mesh cell for quad4 mesh
    Data.MeshOrder=1;
    Data.BulkMeshTypeName="quad4";
    dx=(Data.Xmax-Data.Xmin)/Data.Nx;
    dy=(Data.Ymax-Data.Ymin)/Data.Ny;

    Data.BulkElmtsNum=Data.Nx*Data.Ny;
    Data.PointElmtsNum=0;
    Data.LineElmtsNum=2*(Data.Nx+Data.Ny);
    Data.SurfElmtsNum=0;
    Data.ElmtsNum=Data.BulkElmtsNum
                 +Data.SurfElmtsNum
                 +Data.LineElmtsNum
                 +Data.PointElmtsNum;

    Data.NodesNum=(Data.Nx+1)*(Data.Ny+1);
    Data.NodesNumPerBulkElmt=4;
    Data.NodesNumPerSurfElmt=0;
    Data.NodesNumPerLineElmt=2;

    Data.BulkElmtVTKCellType=9;
    Data.BulkElmtMeshType=MeshType::QUAD4;

    Data.LineElmtVTKCellType=3;
    Data.LineElmtMeshType=MeshType::EDGE2;

    Data.SurfElmtVTKCellType=0;
    Data.SurfElmtMeshType=MeshType::NULLTYPE;

    Data.NodeCoords.resize(Data.NodesNum*3,0.0);
    leftnodes.clear();rightnodes.clear();
    bottomnodes.clear();rightnodes.clear();
    for(j=1;j<=Data.Ny+1;j++){
        for(i=1;i<=Data.Nx+1;i++){
            k=(j-1)*(Data.Nx+1)+i;
            Data.NodeCoords[(k-1)*3+1-1]=Data.Xmin+(i-1)*dx;
            Data.NodeCoords[(k-1)*3+2-1]=Data.Ymin+(j-1)*dy;
            Data.NodeCoords[(k-1)*3+3-1]=0.0;
        }
    }// end-of-node-generation

    // for the connectivity information of bulk elements
    Data.BulkElmtConn.resize(Data.BulkElmtsNum);
    leftconn.resize(Data.Ny);
    rightconn.resize(Data.Ny);
    //
    bottomconn.resize(Data.Nx);
    topconn.resize(Data.Nx);

    leftnodes.clear();rightnodes.clear();
    bottomnodes.clear();rightnodes.clear();

    Data.PhyGroupName2BulkElmtIDVecMap.clear();
    Data.PhyGroupID2BulkElmtIDVecMap.clear();
    Data.PhyGroupID2ElmtConnMap.clear();

    for(j=1;j<=Data.Ny;j++){
        for(i=1;i<=Data.Nx;i++){
            e=(j-1)*Data.Nx+i;
            i1=(j-1)*(Data.Nx+1)+i;
            i2=i1+1;
            i3=i2+Data.Nx+1;
            i4=i3-1;

            Data.PhyGroupName2BulkElmtIDVecMap["alldomain"].push_back(e);
            Data.PhyGroupID2BulkElmtIDVecMap[0].push_back(e);

            Data.BulkElmtConn[e-1].clear();
            Data.BulkElmtConn[e-1].push_back(i1);
            Data.BulkElmtConn[e-1].push_back(i2);
            Data.BulkElmtConn[e-1].push_back(i3);
            Data.BulkElmtConn[e-1].push_back(i4);

            // for the boundary element
            // the layout of your quad4 should be:
            // 4-----3
            // |     |
            // |     |
            // 1-----2
            if(j==1){
                // for bottom bc elements
                bottomconn[i-1].clear();
                bottomconn[i-1].push_back(i1);
                bottomconn[i-1].push_back(i2);

                bottomnodes.push_back(i1);
                bottomnodes.push_back(i2);
            }
            if(j==Data.Ny){
                // for top bc elements
                topconn[i-1].clear();
                topconn[i-1].push_back(i3);
                topconn[i-1].push_back(i4);

                topnodes.push_back(i3);
                topnodes.push_back(i4);
            }
            if(i==1){
                // for left bc elements
                leftconn[j-1].clear();
                leftconn[j-1].push_back(i4);
                leftconn[j-1].push_back(i1);

                leftnodes.push_back(i4);
                leftnodes.push_back(i1);
            }
            if(i==Data.Nx){
                // for right bc elements
                rightconn[j-1].clear();
                rightconn[j-1].push_back(i2);
                rightconn[j-1].push_back(i3);

                rightnodes.push_back(i2);
                rightnodes.push_back(i3);
            }
        }
    }// end-of-element-generation-loop

    Data.ElmtConn.clear();
    for (const auto &cell:leftconn) Data.ElmtConn.push_back(cell);
    for (const auto &cell:rightconn) Data.ElmtConn.push_back(cell);
    for (const auto &cell:bottomconn) Data.ElmtConn.push_back(cell);
    for (const auto &cell:topconn) Data.ElmtConn.push_back(cell);
    for (const auto &cell:Data.BulkElmtConn) Data.ElmtConn.push_back(cell);

    // remove the duplicate node id, for nodal type set, you don't need these duplicate node ids!
    sort(leftnodes.begin(),leftnodes.end());
    leftnodes.erase(unique(leftnodes.begin(),leftnodes.end()),leftnodes.end());
    // for rightnodes
    sort(rightnodes.begin(),rightnodes.end());
    rightnodes.erase(unique(rightnodes.begin(),rightnodes.end()),rightnodes.end());
    // for bottomnodes
    sort(bottomnodes.begin(),bottomnodes.end());
    bottomnodes.erase(unique(bottomnodes.begin(),bottomnodes.end()),bottomnodes.end());
    // for topnodes
    sort(topnodes.begin(),topnodes.end());
    topnodes.erase(unique(topnodes.begin(),topnodes.end()),topnodes.end());

    // setup the physical group information
    Data.PhyGroupNum=1+4;
    Data.PhyGroupDimVec.resize(Data.PhyGroupNum,0);
    Data.PhyGroupIDVec.resize(Data.PhyGroupNum,0);
    Data.PhyGroupNameVec.resize(Data.PhyGroupNum);
    Data.PhyGroupElmtsNumVec.resize(Data.PhyGroupNum,0);

    // for physical dim vector
    Data.PhyGroupDimVec[0]=2;
    Data.PhyGroupDimVec[1]=1;
    Data.PhyGroupDimVec[2]=1;
    Data.PhyGroupDimVec[3]=1;
    Data.PhyGroupDimVec[4]=1;


    // for physical id vector
    Data.PhyGroupIDVec[0]=0;
    Data.PhyGroupIDVec[1]=1;
    Data.PhyGroupIDVec[2]=2;
    Data.PhyGroupIDVec[3]=3;
    Data.PhyGroupIDVec[4]=4;


    // for physical name vector
    Data.PhyGroupNameVec[0]="alldomain";
    Data.PhyGroupNameVec[1]="left";
    Data.PhyGroupNameVec[2]="right";
    Data.PhyGroupNameVec[3]="bottom";
    Data.PhyGroupNameVec[4]="top";


    // for phy group elmts num vector
    Data.PhyGroupElmtsNumVec[0]=Data.BulkElmtsNum;
    //
    Data.PhyGroupElmtsNumVec[1]=static_cast<int>(leftconn.size());
    Data.PhyGroupElmtsNumVec[2]=static_cast<int>(rightconn.size());
    Data.PhyGroupElmtsNumVec[3]=static_cast<int>(bottomconn.size());
    Data.PhyGroupElmtsNumVec[4]=static_cast<int>(topconn.size());

    /**
     * setup id<---->name map
     */
    // id--->name map
    Data.PhyGroupID2NameMap[0]="alldomain";
    Data.PhyGroupID2NameMap[1]="left";
    Data.PhyGroupID2NameMap[2]="right";
    Data.PhyGroupID2NameMap[3]="bottom";
    Data.PhyGroupID2NameMap[4]="top";
    // name--->id map
    Data.PhyGroupName2IDMap["alldomain"]=0;
    Data.PhyGroupName2IDMap["left"]=1;
    Data.PhyGroupName2IDMap["right"]=2;
    Data.PhyGroupName2IDMap["bottom"]=3;
    Data.PhyGroupName2IDMap["top"]=4;


    /**
     * Setup the global mapping, this should be nonzero only on master rank !!!
     */
    Data.PhyGroupName2ElmtConnMap["alldomain"]=Data.BulkElmtConn;
    //
    Data.PhyGroupName2ElmtConnMap["left"]=leftconn;
    Data.PhyGroupName2ElmtConnMap["right"]=rightconn;
    Data.PhyGroupName2ElmtConnMap["bottom"]=bottomconn;
    Data.PhyGroupName2ElmtConnMap["top"]=topconn;

    Data.PhyGroupID2ElmtConnMap[0]=Data.BulkElmtConn;
    //
    Data.PhyGroupID2ElmtConnMap[1]=leftconn;
    Data.PhyGroupID2ElmtConnMap[2]=rightconn;
    Data.PhyGroupID2ElmtConnMap[3]=bottomconn;
    Data.PhyGroupID2ElmtConnMap[4]=topconn;


    m_MeshGenerated=true;

    return m_MeshGenerated;
}