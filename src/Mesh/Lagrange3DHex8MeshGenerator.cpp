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
//+++ Function: the 3D hex8 lagrange mesh generator
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/Lagrange3DHex8MeshGenerator.h"

Lagrange3DHex8MeshGenerator::Lagrange3DHex8MeshGenerator(){
    m_MeshGenerated=false;
}
Lagrange3DHex8MeshGenerator::~Lagrange3DHex8MeshGenerator(){
    m_MeshGenerated=false;
}
//*********************************************************
bool Lagrange3DHex8MeshGenerator::generateMesh(MeshData &Data){
    Data.MeshDim=3;

    vector<vector<int>> leftconn,rightconn;
    vector<vector<int>> bottomconn,topconn;
    vector<vector<int>> backconn,frontconn;
    vector<int> leftnodes,rightnodes;
    vector<int> bottomnodes,topnodes;
    vector<int> backnodes,frontnodes;
    // only create mesh on the master rank, then distributed them into different ranks !
    double dx,dy,dz;
    int i,j,k,kk,e;
    int i1,i2,i3,i4,i5,i6,i7,i8;

    // generate the mesh cell for hex8 mesh
    Data.MeshOrder=1;
    Data.BulkMeshTypeName="hex8";
    dx=(Data.Xmax-Data.Xmin)/Data.Nx;
    dy=(Data.Ymax-Data.Ymin)/Data.Ny;
    dz=(Data.Zmax-Data.Zmin)/Data.Nz;

    Data.BulkElmtsNum=Data.Nx*Data.Ny*Data.Nz;
    Data.PointElmtsNum=0;
    Data.LineElmtsNum=0;
    Data.SurfElmtsNum=2*(Data.Nx*Data.Ny
                     +Data.Nx*Data.Nz
                     +Data.Ny*Data.Nz);
    Data.ElmtsNum=Data.BulkElmtsNum
                 +Data.SurfElmtsNum
                 +Data.LineElmtsNum
                 +Data.PointElmtsNum;

    Data.NodesNum=(Data.Nx+1)*(Data.Ny+1)*(Data.Nz+1);
    Data.NodesNumPerBulkElmt=8;
    Data.NodesNumPerSurfElmt=4;
    Data.NodesNumPerLineElmt=2;

    Data.BulkElmtVTKCellType=12;
    Data.BulkElmtMeshType=MeshType::HEX8;

    Data.LineElmtVTKCellType=3;
    Data.LineElmtMeshType=MeshType::EDGE2;

    Data.SurfElmtVTKCellType=9;
    Data.SurfElmtMeshType=MeshType::QUAD4;

    Data.NodeCoords.resize(Data.NodesNum*3,0.0);
    leftnodes.clear();rightnodes.clear();
    bottomnodes.clear();rightnodes.clear();
    backnodes.clear();frontnodes.clear();
    for(k=1;k<=Data.Nz+1;k++){
        for(j=1;j<=Data.Ny+1;j++){
            for(i=1;i<=Data.Nx+1;i++){
                kk=(j-1)*(Data.Nx+1)+i+(k-1)*(Data.Nx+1)*(Data.Ny+1);

                Data.NodeCoords[(kk-1)*3+1-1]=Data.Xmin+(i-1)*dx;
                Data.NodeCoords[(kk-1)*3+2-1]=Data.Ymin+(j-1)*dy;
                Data.NodeCoords[(kk-1)*3+3-1]=Data.Zmin+(k-1)*dz;
            }
        }
    }// end-of-node-generation

    // for the connectivity information of bulk elements
    vector<vector<int>> LeftBulkConn,RightBulkConn;
    vector<vector<int>> BottomBulkConn,TopBulkConn;
    vector<vector<int>> FrontBulkConn,BackBulkConn;
    Data.BulkElmtConn.resize(Data.BulkElmtsNum);
    leftconn.resize(Data.Ny*Data.Nz);
    rightconn.resize(Data.Ny*Data.Nz);
    LeftBulkConn.resize(Data.Ny*Data.Nz);
    RightBulkConn.resize(Data.Ny*Data.Nz);
    //
    bottomconn.resize(Data.Nx*Data.Nz);
    topconn.resize(Data.Nx*Data.Nz);
    BottomBulkConn.resize(Data.Nx*Data.Nz);
    TopBulkConn.resize(Data.Nx*Data.Nz);
    //
    backconn.resize(Data.Nx*Data.Ny);
    frontconn.resize(Data.Nx*Data.Ny);
    FrontBulkConn.resize(Data.Nx*Data.Ny);
    BackBulkConn.resize(Data.Nx*Data.Ny);

    leftnodes.clear();rightnodes.clear();
    bottomnodes.clear();rightnodes.clear();
    backnodes.clear();frontnodes.clear();

    Data.PhyGroupName2BulkElmtIDVecMap.clear();
    Data.PhyGroupID2BulkElmtIDVecMap.clear();
    Data.PhyGroupID2ElmtConnMap.clear();

    for(k=1;k<=Data.Nz;k++){
        for(j=1;j<=Data.Ny;j++){
            for(i=1;i<=Data.Nx;i++){
                e=(j-1)*Data.Nx+i+(k-1)*Data.Nx*Data.Ny;
                i1=(j-1)*(Data.Nx+1)+i+(k-1)*(Data.Nx+1)*(Data.Ny+1);
                i2=i1+1;
                i3=i2+Data.Nx+1;
                i4=i3-1;
                i5=i1+(Data.Nx+1)*(Data.Ny+1);
                i6=i2+(Data.Nx+1)*(Data.Ny+1);
                i7=i3+(Data.Nx+1)*(Data.Ny+1);
                i8=i4+(Data.Nx+1)*(Data.Ny+1);

                Data.PhyGroupName2BulkElmtIDVecMap["alldomain"].push_back(e);
                Data.PhyGroupID2BulkElmtIDVecMap[0].push_back(e);

                Data.BulkElmtConn[e-1].clear();
                Data.BulkElmtConn[e-1].push_back(i1);
                Data.BulkElmtConn[e-1].push_back(i2);
                Data.BulkElmtConn[e-1].push_back(i3);
                Data.BulkElmtConn[e-1].push_back(i4);
                Data.BulkElmtConn[e-1].push_back(i5);
                Data.BulkElmtConn[e-1].push_back(i6);
                Data.BulkElmtConn[e-1].push_back(i7);
                Data.BulkElmtConn[e-1].push_back(i8);

                if(i==1){
                    // for left bc elements
                    leftconn[(k-1)*Data.Ny+j-1].clear();
                    leftconn[(k-1)*Data.Ny+j-1].push_back(i1);
                    leftconn[(k-1)*Data.Ny+j-1].push_back(i5);
                    leftconn[(k-1)*Data.Ny+j-1].push_back(i8);
                    leftconn[(k-1)*Data.Ny+j-1].push_back(i4);

                    leftnodes.push_back(i1);
                    leftnodes.push_back(i5);
                    leftnodes.push_back(i8);
                    leftnodes.push_back(i4);

                    LeftBulkConn[(k-1)*Data.Ny+j-1]=Data.BulkElmtConn[e-1];
                }
                if(i==Data.Nx){
                    // for right bc elements
                    rightconn[(k-1)*Data.Ny+j-1].clear();
                    rightconn[(k-1)*Data.Ny+j-1].push_back(i2);
                    rightconn[(k-1)*Data.Ny+j-1].push_back(i3);
                    rightconn[(k-1)*Data.Ny+j-1].push_back(i7);
                    rightconn[(k-1)*Data.Ny+j-1].push_back(i6);

                    rightnodes.push_back(i2);
                    rightnodes.push_back(i3);
                    rightnodes.push_back(i7);
                    rightnodes.push_back(i6);

                    RightBulkConn[(k-1)*Data.Ny+j-1]=Data.BulkElmtConn[e-1];
                }
                if(j==1){
                    // for bottom bc elements
                    bottomconn[(k-1)*Data.Nx+i-1].clear();
                    bottomconn[(k-1)*Data.Nx+i-1].push_back(i1);
                    bottomconn[(k-1)*Data.Nx+i-1].push_back(i2);
                    bottomconn[(k-1)*Data.Nx+i-1].push_back(i6);
                    bottomconn[(k-1)*Data.Nx+i-1].push_back(i5);

                    bottomnodes.push_back(i1);
                    bottomnodes.push_back(i2);
                    bottomnodes.push_back(i6);
                    bottomnodes.push_back(i5);

                    BottomBulkConn[(k-1)*Data.Nx+i-1]=Data.BulkElmtConn[e-1];
                }
                if(j==Data.Ny){
                    // for top bc elements
                    topconn[(k-1)*Data.Nx+i-1].clear();
                    topconn[(k-1)*Data.Nx+i-1].push_back(i4);
                    topconn[(k-1)*Data.Nx+i-1].push_back(i8);
                    topconn[(k-1)*Data.Nx+i-1].push_back(i7);
                    topconn[(k-1)*Data.Nx+i-1].push_back(i3);

                    topnodes.push_back(i4);
                    topnodes.push_back(i8);
                    topnodes.push_back(i7);
                    topnodes.push_back(i3);

                    TopBulkConn[(k-1)*Data.Nx+i-1]=Data.BulkElmtConn[e-1];
                }
                if(k==1){
                    // for back bc elements
                    backconn[(j-1)*Data.Nx+i-1].clear();
                    backconn[(j-1)*Data.Nx+i-1].push_back(i1);
                    backconn[(j-1)*Data.Nx+i-1].push_back(i4);
                    backconn[(j-1)*Data.Nx+i-1].push_back(i3);
                    backconn[(j-1)*Data.Nx+i-1].push_back(i2);

                    backnodes.push_back(i1);
                    backnodes.push_back(i4);
                    backnodes.push_back(i3);
                    backnodes.push_back(i2);

                    BackBulkConn[(j-1)*Data.Nx+i-1]=Data.BulkElmtConn[e-1];
                }
                if(k==Data.Nz){
                    // for front bc elements
                    frontconn[(j-1)*Data.Nx+i-1].clear();
                    frontconn[(j-1)*Data.Nx+i-1].push_back(i5);
                    frontconn[(j-1)*Data.Nx+i-1].push_back(i6);
                    frontconn[(j-1)*Data.Nx+i-1].push_back(i7);
                    frontconn[(j-1)*Data.Nx+i-1].push_back(i8);

                    frontnodes.push_back(i5);
                    frontnodes.push_back(i6);
                    frontnodes.push_back(i7);
                    frontnodes.push_back(i8);

                    FrontBulkConn[(j-1)*Data.Nx+i-1]=Data.BulkElmtConn[e-1];
                }
            }
        }
    }// end-of-element-generation-loop

    Data.ElmtConn.clear();
    for (const auto& cell:leftconn) Data.ElmtConn.push_back(cell);
    for (const auto& cell:rightconn) Data.ElmtConn.push_back(cell);
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
    // for backnodes
    sort(backnodes.begin(),backnodes.end());
    backnodes.erase(unique(backnodes.begin(),backnodes.end()),backnodes.end());
    // for frontnodes
    sort(frontnodes.begin(),frontnodes.end());
    frontnodes.erase(unique(frontnodes.begin(),frontnodes.end()),frontnodes.end());

    // setup the physical group information
    Data.PhyGroupNum=1+6+6;
    Data.PhyGroupDimVec.resize(Data.PhyGroupNum,0);
    Data.PhyGroupIDVec.resize(Data.PhyGroupNum,0);
    Data.PhyGroupNameVec.resize(Data.PhyGroupNum);
    Data.PhyGroupElmtsNumVec.resize(Data.PhyGroupNum,0);

    // for physical dim vector
    Data.PhyGroupDimVec[0]=3;
    Data.PhyGroupDimVec[1]=3;
    Data.PhyGroupDimVec[2]=3;
    Data.PhyGroupDimVec[3]=3;
    Data.PhyGroupDimVec[4]=3;
    Data.PhyGroupDimVec[5]=3;
    Data.PhyGroupDimVec[6]=3;
    //
    Data.PhyGroupDimVec[ 7]=2;
    Data.PhyGroupDimVec[ 8]=2;
    Data.PhyGroupDimVec[ 9]=2;
    Data.PhyGroupDimVec[10]=2;
    Data.PhyGroupDimVec[11]=2;
    Data.PhyGroupDimVec[12]=2;

    // for physical id vector
    Data.PhyGroupIDVec[0]=0;
    Data.PhyGroupIDVec[1]=1;
    Data.PhyGroupIDVec[2]=2;
    Data.PhyGroupIDVec[3]=3;
    Data.PhyGroupIDVec[4]=4;
    Data.PhyGroupIDVec[5]=5;
    Data.PhyGroupIDVec[6]=6;
    //
    Data.PhyGroupIDVec[7]=7;
    Data.PhyGroupIDVec[8]=8;
    Data.PhyGroupIDVec[9]=9;
    Data.PhyGroupIDVec[10]=10;
    Data.PhyGroupIDVec[11]=11;
    Data.PhyGroupIDVec[12]=12;

    // for physical name vector
    Data.PhyGroupNameVec[0]="alldomain";
    Data.PhyGroupNameVec[1]="leftvolume";
    Data.PhyGroupNameVec[2]="rightvolume";
    Data.PhyGroupNameVec[3]="bottomvolume";
    Data.PhyGroupNameVec[4]="topvolume";
    Data.PhyGroupNameVec[5]="backvolume";
    Data.PhyGroupNameVec[6]="frontvolume";
    //
    Data.PhyGroupNameVec[ 7]="left";
    Data.PhyGroupNameVec[ 8]="right";
    Data.PhyGroupNameVec[ 9]="bottom";
    Data.PhyGroupNameVec[10]="top";
    Data.PhyGroupNameVec[11]="back";
    Data.PhyGroupNameVec[12]="front";

    // for phy group elmts num vector
    Data.PhyGroupElmtsNumVec[0]=Data.BulkElmtsNum;
    Data.PhyGroupElmtsNumVec[1]=static_cast<int>(LeftBulkConn.size());
    Data.PhyGroupElmtsNumVec[2]=static_cast<int>(RightBulkConn.size());
    Data.PhyGroupElmtsNumVec[3]=static_cast<int>(BottomBulkConn.size());
    Data.PhyGroupElmtsNumVec[4]=static_cast<int>(TopBulkConn.size());
    Data.PhyGroupElmtsNumVec[5]=static_cast<int>(BackBulkConn.size());
    Data.PhyGroupElmtsNumVec[6]=static_cast<int>(FrontBulkConn.size());
    //
    Data.PhyGroupElmtsNumVec[ 7]=static_cast<int>(leftconn.size());
    Data.PhyGroupElmtsNumVec[ 8]=static_cast<int>(rightconn.size());
    Data.PhyGroupElmtsNumVec[ 9]=static_cast<int>(bottomconn.size());
    Data.PhyGroupElmtsNumVec[10]=static_cast<int>(topconn.size());
    Data.PhyGroupElmtsNumVec[11]=static_cast<int>(backconn.size());
    Data.PhyGroupElmtsNumVec[12]=static_cast<int>(frontconn.size());

    /**
     * setup id<---->name map
     */
    // id--->name map
    Data.PhyGroupID2NameMap[0]="alldomain";
    Data.PhyGroupID2NameMap[1]="leftvolume";
    Data.PhyGroupID2NameMap[2]="rightvolume";
    Data.PhyGroupID2NameMap[3]="bottomvolume";
    Data.PhyGroupID2NameMap[4]="topvolume";
    Data.PhyGroupID2NameMap[5]="backvolume";
    Data.PhyGroupID2NameMap[6]="frontvolume";
    //
    Data.PhyGroupID2NameMap[ 7]="left";
    Data.PhyGroupID2NameMap[ 8]="right";
    Data.PhyGroupID2NameMap[ 9]="bottom";
    Data.PhyGroupID2NameMap[10]="top";
    Data.PhyGroupID2NameMap[11]="back";
    Data.PhyGroupID2NameMap[12]="front";
    // name--->id map
    Data.PhyGroupName2IDMap["alldomain"]=0;
    Data.PhyGroupName2IDMap["leftvolume"]=1;
    Data.PhyGroupName2IDMap["rightvolume"]=2;
    Data.PhyGroupName2IDMap["bottomvolume"]=3;
    Data.PhyGroupName2IDMap["topvolume"]=4;
    Data.PhyGroupName2IDMap["backvolume"]=5;
    Data.PhyGroupName2IDMap["frontvolume"]=6;
    //
    Data.PhyGroupName2IDMap["left"]=7;
    Data.PhyGroupName2IDMap["right"]=8;
    Data.PhyGroupName2IDMap["bottom"]=9;
    Data.PhyGroupName2IDMap["top"]=10;
    Data.PhyGroupName2IDMap["back"]=11;
    Data.PhyGroupName2IDMap["front"]=12;

    /**
     * Setup the global mapping, this should be nonzero only on master rank !!!
     */
    Data.PhyGroupName2ElmtConnMap["alldomain"]=Data.BulkElmtConn;
    Data.PhyGroupName2ElmtConnMap["leftvolume"]=LeftBulkConn;
    Data.PhyGroupName2ElmtConnMap["rightvolume"]=RightBulkConn;
    Data.PhyGroupName2ElmtConnMap["bottomvolume"]=BottomBulkConn;
    Data.PhyGroupName2ElmtConnMap["topvolume"]=TopBulkConn;
    Data.PhyGroupName2ElmtConnMap["backvolume"]=BackBulkConn;
    Data.PhyGroupName2ElmtConnMap["frontvolume"]=FrontBulkConn;
    //
    Data.PhyGroupName2ElmtConnMap["left"]=leftconn;
    Data.PhyGroupName2ElmtConnMap["right"]=rightconn;
    Data.PhyGroupName2ElmtConnMap["bottom"]=bottomconn;
    Data.PhyGroupName2ElmtConnMap["top"]=topconn;
    Data.PhyGroupName2ElmtConnMap["back"]=backconn;
    Data.PhyGroupName2ElmtConnMap["front"]=frontconn;

    //
    Data.PhyGroupID2ElmtConnMap[0]=Data.BulkElmtConn;
    Data.PhyGroupID2ElmtConnMap[1]=LeftBulkConn;
    Data.PhyGroupID2ElmtConnMap[2]=RightBulkConn;
    Data.PhyGroupID2ElmtConnMap[3]=BottomBulkConn;
    Data.PhyGroupID2ElmtConnMap[4]=TopBulkConn;
    Data.PhyGroupID2ElmtConnMap[5]=BackBulkConn;
    Data.PhyGroupID2ElmtConnMap[6]=FrontBulkConn;
    //
    Data.PhyGroupID2ElmtConnMap[ 7]=leftconn;
    Data.PhyGroupID2ElmtConnMap[ 8]=rightconn;
    Data.PhyGroupID2ElmtConnMap[ 9]=bottomconn;
    Data.PhyGroupID2ElmtConnMap[10]=topconn;
    Data.PhyGroupID2ElmtConnMap[11]=backconn;
    Data.PhyGroupID2ElmtConnMap[12]=frontconn;


    m_MeshGenerated=true;

    return m_MeshGenerated;
}