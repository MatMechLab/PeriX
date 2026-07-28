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
//+++ Function: the mesh class of PeriX
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include "Mesh/MeshData.h"
#include "Mesh/Nodes.h"
#include "Utils/MessagePrinter.h"

using namespace std;

/**
 * the mesh class of PeriX
 */
class Mesh {
public:
    Mesh();

    /**
     * reset the mesh data
     */
    void resetMeshData();

    /**
     * save the mesh into a vtu file
     * @param filename mesh file name for vtu output
     */
    void saveMesh(const string &filename);

    //********************************************************
    //*** settings
    //********************************************************
    /**
     * setup mesh geometry info for 1d case
     * @param nx mesh number along x-axis
     * @param xmin xmin value of the domain
     * @param xmax xmax value of the domain
     * @param meshtype the given mesh type
    */
    void setMeshInfo(const int &nx,const double &xmin,const double &xmax,const MeshType &meshtype);

    /**
     * setup mesh geometry info for 2d case
     * @param nx mesh number along x-axis
     * @param ny mesh number along y-axis
     * @param xmin xmin value of the domain
     * @param xmax xmax value of the domain
     * @param ymin ymin value of the domain
     * @param ymax ymax value of the domain
     * @param meshtype the given mesh type
    */
    void setMeshInfo(const int &nx,const int &ny,
                     const double &xmin,const double &xmax,
                     const double &ymin,const double &ymax,const MeshType &meshtype);

    /**
     * setup mesh geometry info for 3d case
     * @param nx mesh number along x-axis
     * @param ny mesh number along y-axis
     * @param xmin xmin value of the domain
     * @param xmax xmax value of the domain
     * @param ymin ymin value of the domain
     * @param ymax ymax value of the domain
     * @param meshtype the given mesh type
    */
    void setMeshInfo(const int &nx,const int &ny,const int &nz,
                     const double &xmin,const double &xmax,
                     const double &ymin,const double &ymax,
                     const double &zmin,const double &zmax,const MeshType &meshtype);



    //********************************************************
    //*** for get funs
    //********************************************************
    /**
     * get the dim of current mesh
     * @return dim of current mesh
     */
    [[nodiscard]] int getMeshDim()const {
        return m_Data.MeshDim;
    }

    /**
     * get the order of current mesh
     * @return mesh order
     */
    [[nodiscard]] int getMeshOrder()const {
        return m_Data.MeshOrder;
    }

    /**
     * get the mesh type of bulk element
     * @return mesh type of bulk element
     */
    MeshType getBulkElmtMeshType()const {
        return m_Data.BulkElmtMeshType;
    }

    /**
     * get the mesh type of surface element
     * @return mesh type of surface element
     */
    MeshType getSurfElmtMeshType()const {
        return m_Data.SurfElmtMeshType;
    }

    /**
     * get the mesh type of line element
     * @return mesh type of line element
     */
    MeshType getLineElmtMeshType() const {
        return m_Data.LineElmtMeshType;
    }

    /**
     * get the total elements num
     * @return total elements num
     */
    [[nodiscard]] int getElmtsNum()const {
        return m_Data.ElmtsNum;
    }

    /**
     * get the bulk elements num
     * @return bulk elements num
     */
    [[nodiscard]] int getBulkElmtsNum()const {
        return m_Data.BulkElmtsNum;
    }

    /**
     * get the surf elements num
     * @return surf elements num
     */
    [[nodiscard]] int getSurfElmtsNum()const {
        return m_Data.SurfElmtsNum;
    }

    /**
     * get the line elements num
     * @return line elements num
     */
    [[nodiscard]] int getLineElmtsNum()const {
        return m_Data.LineElmtsNum;
    }

    /**
     * get point elements num
     * @return point elements num
     */
    [[nodiscard]] int getPointElmtsNum()const {
        return m_Data.PointElmtsNum;
    }

    /**
     * get the elements number of specific physical group
     * @param phyname string name of physical group
     * @return elements number
     */
    [[nodiscard]] int getElmtsNumViaPhyName(const string &phyname)const {
        if (m_Data.PhyGroupName2ElmtConnMap.contains(phyname)) {
            return static_cast<int>(m_Data.PhyGroupName2ElmtConnMap.at(phyname).size());
        }
        else {
            MessagePrinter::printErrorTxt("phyname="+phyname+" do not exist in your physical group, getElmtsNumViaPhyName fails");
            MessagePrinter::exitPeriX();
        }
        return -1;
    }

    /**
     * get the elements number of specific physical id
     * @param id id of physical group
     * @return elements number
     */
    [[nodiscard]] int getElmtsNumViaPhyID(const int &id)const {
        if (m_Data.PhyGroupID2ElmtConnMap.contains(id)) {
            return static_cast<int>(m_Data.PhyGroupID2ElmtConnMap.at(id).size());
        }
        else {
            MessagePrinter::printErrorTxt("phyid="+to_string(id)+" do not exist in your physical group, getElmtsNumViaPhyID fails");
            MessagePrinter::exitPeriX();
        }
        return -1;
    }

    /**
     * get the bulk element's id vector
     * @param phyname specific phyname
     * @return the bulk element's id vector
     */
    [[nodiscard]] vector<int> getBulkElmtIDVecViaPhyName(const string &phyname) const {
        if (m_Data.PhyGroupName2BulkElmtIDVecMap.contains(phyname)) {
            return m_Data.PhyGroupName2BulkElmtIDVecMap.at(phyname);
        }
        else {
            MessagePrinter::printErrorTxt("phyname="+phyname+" is not found in your meshdata, getBulkElmtIDVecViaPhyName fails");
            MessagePrinter::exitPeriX();
        }
        return vector<int>(0);
    }

    /**
     * get total nodes num
     * @return total nodes num
     */
    [[nodiscard]] int getNodesNum()const {
        return m_Data.NodesNum;
    }

    /**
     * get the nodes num per bulk element
     * @return nodes num per bulk element
     */
    [[nodiscard]] int getNodesNumPerBulkElmt()const {
        return m_Data.NodesNumPerBulkElmt;
    }
    /**
     * get the nodes num per surf element
     * @return nodes num per surf element
     */
    [[nodiscard]] int getNodesNumPerSurfElmt()const {
        return m_Data.NodesNumPerSurfElmt;
    }
    /**
     * get the nodes num per line element
     * @return nodes num per line element
     */
    [[nodiscard]] int getNodesNumPerLineElmt()const {
        return m_Data.NodesNumPerLineElmt;
    }

    /**
     * get the nodes num of i-th element
     * @param e element index
     * @return nodes num of i-th element
     */
    [[nodiscard]] int getIthElmtNodesNum(const int &e)const {
        if (e<1||e>m_Data.ElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of element's range");
            MessagePrinter::exitPeriX();
        }
        return static_cast<int>(m_Data.ElmtConn[e-1].size());
    }

    /**
     * get the connectivity of i-th element
     * @param e element index, start from 1
     * @return i-th element's connectivity
     */
    [[nodiscard]] vector<int> getIthElmtConn(const int &e)const {
        if (e<1||e>m_Data.ElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of element's range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.ElmtConn[e-1];
    }

    /**
     * get the i-th element's connectivity
     * @param e element index
     * @param elconn element connectivity
     */
    void getIthElmtConn(const int &e,vector<int> &elconn)const {
        if (e<1||e>m_Data.ElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of element's range");
            MessagePrinter::exitPeriX();
        }
        for (int i=0;i<getIthElmtNodesNum(e);i++) elconn[i]=m_Data.ElmtConn[e-1][i];
    }

    /**
     * get the connectivity of i-th bulk element
     * @param e bulk element index, start from 1
     * @return connectivity of i-th bulk element
     */
    [[nodiscard]] vector<int> getIthBulkElmtConn(const int &e)const {
        if (e<1||e>m_Data.BulkElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of bulk element's range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.BulkElmtConn[e-1];
    }

    /**
     * get the i-th bulk element's connectivity
     * @param e element index
     * @param elconn bulk element connectivity
     */
    void getIthBulkElmtConn(const int &e,vector<int> &elconn)const {
        if (e<1||e>m_Data.BulkElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of bulk element's range");
            MessagePrinter::exitPeriX();
        }
        for (int i=0;i<m_Data.NodesNumPerBulkElmt;i++) elconn[i]=m_Data.BulkElmtConn[e-1][i];
    }

    void getIthBulkElmtNodes(const int &e,Nodes &t_Nodes) const {
        if (e<1||e>m_Data.BulkElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(e)+" is out of bulk element's range");
            MessagePrinter::exitPeriX();
        }
        for (int i=1;i<=m_Data.NodesNumPerBulkElmt;i++) {
            t_Nodes(i,1)=getIthNodeJthCoord(m_Data.BulkElmtConn[e-1][i-1],1);
            t_Nodes(i,2)=getIthNodeJthCoord(m_Data.BulkElmtConn[e-1][i-1],2);
            t_Nodes(i,3)=getIthNodeJthCoord(m_Data.BulkElmtConn[e-1][i-1],3);
        }
    }

    /**
     * get i-th element's j-th nodeid
     * @param i i-th element index, start from 1
     * @param j j-th node index, start from 1
     * @return i-th element's j-th nodeid
     */
    [[nodiscard]] int getIthElmtJthNodeID(const int &i,const int &j)const {
        if (i<1||i>m_Data.ElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(i)+" is out of element's range");
            MessagePrinter::exitPeriX();
        }
        if (j<1||j>getIthElmtNodesNum(i)) {
            MessagePrinter::printErrorTxt("j="+to_string(j)+" is out of element's nodes number range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.ElmtConn[i-1][j-1];
    }

    /**
     * get i-th bulk element's j-th nodeid
     * @param i i-th bulk element index, start from 1
     * @param j j-th node index, start from 1
     * @return i-th bulk element's j-th nodeid
     */
    [[nodiscard]] int getIthBulkElmtJthNodeID(const int &i,const int &j)const {
        if (i<1||i>m_Data.BulkElmtsNum) {
            MessagePrinter::printErrorTxt("e="+to_string(i)+" is out of bulk element's range");
            MessagePrinter::exitPeriX();
        }
        if (j<1||j>m_Data.NodesNumPerBulkElmt) {
            MessagePrinter::printErrorTxt("j="+to_string(j)+" is out of bulk element's nodes number range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.BulkElmtConn[i-1][j-1];
    }

    /**
     * get the i-th node's j-th coord
     * @param i i-th node index, start from 1
     * @param j j-th coord index, start from 1~3
     * @return i-th node's j-th coord
     */
    [[nodiscard]] double getIthNodeJthCoord(const int &i,const int &j)const {
        return m_Data.NodeCoords[(i-1)*3+j-1];
    }

    /**
     * get the vtk cell type of bulk element
     * @return vtk cell type of bulk element
     */
    [[nodiscard]] int getBulkElmtVTKCellType()const {
        return m_Data.BulkElmtVTKCellType;
    }

    /**
     * get vtk cell type of line element
     * @return vtk cell type of line element
     */
    [[nodiscard]] int getLineElmtVTKCellType()const {
        return m_Data.LineElmtVTKCellType;
    }

    /**
     * get vtk cell type of surf element
     * @return vtk cell type of surf element
     */
    [[nodiscard]] int getSurfElmtVTKCellType()const {
        return m_Data.SurfElmtVTKCellType;
    }

    //********************************************************
    //*** for physical group info
    //********************************************************
    /**
     * get physical group number
     * @return physical group number
     */
    [[nodiscard]] int getPhyGroupNum()const {
        return m_Data.PhyGroupNum;
    }

    /**
     * get i-th phy group id
     * @param i index, start from 1
     * @return phy group id
     */
    [[nodiscard]] int getIthPhyGroupID(const int &i)const {
        if (i<1||i>m_Data.PhyGroupNum) {
            MessagePrinter::printErrorTxt("i="+to_string(i)+" is out of physical group number range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.PhyGroupIDVec[i-1];
    }

    /**
     * get i-th phy group dim
     * @param i index, start from 1
     * @return phy group dim
     */
    [[nodiscard]] int getIthPhyGroupDim(const int &i)const {
        if (i<1||i>m_Data.PhyGroupNum) {
            MessagePrinter::printErrorTxt("i="+to_string(i)+" is out of physical group number range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.PhyGroupDimVec[i-1];
    }

    /**
     * get the dim of specific phyname group
     * @param name phy name
     * @return the dim of specific phyname group
     */
    [[nodiscard]] int getPhyGroupDimViaPhyName(const string &name) const {
        for (int i=0;i<m_Data.PhyGroupNum;i++) {
            if (m_Data.PhyGroupNameVec[i]==name) return m_Data.PhyGroupDimVec[i];
        }
        MessagePrinter::printErrorTxt("can\'t find physical group name="+name+" for getPhyGroupDimViaPhyName");
        MessagePrinter::exitPeriX();
        return -1;
    }

    /**
     * get i-th phy group name
     * @param i index, start from 1
     * @return phy group name
     */
    [[nodiscard]] string getIthPhyGroupName(const int &i)const {
        if (i<1||i>m_Data.PhyGroupNum) {
            MessagePrinter::printErrorTxt("i="+to_string(i)+" is out of physical group number range");
            MessagePrinter::exitPeriX();
        }
        return m_Data.PhyGroupNameVec[i-1];
    }

    /**
     * get the element connectivity vector via specific physical group id
     * @param phyid physical group id
     * @return element connectivity vector
     */
    [[nodiscard]] vector<vector<int>> getElmtConnViaPhyID(const int &phyid)const {
        if (m_Data.PhyGroupID2ElmtConnMap.contains(phyid)) {
            return m_Data.PhyGroupID2ElmtConnMap.at(phyid);
        }
        else {
            MessagePrinter::printErrorTxt("can\'t find phyid="+to_string(phyid)+" in your PhyGroupID2ElmtConnMap");
            MessagePrinter::exitPeriX();
        }
        return vector<vector<int>>(0);
    }

    /**
     * get the element connectivity vector via specific physical group name
     * @param phyname physical group name
     * @return element connectivity vector
     */
    [[nodiscard]] vector<vector<int>> getElmtConnViaPhyName(const string &phyname)const {
        if (m_Data.PhyGroupName2ElmtConnMap.contains(phyname)) {
            return m_Data.PhyGroupName2ElmtConnMap.at(phyname);
        }
        MessagePrinter::printErrorTxt("can\'t find phyname="+phyname+" in your PhyGroupID2ElmtConnMap");
        MessagePrinter::exitPeriX();

        return vector<vector<int>>(0);
    }
    /**
     * get the element connectivity vector via specific physical group name
     * @param phyname physical group name
     * @return element connectivity vector
     */
    [[nodiscard]] vector<vector<int>> getBulkElmtConnViaPhyName(const string &phyname)const {
        if (m_Data.PhyGroupName2ElmtConnMap.contains(phyname)&&getPhyGroupDimViaPhyName(phyname)==getMeshDim()) {
            return m_Data.PhyGroupName2ElmtConnMap.at(phyname);
        }
        MessagePrinter::printErrorTxt("can\'t find phyname="+phyname+" in your getBulkElmtConnViaPhyName");
        MessagePrinter::exitPeriX();

        return vector<vector<int>>(0);
    }


    /**
     * check if a given string name is a valid name for boundary mesh
     * @param phyname input string name for physical group
     * @return true if phyname is a valid name for boundary mesh
     */
    [[nodiscard]] bool isBCElmtPhyNameValid(const string &phyname)const {
        for (int i=0;i<m_Data.PhyGroupNum;i++) {
            if (phyname==m_Data.PhyGroupNameVec[i] && m_Data.PhyGroupDimVec[i]<m_Data.MeshDim) {
                return true;
            }
        }
        MessagePrinter::printErrorTxt("phyname="+phyname+" is not a valid name for boundary element set");
        MessagePrinter::exitPeriX();
        return false;
    }

    /**
     * check if a given string name is a valid name for bulk mesh
     * @param phyname input string name for physical group
     * @return true if phyname is a valid name for bulk mesh
     */
    [[nodiscard]] bool isBulkElmtPhyNameValid(const string &phyname)const {
        for (int i=0;i<m_Data.PhyGroupNum;i++) {
            if (phyname==m_Data.PhyGroupNameVec[i] && m_Data.PhyGroupDimVec[i]==m_Data.MeshDim) {
                return true;
            }
        }
        MessagePrinter::printErrorTxt("phyname="+phyname+" is not a valid name for bulk element set");
        MessagePrinter::exitPeriX();
        return false;
    }


    /**
     * get the copy of mesh data
     * @return copy of mesh data
     */
    [[nodiscard]] MeshData getMeshDataCopy()const {
        return m_Data;
    }

    /**
     * get the reference of meshdata
     * @return reference of meshdata
     */
    MeshData& getMeshDataRef() {
        return m_Data;
    }

    /**
     * print out the mesh info
     */
    void printInfo()const;


private:
    MeshData m_Data;/**< the mesh data structure */
};