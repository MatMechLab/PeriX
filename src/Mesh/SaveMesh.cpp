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
//+++ Function: save mesh to vtu file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/Mesh.h"

void Mesh::saveMesh(const string &filename) {
    std::ofstream out;
    out.open(filename, std::ios::out);
    if (!out.is_open()) {
        cout<<"*** Error: can\'t create/open file="<<filename<<", please make sure you have the write permission"<<endl;
        abort();
    }
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
    out << "<UnstructuredGrid>\n";
    out << "<Piece NumberOfPoints=\"" << m_Data.NodesNum << "\" NumberOfCells=\"" << m_Data.BulkElmtsNum <<"\">\n";
    out << "<Points>\n";
    out << "<DataArray type=\"Float64\" Name=\"nodes\"  NumberOfComponents=\"3\"  format=\"ascii\">\n";

    //*****************************
    // print out node coordinates
    out << std::scientific << std::setprecision(6);
    for (int i = 0; i < m_Data.NodesNum; i++) {
        out << m_Data.NodeCoords[i * 3 + 1 - 1] << " "
            << m_Data.NodeCoords[i * 3 + 2 - 1] << " "
            << m_Data.NodeCoords[i * 3 + 3 - 1] << "\n";
    }
    out << "</DataArray>\n";
    out << "</Points>\n";

    //***************************************
    //*** For cell information
    //***************************************
    out << "<Cells>\n";
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (const auto &cell: m_Data.BulkElmtConn) {
        for (const auto &id: cell) {
            out << id - 1 << " ";
        }
        out << "\n";
    }
    out << "</DataArray>\n";

    //***************************************
    //*** For offset
    //***************************************
    out << "<DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    int offset = 0;
    for (int e=1;e<=m_Data.BulkElmtsNum;e++) {
        offset += m_Data.NodesNumPerBulkElmt;
        out << offset << "\n";
    }
    out << "</DataArray>\n";

    //***************************************
    //*** For vtk cell type
    //***************************************
    out << "<DataArray type=\"Int32\" Name=\"types\"  NumberOfComponents=\"1\"  format=\"ascii\">\n";
    for (int e=1;e<=m_Data.BulkElmtsNum;e++) {
        out << m_Data.BulkElmtVTKCellType << "\n";
    }
    out << "</DataArray>\n";
    out << "</Cells>\n";

    //***************************************
    //*** End of output
    //***************************************
    out << "</Piece>\n";
    out << "</UnstructuredGrid>\n";
    out << "</VTKFile>" << endl;

    out.close();
}
