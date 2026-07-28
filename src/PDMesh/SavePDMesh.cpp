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
//+++ Function: save the pd mesh into vtu file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDMesh/PDMesh.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void PDMesh::savePDMesh(const string &inputfilename) {
    const string filename = inputfilename.substr(0,inputfilename.size()-5)+"-pdmesh.vtu";
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Error: cannot open file " << filename << '\n';
        return;
    }

    const std::size_t N = m_Data.NodesNum;

    // Use high precision for coordinates / weights
    out << std::setprecision(16);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    // -------------------------------------------------------------------------
    // Point data: scalar weight attached to each node
    // -------------------------------------------------------------------------
    out << "      <PointData Scalars=\"weight\">\n";
    out << "        <DataArray type=\"Int64\" Name=\"ElmtID\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i){
        out << m_Data.NodesElmtID[i]<<' ';
    }
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </PointData>\n";

    // Empty CellData block
    out << "      <CellData>\n";
    out << "      </CellData>\n";

    // -------------------------------------------------------------------------
    // Points: x y z for each node
    // -------------------------------------------------------------------------
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << m_Data.NodeCoords[i*3+0] << ' '
            << m_Data.NodeCoords[i*3+1] << ' '
            << m_Data.NodeCoords[i*3+2]<<"\n";
    }
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // -------------------------------------------------------------------------
    // Cells: one VTK_VERTEX cell per point
    //
    // connectivity: point index for each cell
    // offsets: cumulative number of point indices after each cell
    // types: VTK cell type, 1 = VTK_VERTEX
    // -------------------------------------------------------------------------
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << static_cast<long long>(i);
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    // offsets
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << static_cast<long long>(i + 1);
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    // types
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << 1; // VTK_VERTEX
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();
}
