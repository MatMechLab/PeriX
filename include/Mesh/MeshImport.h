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
//+++ Date    : 2026.06.07
//+++ Function: import external mesh files into PeriX MeshData.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>

#include "Mesh/MeshData.h"

class MeshImport {
public:
    MeshImport()=default;
    ~MeshImport()=default;

    /**
     * Import an ASCII Gmsh MSH 4.1 mesh file into MeshData.
     *
     * Supported low-order element types:
     *   1: EDGE2, 2: TRI3, 3: QUAD4, 4: TET4, 5: HEX8, 15: POINT.
     * The highest-dimensional imported elements become PeriX bulk elements.
     */
    [[nodiscard]] bool importMSH(const std::string &fileName,MeshData &data) const;
};
