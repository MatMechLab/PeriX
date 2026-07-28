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

#pragma once

#include "Mesh/MeshGeneratorBase.h"

/**
 * the cell generator for 2d lagrange mesh
 */
class Lagrange2DQuad4MeshGenerator:public MeshGeneratorBase{
public:
    /**
     * constructor
     */
    Lagrange2DQuad4MeshGenerator();
    /**
     * deconstructor
     */
    ~Lagrange2DQuad4MeshGenerator() override;

    /**
     * function for the details of 3d lagrange mesh generation, if everything works fine, it will return true.
     * @param Data the fe cell data structure, which should be updated within each cell generator!
     */
    bool generateMesh(MeshData &Data) final;
private:
    bool m_MeshGenerated=false;

};