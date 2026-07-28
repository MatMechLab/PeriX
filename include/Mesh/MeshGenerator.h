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
//+++ Function: the mesh generator class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "Mesh/Lagrange2DQuad4MeshGenerator.h"
//
#include "Mesh/Lagrange3DHex8MeshGenerator.h"

/**
 * This class offers the mesh generator for default mesh in AsFem
*/
class MeshGenerator:public Lagrange2DQuad4MeshGenerator,
                    //
                    public Lagrange3DHex8MeshGenerator{
public:
    /**
     * Constructor
    */
    MeshGenerator()=default;

    /**
     * create the finite element mesh cell and store them in meshdata
     * @param meshtype the mesh type used for mesh generation
     * @param Data the FECell data structure
    */
    bool createMesh(const MeshType &meshtype,MeshData &Data);
};