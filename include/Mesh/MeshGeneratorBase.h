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
//+++ Function: the base class for mesh generator
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <algorithm>

#include "Mesh/MeshData.h"

using std::sort;

/**
 * this is the base class for fe cell(mesh) generator, all the generator should inherit this base class
 */
class MeshGeneratorBase{
public:
    virtual ~MeshGeneratorBase() = default;

    /**
     * constructor
     */
    MeshGeneratorBase()= default;

    /**
     * virtual function for the details of different cell(mesh) generation, the child class should
     * offer the implementations
     * @param t_MeshData the fe cell data structure, which should be updated within each cell generator!
     */
    virtual bool generateMesh(MeshData &t_MeshData)=0;

};