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
//+++ Function: the mesh generator
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/MeshGenerator.h"
#include "Utils/MessagePrinter.h"


//********************************************
bool MeshGenerator::createMesh(const MeshType &meshtype,MeshData &Data){
    switch (meshtype)
    {
    // for 2d
    case MeshType::QUAD4:
        return Lagrange2DQuad4MeshGenerator::generateMesh(Data);
        break;
    // for 3d
    case MeshType::HEX8:
        return Lagrange3DHex8MeshGenerator::generateMesh(Data);
        break;
    default:
        MessagePrinter::printErrorTxt("Unsupported meshtype for MeshGenerator::createMesh");
        MessagePrinter::exitPeriX();
        break;
    }
    return true;
}