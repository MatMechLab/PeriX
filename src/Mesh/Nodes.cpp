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
//+++ Function: this class stores the coordinates of nodes of
//+++           a single element
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/Nodes.h"

Nodes::Nodes(){
    m_Coordinates.clear();m_Size=0;
}
Nodes::Nodes(const int &n){
    m_Size=n;m_Coordinates.resize(n*3,0.0);
}
Nodes::Nodes(const Nodes &nodes){
    m_Size=nodes.m_Size;
    m_Coordinates.clear();
    for(const auto &it:nodes.m_Coordinates) m_Coordinates.push_back(it);
}

Nodes::~Nodes(){
    m_Size=0;
    m_Coordinates.clear();
}