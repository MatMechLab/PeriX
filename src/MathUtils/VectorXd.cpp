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
//+++ Function: the vectorxd class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MathUtils/VectorXd.h"

void VectorXd::print()const{
    std::printf("*** ");
    for (int i = 1; i <= getSize(); ++i) {
        std::printf("%12.4e ", m_Vals[static_cast<std::size_t>(i - 1)]);
        if (i % 8 == 0) std::printf("\n*** ");
    }
    std::printf("\n");
}