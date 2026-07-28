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
//+++ Function: Implement some commonly used mathematic functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MathUtils/MathFuns.h"

MathFuns::MathFuns(){}

double MathFuns::bracketPos(const double &x){
    return 0.5*(x+abs(x));
}
double MathFuns::bracketNeg(const double &x){
    return 0.5*(x-abs(x));
}
//***************************************
double MathFuns::sign(const double &x){
    return x >= 0.0 ? 1.0 : -1.0;
}