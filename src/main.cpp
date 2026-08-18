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
//+++ Function: program entry point for the perix executable.
//+++           Prints the project banner with the current release
//+++           tag, constructs a PDProblem, and forwards command-
//+++           line arguments to PDProblem::run().
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Utils/ProjectBanner.h"
#include "PDProblem/PDProblem.h"
int main(int args,char *argv[]) {
    ProjectBanner::print("0.1");

    PDProblem problem;
    problem.run(args,argv);

    return 0;
}
