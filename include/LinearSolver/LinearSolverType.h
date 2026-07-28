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
//+++ Purpose: enum class for the linear solver type definition
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

/**
 * enum class for linear solver definition
 */
enum class LinearSolverType {
    DEFAULT, /**< PeriX's in-house profile LDU direct solver (default) */
    PARDISO, /**< Intel MKL PARDISO */
    CUDSS    /**< NVIDIA cuDSS */
};
