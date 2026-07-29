//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>

/**
 * Residual/Jacobian assembly backends documented by PeriX.
 *
 * Availability is checked by InputSystem against the build configuration;
 * this enum records the single backend selected for one run.
 */
enum class AssembleType {
    SERIAL,
    OPENMP,
    CUDA
};

[[nodiscard]] std::string assembleTypeName(AssembleType type);
