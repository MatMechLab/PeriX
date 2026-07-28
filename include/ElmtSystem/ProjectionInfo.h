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
//+++ Date    : 2026.05.08
//+++ Function: descriptor for one user-selectable output projection
//+++           field. An element kernel publishes the list of
//+++           projections it can compute (strain, stress, flux,
//+++           gradient, ...). The output system uses this metadata
//+++           to validate the user's request and to size the VTU
//+++           DataArray correctly.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>

enum class ProjectionType {
    Scalar,   ///< 1 component per node (e.g. von Mises stress)
    Vector,   ///< 3 components per node (z-padded in 2D)
    Tensor    ///< 9 components per node, full row-major 3x3 (z-padded in 2D)
};

struct ProjectionInfo {
    std::string    Name;       ///< user-facing key in the OutputSystem JSON list
    ProjectionType Type;       ///< Scalar / Vector / Tensor
    int            Components; ///< 1, 3, or 9 (post z-padding for VTU output)
};

[[nodiscard]] inline int projectionComponentsFor(const ProjectionType &t) {
    switch (t) {
        case ProjectionType::Scalar: return 1;
        case ProjectionType::Vector: return 3;
        case ProjectionType::Tensor: return 9;
    }
    return 1;
}
