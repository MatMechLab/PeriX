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
//+++ Date    : 2026.07.28
//+++ Function: manuscript Gc-to-critical-stretch calibration.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <cmath>

namespace FractureCriterion {

/**
 * Classical critical-stretch calibration used in the manuscript:
 *
 *   3D:          s0^2 = 5 Gc / (6 E delta)
 *   plane stress:s0^2 = 4 pi Gc / (9 E delta)
 *   plane strain:s0^2 = 5 pi Gc / (12 E delta)
 *
 * A non-positive material value disables bond failure.
 */
[[nodiscard]] inline double s0SqFromGc(
    const double Gc,const double E,const double delta,
    const int dim,const bool planeStress) {
    if (Gc<=0.0 || E<=0.0 || delta<=0.0) return 0.0;
    if (dim>=3) return 5.0*Gc/(6.0*E*delta);
    if (planeStress) return 4.0*M_PI*Gc/(9.0*E*delta);
    return 5.0*M_PI*Gc/(12.0*E*delta);
}

[[nodiscard]] inline double criticalStretchFromGc(
    const double Gc,const double E,const double delta,
    const int dim,const bool planeStress) {
    return std::sqrt(s0SqFromGc(Gc,E,delta,dim,planeStress));
}

} // namespace FractureCriterion
