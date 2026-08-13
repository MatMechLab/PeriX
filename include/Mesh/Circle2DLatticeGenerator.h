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
//+++ Date    : 2026.07.08
//+++ Function: 2D concentric-ring peridynamic point-lattice for a disk
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "Mesh/PDLatticeGeneratorBase.h"

/**
 * 2D disk lattice: layer 0 is the centre disk, layer k>=1 is the ring
 * [(k-0.5)dx,(k+0.5)dx]. The n_k points of a layer sit evenly on the circle of
 * radius r_eff=(k+0.5)dx. The circumference-based raw count is rounded to
 * the nearest positive multiple of four, and points are offset by half an
 * angular step, giving every ring fourfold rotational symmetry without
 * placing points directly on the coordinate axes.
 */
class Circle2DLatticeGenerator final : public PDLatticeGeneratorBase {
protected:
    [[nodiscard]] int dim() const override { return 2; }
    [[nodiscard]] const char* shapeName() const override { return "circle"; }

    [[nodiscard]] int pointsOnShell(const int k,const double r_eff,const double dx) const override;
    [[nodiscard]] double shellMeasure(const int k,const double dx) const override;
    void placeShellPoints(const int k,const double r_eff,const int n,
                          const std::array<double,3> &c,
                          std::vector<std::array<double,3>> &pts) const override;
    [[nodiscard]] double boundarySurface(const double R) const override;
};
