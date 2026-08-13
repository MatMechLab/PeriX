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
//+++ Function: abstract base for the concentric-shell peridynamic
//+++           point-lattice generators (2D disk / 3D ball).
//+++
//+++   The radius R is split into N layers of thickness dx=R/N.
//+++   Layer 0 is the centre disk/ball; layer k>=1 is the ring/shell
//+++   [(k-0.5)dx, (k+0.5)dx]. On each layer the points sit on the
//+++   sphere of radius r_eff=(k+0.5)dx; the number of points is the
//+++   nearest integer to (shell "capacity")/dx (perimeter/dx in 2D,
//+++   surface/dx^2 in 3D), subject to a concrete generator's symmetry
//+++   requirements, and each point carries the shell measure
//+++   (area/volume) divided by that count as its tributary volume.
//+++   (see the reference driver gen_pd_lattice.py)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <array>
#include <string>
#include <vector>

#include "Mesh/MeshData.h"

/**
 * Abstract base for the peridynamic concentric-lattice generators. The common
 * driver generate() walks the N shells, asks the concrete subclass for the
 * per-shell point count, measure and point placement, and fills t_MeshData as
 * an imported-style PD point cloud -- one bulk PD point per lattice site with
 * its tributary area/volume -- plus (optionally) an outer boundary group that
 * PDMesh::createPDMesh turns into a ring/shell of radial ghost points so
 * traction/pressure BCs can be applied.
 *
 * To support a new lattice geometry, inherit this class and implement the four
 * hooks (dim / shapeName / pointsOnShell / shellMeasure / placeShellPoints).
 */
class PDLatticeGeneratorBase {
public:
    struct Params {
        double R=1.0;                       /**< outer radius (>0) */
        int    N=20;                        /**< number of concentric layers (dx=R/N, >=1) */
        std::array<double,3> center{0.0,0.0,0.0}; /**< lattice centre */
        bool   makeBoundaryGroup=true;      /**< build the outer ghost boundary group */
        std::string boundaryName="outer";   /**< base name -> "<name>nodes_ghost"/"<name>nodes" */
    };

    virtual ~PDLatticeGeneratorBase()=default;

    /** Build the lattice and populate t_MeshData (point cloud + optional outer
     *  boundary group). Returns false on an invalid parameter set. */
    [[nodiscard]] bool generate(const Params &params,MeshData &t_MeshData) const;

protected:
    // ---- geometry hooks implemented by the concrete lattice ----
    [[nodiscard]] virtual int dim() const =0;
    [[nodiscard]] virtual const char* shapeName() const =0;

    /** number of points on shell k (mid radius r_eff): derived from the shell
     *  "capacity" over the spacing -- perimeter/dx in 2D, sphere surface/dx^2
     *  in 3D -- and any concrete-generator symmetry rule. Never fewer than 1. */
    [[nodiscard]] virtual int pointsOnShell(const int k,const double r_eff,const double dx) const =0;

    /** measure of shell k: the centre disk/ball area/volume for k==0, else the
     *  ring/spherical-shell measure of [(k-0.5)dx,(k+0.5)dx]. */
    [[nodiscard]] virtual double shellMeasure(const int k,const double dx) const =0;

    /** append the n evenly distributed points of shell k (radius r_eff, centred
     *  at c) to pts. */
    virtual void placeShellPoints(const int k,const double r_eff,const int n,
                                  const std::array<double,3> &c,
                                  std::vector<std::array<double,3>> &pts) const =0;

    /** total measure of the outer boundary at radius R (perimeter in 2D, sphere
     *  surface area in 3D); used to size each outer ghost's tributary boundary
     *  measure for conservative traction. */
    [[nodiscard]] virtual double boundarySurface(const double R) const =0;

protected:
    static constexpr double kPi=3.141592653589793238462643383279502884;
};
