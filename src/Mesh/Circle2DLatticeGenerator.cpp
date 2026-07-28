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

#include "Mesh/Circle2DLatticeGenerator.h"

#include <algorithm>
#include <cmath>

int Circle2DLatticeGenerator::pointsOnShell(const int,const double r_eff,const double dx) const {
    // capacity = circumference of the shell circle; nearest integer of
    // capacity/dx points (>=1).
    return std::max(1,static_cast<int>(std::lround(2.0*kPi*r_eff/dx)));
}

double Circle2DLatticeGenerator::shellMeasure(const int k,const double dx) const {
    if (k==0) {
        const double r=0.5*dx;                       // centre disk radius
        return kPi*r*r;
    }
    const double r_in =(static_cast<double>(k)-0.5)*dx;
    const double r_out=(static_cast<double>(k)+0.5)*dx;
    return kPi*(r_out*r_out-r_in*r_in);              // ring area
}

void Circle2DLatticeGenerator::placeShellPoints(const int,const double r_eff,const int n,
                                                const std::array<double,3> &c,
                                                std::vector<std::array<double,3>> &pts) const {
    // n points evenly spaced on the circle, starting at angle 0 (matches the
    // reference driver gen_pd_lattice.py: linspace(0,2*pi,n,endpoint=False)).
    for (int i=0;i<n;i++) {
        const double theta=2.0*kPi*static_cast<double>(i)/static_cast<double>(n);
        pts.push_back({c[0]+r_eff*std::cos(theta),
                       c[1]+r_eff*std::sin(theta),
                       c[2]});
    }
}

double Circle2DLatticeGenerator::boundarySurface(const double R) const {
    return 2.0*kPi*R; // circumference
}
