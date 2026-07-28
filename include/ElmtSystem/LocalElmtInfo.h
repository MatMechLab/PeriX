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
//+++ Function: per-call context passed to element kernels.
//+++           Steady-state callers can leave Dt/T/Timestep at 0.
//+++           Inside a per-bond call, Xsi[]/XsiNorm hold the
//+++           reference-frame bond vector X_j - X_i and its
//+++           Euclidean length.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

struct LocalElmtInfo {
    double Dt=0.0;            /**< current time-step size (0 for static analysis) */
    double T=0.0;             /**< current physical time */
    int    Timestep=0;        /**< 1-based time-step index (0 for static analysis) */
    double Xsi[3]={0,0,0};    /**< per-bond reference vector X_j - X_i */
    double XsiNorm=0.0;       /**< per-bond reference vector length |X_j - X_i| */
    bool   UseParallel=false; /**< run the matrix-free explicit assembly (and the
                               *   explicit kernels' per-node preprocessIteration
                               *   loops) with OpenMP. Set by the explicit time
                               *   integrators from JobSystem.assemble==openmp;
                               *   the worker width is OMP_NUM_THREADS. Defaults
                               *   to false so every other caller stays serial and
                               *   the result is bitwise identical to the serial
                               *   path (the parallel loop carries no cross-node
                               *   reduction). */
    bool   UseCuda=false;     /**< run the matrix-free explicit assembly on the GPU.
                               *   Set by the explicit-dynamics integrators from
                               *   JobSystem.assemble==cuda; only kernels with a
                               *   device port act on it (others use the CPU path).
                               *   UseParallel is also set so any CPU fallback runs
                               *   with OpenMP. */
};
