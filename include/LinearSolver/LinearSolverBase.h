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
//+++ Function: the base class of linear solver
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>

#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"

#include "nlohmann/json.hpp"

using std::string;

/**
 * Abstract class for the linear solver
 */
class LinearSolverBase {
public:
    /**
     * deconstructor
     */
    virtual ~LinearSolverBase() = default;
    /**
     * init the linear solver via the input sparse matrix
     * @param K sparse matrix
     * @param Parameters json parameters read from input file
     */
    virtual void initSolver(SparseMatrix &K,nlohmann::ordered_json &Parameters)=0;
    /**
     * Solve the linear equation Ax=b, where A is the sparse matrix, b is the right hand side vector, and x is the solution
     * @param A the input sparse matrix
     * @param b the rand hand side vector
     * @param x the solution vector
     * @return if success then return true, otherwise return false
     */
    virtual bool solveLinearSystem(SparseMatrix &A,VectorXd &b,VectorXd &x)=0;
};