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
//+++ Function: the linear solver dispatcher class. Owns one
//+++           instance of each available backend (DefaultSolver
//+++           always; PardisoSolver only when PeriX is built
//+++           with -DUSE_ONEAPI=on, gated by PERIX_HAS_PARDISO)
//+++           and forwards init()/solve() to the active one.
//+++           DefaultSolver is the default backend; Pardiso is
//+++           selected explicitly via setSolverType().
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "LinearSolver/LinearSolverType.h"
#include "LinearSolver/DefaultSolver.h"
#ifdef PERIX_HAS_PARDISO
#include "LinearSolver/PardisoSolver.h"
#endif
#ifdef PERIX_HAS_CUDSS
#include "LinearSolver/CudssSolver.h"
#endif
#ifdef PERIX_HAS_AMGCL
#include "LinearSolver/AmgclSolver.h"
#endif

/**
 * The linear solver dispatcher class
 */
class LinearSolver {
public:
    /**
     * constructor (defaults to the in-house profile LDU solver)
     */
    LinearSolver();

    /**
     * set the linear solver type
     * @param type linear solver type
     */
    void setSolverType(const LinearSolverType &type);

    /**
     * set solver parameters
     * @param Params json parameters
     */
    void setSolverParameters(nlohmann::ordered_json &Params) {
        m_Parameters=Params;
    }

    /**
     * init the linear solver using the given sparse matrix
     * @param A sparse matrix
     */
    void init(SparseMatrix &A);

    /**
     * solve the linear equation system
     * @param A sparse matrix
     * @param rhs right hand side vector
     * @param x solution vector
     * @return true if everything is ok, otherwise false
     */
    bool solve(SparseMatrix &A,VectorXd &rhs,VectorXd &x);

    /**
     * get the linear solver type
     * @return linear solver type
     */
    [[nodiscard]] LinearSolverType getSolverType() const {
        return m_SolverType;
    }

    /**
     * get the string name of current linear solver
     * @return string name of current linear solver
     */
    [[nodiscard]] string getLinearSolverName()const {
        return m_LinearSolverName;
    }

    /** get the JSON parameters currently registered for the active
     *  backend (used by printLinearSolverInfo to echo them back). */
    [[nodiscard]] const nlohmann::ordered_json& getSolverParameters() const {
        return m_Parameters;
    }

    /**
     * print a one-block summary of the selected backend and its
     * parameter overrides.
     */
    void printLinearSolverInfo() const;

private:
    LinearSolverType m_SolverType;/**< active linear solver backend */
    string m_LinearSolverName;/**< string name of active backend */
    nlohmann::ordered_json m_Parameters;/**< json parameters read from input file */

    DefaultSolver m_DefaultSolver;/**< in-house profile LDU backend */
#ifdef PERIX_HAS_PARDISO
    PardisoSolver m_PardisoSolver;/**< Intel MKL PARDISO backend */
#endif
#ifdef PERIX_HAS_CUDSS
    CudssSolver m_CudssSolver;/**< NVIDIA cuDSS GPU backend */
#endif
#ifdef PERIX_HAS_AMGCL
    AmgclSolver m_AmgclSolver;/**< AMGCL CPU/OpenMP iterative backend */
#endif
};
