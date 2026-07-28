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
//+++ Function: the pd system class. Walks all PD bonds and
//+++           assembles the global residual / jacobian. The
//+++           per-bond and per-node physics is provided by
//+++           ElmtSystem. Dirichlet/Neumann conditions are
//+++           applied separately via BCSystem.
//+++
//+++           Newton convention used by the global linear solve:
//+++             K = -dF/dU,  RHS = +F,   K * dU = RHS.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <vector>

#include "ElmtSystem/ElmtSystem.h"
#include "ElmtSystem/LocalElmtInfo.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"


class PDSystem {
public:
    PDSystem()=default;
    ~PDSystem()=default;

    /**
     * assemble global residual and jacobian by looping every PD
     * node and every neighbor bond, dispatching the per-bond and
     * per-node physics through ElmtSystem.
     * @param Mesh         pd mesh
     * @param t_PDOperators pd operators (state mutated per-source-node)
     * @param t_ElmtSystem element registry
     * @param Info         per-call context (dt, t, step)
     * @param U            current global solution
     * @param Uold         previous-step solution (used by transient kernels;
     *                     for static analysis pass U or a zero vector)
     * @param K            output sparse matrix
     * @param RHS          output residual vector
     */
    void formResidualAndJacobian(const PDMesh &Mesh,
                                 PDOperators &t_PDOperators,
                                 const ElmtSystem &t_ElmtSystem,
                                 const LocalElmtInfo &Info,
                                 const VectorXd &U,
                                 const VectorXd &Uold,
                                 SparseMatrix &K,
                                 VectorXd &RHS);

    /**
     * OpenMP-parallel counterpart of formResidualAndJacobian. The outer
     * loop over source PD nodes is distributed across threads; every
     * contribution of source node i lands in row i of (K,RHS), so threads
     * own disjoint rows and need no locks on the assembled system. Each
     * thread keeps a private PDOperators copy and private local element
     * buffers (so the per-node/per-bond scratch is never shared), while the
     * frozen per-node caches inside the element kernels are populated by a
     * single serial preprocessIteration() call beforehand and only read
     * during assembly. The result is bitwise identical to the serial
     * routine because each row is accumulated by one thread in the same
     * bond order. Enabled at the call site by -DPARALLEL_ASSEMBLE=on.
     */
    void formResidualAndJacobianParallel(const PDMesh &Mesh,
                                         PDOperators &t_PDOperators,
                                         const ElmtSystem &t_ElmtSystem,
                                         const LocalElmtInfo &Info,
                                         const VectorXd &U,
                                         const VectorXd &Uold,
                                         SparseMatrix &K,
                                         VectorXd &RHS);

    /**
     * GPU (CUDA) counterpart of formResidualAndJacobian. Each PD source
     * node is handled by one device thread which builds the PDDO moment
     * matrix, inverts it, evaluates the per-bond operators and scatters
     * the element physics into row i of (K,RHS) -- row-disjoint, so no
     * atomics. Supported element kernels run on the device; any other
     * configuration (or a device error) transparently falls back to the
     * serial formResidualAndJacobian, so the result is always correct.
     * Built and called only when -DCUDA_ASSEMBLE=on (PERIX_CUDA_ASSEMBLE).
     */
    void formResidualAndJacobianCUDA(const PDMesh &Mesh,
                                     PDOperators &t_PDOperators,
                                     const ElmtSystem &t_ElmtSystem,
                                     const LocalElmtInfo &Info,
                                     const VectorXd &U,
                                     const VectorXd &Uold,
                                     SparseMatrix &K,
                                     VectorXd &RHS);

    /**
     * matrix-free EXPLICIT assembler: evaluate the nodal rate R_i of
     * du_i/dt = R_i(U) for every PD node and write it into Rate (no
     * Jacobian, no sparse matrix). Each source node i contributes only to
     * its own rows of Rate, exactly as the residual path does for (K,RHS).
     * Used by the forward-Euler integrator, which then advances
     * U^{n+1} = U^n + dt * Rate. Only explicit element kernels
     * (ElementBase::isExplicit()) contribute; Uold is forwarded to
     * preprocessIteration for kernels that cache per-node coefficients.
     */
    void formExplicitRate(const PDMesh &Mesh,
                          PDOperators &t_PDOperators,
                          const ElmtSystem &t_ElmtSystem,
                          const LocalElmtInfo &Info,
                          const VectorXd &U,
                          const VectorXd &Uold,
                          VectorXd &Rate);

    /**
     * matrix-free EXPLICIT-DYNAMICS assembler: evaluate the nodal
     * acceleration a_i = R_i of the second-order equation of motion
     * d2u_i/dt2 = R_i(U) for every PD node and write it into Rate. Unlike
     * formExplicitRate this is a nodal gather: the PDDO fracture element
     * evaluates its health-gated bond forces during preprocessIteration()
     * and caches the per-node
     * acceleration; the gather then just reads it back through
     * computeNodalResidual. No PDDO moment matrix is factorised here and
     * the family loop is not repeated, so there is no wasted work. Used by
     * the central-difference integrator, which advances
     * u^{n+1}=2u^n-u^{n-1}+dt^2*Rate.
     */
    void formExplicitDynamicsRate(const PDMesh &Mesh,
                                  PDOperators &t_PDOperators,
                                  const ElmtSystem &t_ElmtSystem,
                                  const LocalElmtInfo &Info,
                                  const VectorXd &U,
                                  const VectorXd &Uold,
                                  VectorXd &Rate);

private:
    VectorXd m_LocalR;
    VectorXd m_LocalR_node;
    MatrixXd m_LocalK_II;
    MatrixXd m_LocalK_IJ;
    MatrixXd m_LocalK_node;
    VectorXd m_U_I;
    VectorXd m_U_J;
    VectorXd m_Uold_I;
    VectorXd m_Uold_J;

    // --- forward-Euler PDDO operator cache (see FormExplicitRate.cpp) ---
    // The per-bond PDDO operator vector depends only on the reference geometry,
    // which is fixed for a transient explicit run, so it is identical at every
    // time step. Built once and replayed, trading O(bonds * opVecSize) doubles
    // of RAM for the per-step, per-node moment-matrix solve.
    bool m_FEOpCacheValid=false;        /**< cache populated and matches the mesh */
    int  m_FEOpCacheNodes=-1;           /**< node count the cache was built for */
    int  m_FEOpVecSize=-1;              /**< operator-vector length per bond */
    std::vector<long long> m_FEOpRowStart; /**< per-node offset into m_FEOpVals (size N+1) */
    std::vector<double>    m_FEOpVals;     /**< [total_bonds * opVecSize] cached operators */
};
