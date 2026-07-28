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
//+++ Function: abstract base class for a single PD element kernel
//+++           (a "model" in PeriX terminology). Each kernel
//+++           describes the per-bond and per-node residual /
//+++           jacobian contribution of one physical model.
//+++
//+++           Sign convention for derived classes:
//+++             LocalR(a)        = +F_i^a (strong-form residual
//+++                                contribution from this bond/node)
//+++             LocalK_II(a,b)   = dR_i^a / du_i^b
//+++             LocalK_IJ(a,b)   = dR_i^a / du_j^b
//+++           PDSystem applies the Newton sign flip during global
//+++           assembly (K = -dF/dU, RHS = +F).
//+++
//+++           For transient kernels, the previous-step solution is
//+++           supplied via U_old_I / U_old_J (per-bond) and
//+++           U_old_I (per-node). Time-step / step-index info is
//+++           in LocalElmtInfo.
//+++
//+++           Explicit (forward-Euler) kernels override isExplicit()
//+++           to return true and supply ONLY the residual via
//+++           computeBondResidual / computeNodalResidual -- no
//+++           Jacobian. For these the residual IS the rate R_i of
//+++           du_i/dt = R_i(u); the forward-Euler integrator then
//+++           advances u^{n+1} = u^n + dt * R_i(u^n). They do not
//+++           need to implement computeBondResidualAndJacobian.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>
#include <vector>

#include "ElmtSystem/LocalElmtInfo.h"
#include "ElmtSystem/ProjectionInfo.h"
#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDOperators/PDOperators.h"
#include "Utils/MessagePrinter.h"

class PDMesh;

class ElementBase {
public:
    ElementBase()=default;
    virtual ~ElementBase()=default;

    /**
     * number of degrees of freedom carried per PD node by this element
     */
    [[nodiscard]] virtual int getDofsPerNode() const = 0;

    /**
     * unique element type tag, e.g. "poisson", "diffusion"
     */
    [[nodiscard]] virtual std::string getElementType() const = 0;

    /**
     * canonical per-node DoF names in the order this kernel expects
     * them in the global solution vector. Used so users can refer to
     * DoFs by name (e.g. "ux", "c") in the BC and IC blocks instead of
     * by 1-based slot index. Length must equal getDofsPerNode().
     */
    [[nodiscard]] virtual std::vector<std::string> getDofNames() const = 0;

    /**
     * is this an explicit (forward-Euler) kernel? Explicit kernels skip
     * the Jacobian entirely: they implement computeBondResidual /
     * computeNodalResidual (the rate R in du/dt = R) and are advanced by
     * the matrix-free forward-Euler integrator. Default: false (implicit
     * backward-Euler / steady kernel that supplies a Jacobian).
     */
    [[nodiscard]] virtual bool isExplicit() const { return false; }

    /**
     * order of the time derivative this kernel models:
     *   1 -> first order  du/dt   = R(u)   (forward Euler; diffusion-type)
     *   2 -> second order d2u/dt2 = R(u)   (central difference; the
     *        elastodynamic equation of motion rho*u'' = div(sigma)+b,
     *        with R the acceleration a=(L+b)/rho returned by the kernel)
     * Only consulted for explicit kernels (isExplicit()==true); it tells
     * the driver whether to advance with the forward-Euler integrator
     * (solveExplicit) or the central-difference one (solveExplicitDynamics).
     * Default 1 keeps every existing explicit kernel on forward Euler.
     */
    [[nodiscard]] virtual int getTimeOrder() const { return 1; }

    /**
     * per-DoF time-derivative order, one entry per DoF in getDofNames() order.
     * Most kernels are uniform (every DoF the same order) and inherit the
     * default, which broadcasts getTimeOrder() across all DoFs. A COUPLED
     * explicit kernel can be mixed-order, e.g. chemo-mechanics carries a
     * first-order concentration (du/dt=R, forward Euler) alongside second-order
     * displacements (rho*u''=L, velocity-Verlet). When the returned mask mixes
     * 1 and 2 the driver selects the velocity-Verlet integrator, which advances
     * each DoF with the rule matching its own order. Length must equal
     * getDofsPerNode(); only consulted for explicit kernels.
     */
    [[nodiscard]] virtual std::vector<int> getDofTimeOrders() const {
        return std::vector<int>(static_cast<std::size_t>(getDofsPerNode()), getTimeOrder());
    }

    /**
     * quasi-static (inertia-free) explicit kernel: the momentum balance
     * div(sigma)+b=0 is reached not by time marching but by Adaptive Dynamic
     * Relaxation (ADR) -- a damped fictitious-mass pseudo-dynamics whose steady
     * state is the static equilibrium. Only consulted for explicit kernels; when
     * true the driver selects TimeStepping::solveADR instead of the
     * forward-Euler / central-difference integrators. The kernel returns the
     * fictitious nodal acceleration (internal force density divided by the ADR
     * fictitious mass) from computeNodalResidual. Default false.
     */
    [[nodiscard]] virtual bool isQuasiStatic() const { return false; }

    /**
     * spatial dimension (2 or 3) of the problem this kernel is assembled
     * for. Set once by the input reader (readElmtSystemBlock) from the
     * mesh dimension, BEFORE getDofsPerNode()/getDofNames() are queried,
     * so kernels whose DoF layout depends on dimension (the mechanics
     * family carries one displacement DoF per axis) report the right
     * count. Default 2 keeps every pre-existing 2D kernel unchanged.
     */
    virtual void setDim(const int &dim) { m_Dim = dim; }
    [[nodiscard]] int getDim() const { return m_Dim; }

    /**
     * conservative a-priori estimate of the largest stable dt for the FIXED-dt
     * matrix-free explicit integrators (a CFL-type bound from the kernel's
     * material constants and the mesh spacing: dx/c for elastodynamics,
     * dx^2/(2*dim*D) for diffusion). <=0 means "no estimate" (implicit kernels
     * and ADR pseudo-time, whose stability is mass-scaled internally). Purely
     * informational: the explicit drivers print it next to the configured dt
     * and warn when dt exceeds it; nothing is enforced.
     */
    [[nodiscard]] virtual double estimateStableDt(const PDMesh &Mesh) const { (void)Mesh; return -1.0; }

    /**
     * 0-based DoF slots whose boundary bonds to GHOST nodes are DROPPED by the
     * kernel's conservative-flux surface correction (Cahn-Hilliard family: the
     * concentration `c` flux and the `mu` gradient-energy Laplacian both skip
     * ghost bonds so a reflected ghost never injects spurious species -- see the
     * species-conservation note). A consequence is that a DIRICHLET condition on
     * such a DoF placed on boundary ghost nodes is INERT: the pinned ghost value
     * is never read by any bulk equation, so it drives no flux (the domain is
     * silently sealed). The input reader uses this list to REJECT that
     * misconfiguration and point the user at the species flux BCs, which inject
     * into the bulk residual instead. Default empty (no ghost-drop DoFs).
     */
    [[nodiscard]] virtual std::vector<int> getGhostDropSpeciesDofSlots() const { return {}; }

    /**
     * per-bond residual / jacobian contribution from bond (NodeI,NodeJ)
     * to the residual at NodeI. Caller (PDSystem) is responsible for:
     *   - calling t_PDOperators.calcAMatrix(NodeI,...) once per source node
     *   - calling t_PDOperators.calcOperators(NodeI,NodeJ,...) once per bond
     *
     * Implicit kernels MUST override this. The default aborts with a clear
     * message so an explicit kernel that is mistakenly driven through the
     * implicit (Jacobian-based) assembler, or an implicit kernel that
     * forgot to implement it, fails loudly instead of silently.
     */
    virtual void computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
                                                const LocalElmtInfo &Info,
                                                const int &NodeI,
                                                const int &NodeJ,
                                                const VectorXd &U_I,
                                                const VectorXd &U_J,
                                                const VectorXd &Uold_I,
                                                const VectorXd &Uold_J,
                                                const double  &Volume,
                                                VectorXd &LocalR,
                                                MatrixXd &LocalK_II,
                                                MatrixXd &LocalK_IJ) const {
        (void)t_PDOperators; (void)Info; (void)NodeI; (void)NodeJ;
        (void)U_I; (void)U_J; (void)Uold_I; (void)Uold_J; (void)Volume;
        (void)LocalR; (void)LocalK_II; (void)LocalK_IJ;
        MessagePrinter::printErrorTxt("ElementBase: kernel '"+getElementType()
            +"' does not implement computeBondResidualAndJacobian. Implicit "
            "(backward-Euler/steady) assembly needs a Jacobian; an explicit "
            "(forward-Euler) kernel must instead be run as a transient analysis "
            "so the matrix-free explicit integrator is used.");
        MessagePrinter::exitPeriX();
    }

    /**
     * per-bond contribution to the EXPLICIT rate at NodeI: the bond part of
     * R_i in du_i/dt = R_i(u). Only the rate (no Jacobian). Called by the
     * forward-Euler integrator for explicit kernels; the field passed in
     * (U_I, U_J) is the current level u^n. Same operator-priming contract as
     * computeBondResidualAndJacobian. Default: no bond contribution.
     */
    virtual void computeBondResidual(const PDOperators &t_PDOperators,
                                     const LocalElmtInfo &Info,
                                     const int &NodeI,
                                     const int &NodeJ,
                                     const VectorXd &U_I,
                                     const VectorXd &U_J,
                                     const VectorXd &Uold_I,
                                     const VectorXd &Uold_J,
                                     const double  &Volume,
                                     VectorXd &LocalR) const {
        (void)t_PDOperators; (void)Info; (void)NodeI; (void)NodeJ;
        (void)U_I; (void)U_J; (void)Uold_I; (void)Uold_J; (void)Volume;
        LocalR.setToZeros();
    }

    /**
     * per-node residual / jacobian contribution at NodeI (e.g. body
     * forces, source terms, time-derivative). Default: nothing.
     */
    virtual void computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                 const int &NodeI,
                                                 const VectorXd &U_I,
                                                 const VectorXd &Uold_I,
                                                 const double  &Volume,
                                                 VectorXd &LocalR_node,
                                                 MatrixXd &LocalK_node) const {
        (void)Info; (void)NodeI; (void)U_I; (void)Uold_I; (void)Volume;
        LocalR_node.setToZeros();
        LocalK_node.setToZeros();
    }

    /**
     * per-node contribution to the EXPLICIT rate at NodeI: the nodal part of
     * R_i in du_i/dt = R_i(u) (sources, reactions, body terms). Only the rate
     * (no Jacobian, and NO time-derivative term -- the integrator owns dt).
     * Default: no nodal contribution.
     */
    virtual void computeNodalResidual(const LocalElmtInfo &Info,
                                      const int &NodeI,
                                      const VectorXd &U_I,
                                      const VectorXd &Uold_I,
                                      const double  &Volume,
                                      VectorXd &LocalR_node) const {
        (void)Info; (void)NodeI; (void)U_I; (void)Uold_I; (void)Volume;
        LocalR_node.setToZeros();
    }

    /**
     * advertise the named output projection fields this kernel can
     * compute beyond "solution". The framework uses these descriptors
     * to validate the user's OutputSystem request and to size each
     * VTU DataArray.
     *
     * Default: no extra projections (only "solution" is implicit).
     */
    [[nodiscard]] virtual std::vector<ProjectionInfo> getAvailableProjections() const {
        return {};
    }

    /**
     * compute the named projection at NodeI given the current global
     * U. The PD A-matrix at NodeI must already be primed by the
     * caller via ops.calcAMatrix(NodeI, ...). The kernel is allowed
     * to call ops.calcOperators(NodeI, NodeJ, ...) to fetch per-bond
     * differential operator values.
     *
     * @param name        which projection to compute (must be one of the
     *                    names returned by getAvailableProjections())
     * @param Mesh        the pd mesh
     * @param ops         the pd operators (mutable: state is bond-level)
     * @param U           the global solution vector
     * @param NodeI       1-based pd node id
     * @param DofsPerNode total dofs per pd node
     * @param out         output buffer; must be sized to the
     *                    Components field of the matching ProjectionInfo
     */
    virtual void computeNodalProjection(const std::string &name,
                                        const PDMesh &Mesh,
                                        PDOperators &ops,
                                        const VectorXd &U,
                                        const int &NodeI,
                                        const int &DofsPerNode,
                                        std::vector<double> &out) const {
        (void)name; (void)Mesh; (void)ops; (void)U;
        (void)NodeI; (void)DofsPerNode; (void)out;
    }

    /**
     * called once at the start of each Newton iteration, before
     * formResidualAndJacobian assembly. Coupled kernels can use this
     * hook to compute frozen auxiliary quantities such as gradients
     * of fields at every node, store them
     * in mutable state, and then read them back during per-bond /
     * per-node residual evaluation.  Default: no-op.
     *
     * Uold is the previous time level's solution -- useful for
     * convex-splitting schemes that need to evaluate explicit terms
     * on the old field once per step.
     */
    virtual void preprocessIteration(const PDMesh &Mesh,
                                     PDOperators &ops,
                                     const LocalElmtInfo &Info,
                                     const int &DofsPerNode,
                                     const VectorXd &U,
                                     const VectorXd &Uold) const {
        (void)Mesh; (void)ops; (void)Info; (void)DofsPerNode; (void)U; (void)Uold;
    }

protected:
    /** spatial dimension (2 or 3); see setDim(). */
    int m_Dim = 2;

    /** PDDO Laplacian value for the bond the operators are currently
     *  primed on: d2/dx2 + d2/dy2 (+ d2/dz2 in 3D). Centralises the
     *  dimension switch so no scalar kernel can forget the z term. */
    [[nodiscard]] double pdLaplacian(const PDOperators &ops) const {
        double lap = ops("d2/dx2") + ops("d2/dy2");
        if (m_Dim >= 3) lap += ops("d2/dz2");
        return lap;
    }

};
