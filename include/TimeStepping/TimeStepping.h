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
//+++ Function: Backward-Euler transient driver. Owns the outer
//+++           time loop and delegates the per-step Newton solve
//+++           to NonlinearSolver.
//+++
//+++           Two adaptive mechanisms (both opt-in via "adaptive"):
//+++             1) Cut-back on Newton failure: dt *= cutback_factor,
//+++                state restored to U_old, retry up to max_cutbacks.
//+++             2) Iter-driven dt control on success: after a
//+++                successful step, if last_iters <= optimal_iters
//+++                grow dt for the next step (dt *= growth_factor),
//+++                else shrink it (dt *= cutback_factor). The dt
//+++                running value is clamped by [min_dt, max_dt].
//+++
//+++           PDProblem owns one TimeStepping instance and reads
//+++           its configuration from the top-level "TimeStepping"
//+++           JSON block.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <functional>
#include <vector>

#include "BCSystem/BCSystem.h"
#include "ElmtSystem/ElmtSystem.h"
#include "ElmtSystem/LocalElmtInfo.h"
#include "LinearSolver/LinearSolver.h"
#include "MathUtils/SparseMatrix.h"
#include "MathUtils/VectorXd.h"
#include "NonlinearSolver/NonlinearSolver.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "PDSystem/PDSystem.h"

class TimeStepping {
public:
    TimeStepping()=default;

    //*****************************************************
    //*** required configuration
    //*****************************************************
    void setDt(const double &dt)                 { m_Dt=dt; }
    [[nodiscard]] double getDt() const           { return m_Dt; }

    void setTotalTime(const double &t)           { m_TotalTime=t; }
    [[nodiscard]] double getTotalTime() const    { return m_TotalTime; }

    /** output cadence (must be >= 1). Set by PDProblem from the
     *  OutputSystem block before solve() runs; the JSON parser does
     *  not accept this value under "TimeStepping" any more. */
    void setOutputInterval(const int &n)         { m_OutputInterval=(n>=1)?n:1; }
    [[nodiscard]] int getOutputInterval() const  { return m_OutputInterval; }

    /** print one >>>>>> Step ... header per step (default true) */
    void setVerbose(const bool &v)               { m_Verbose=v; }
    [[nodiscard]] bool isVerbose() const         { return m_Verbose; }

    /** forward-Euler (explicit) instead of backward-Euler. Set by
     *  PDProblem from ElmtSystem::isExplicit(); only affects which solve
     *  routine the driver calls and the printed scheme name. */
    void setExplicit(const bool &v)              { m_Explicit=v; }
    [[nodiscard]] bool isExplicit() const        { return m_Explicit; }

    //*****************************************************
    //*** adaptive controller (off by default)
    //***
    //*** When adaptive=on:
    //***   * Newton failure -> dt *= cutback_factor, retry.
    //***   * Newton success with iters <= optimal_iters
    //***                     -> dt *= growth_factor for next step.
    //***   * Newton success with iters >  optimal_iters
    //***                     -> dt *= cutback_factor for next step.
    //*** dt is clamped to [min_dt, max_dt]. max_dt defaults to the
    //*** initial dt; min_dt defaults to 0 (no lower bound).
    //*****************************************************
    void setAdaptive(const bool &a)              { m_Adaptive=a; }
    [[nodiscard]] bool isAdaptive() const        { return m_Adaptive; }

    void setOptimalIters(const int &n)           { m_OptimalIters=n; }
    [[nodiscard]] int getOptimalIters() const    { return m_OptimalIters; }

    void setGrowthFactor(const double &g)        { m_GrowthFactor=g; }
    [[nodiscard]] double getGrowthFactor() const { return m_GrowthFactor; }

    void setCutbackFactor(const double &c)       { m_CutbackFactor=c; }
    [[nodiscard]] double getCutbackFactor() const { return m_CutbackFactor; }

    void setMaxCutbacks(const int &n)            { m_MaxCutbacks=n; }
    [[nodiscard]] int getMaxCutbacks() const     { return m_MaxCutbacks; }

    void setMinDt(const double &dt)              { m_MinDt=dt; }
    [[nodiscard]] double getMinDt() const        { return m_MinDt; }

    /** 0 (default) means "use the initial dt as the upper cap" */
    void setMaxDt(const double &dt)              { m_MaxDt=dt; }
    [[nodiscard]] double getMaxDt() const        { return m_MaxDt; }

    //*****************************************************
    //*** last-call status
    //*****************************************************
    [[nodiscard]] int    getLastSteps() const      { return m_LastSteps; }
    [[nodiscard]] double getLastTime() const       { return m_LastTime; }
    [[nodiscard]] bool   getLastConverged() const  { return m_LastConverged; }
    [[nodiscard]] int    getLastCutbacks() const   { return m_LastCutbacks; }
    [[nodiscard]] int    getLastGrows() const      { return m_LastGrows; }
    [[nodiscard]] double getLastDt() const         { return m_LastDt; }

    /**
     * Run the backward-Euler transient analysis. The caller (PDProblem)
     * is responsible for applying initial conditions to U and seeding
     * Uold = U *before* calling this.
     *
     * The save callback is invoked:
     *   - once at t=0, step 0 (initial state),
     *   - whenever (step % m_OutputInterval == 0),
     *   - at the final step (so the time series always ends on disk).
     * The interval value is sourced from OutputSystem.interval and
     * forwarded into m_OutputInterval by PDProblem before solve().
     *
     * Returns true if every step converged, false if Newton failed at
     * a step and either cutbacks are disabled or were exhausted.
     */
    bool solve(const PDMesh &Mesh,
               PDOperators &Operators,
               const ElmtSystem &ElmtSys,
               const BCSystem &BCSys,
               PDSystem &PDSys,
               LinearSolver &LinSolver,
               NonlinearSolver &Newton,
               const int &DofsPerNode,
               SparseMatrix &K,
               VectorXd &U,
               VectorXd &Uold,
               VectorXd &dU,
               VectorXd &RHS,
               const std::function<void(const int&,const double&)> &SaveResults);

    /**
     * Run the FORWARD-Euler (explicit) transient analysis. Matrix-free:
     * each step evaluates the nodal rate R(U^n) via PDSys.formExplicitRate,
     * advances U^{n+1} = U^n + dt * R(U^n), then re-pins the boundary
     * conditions with BCSys.presetSolution (Dirichlet hold + mirror-Neumann
     * ghost refresh). No Newton iteration, no linear solve, no Jacobian.
     *
     * dt is fixed (the adaptive controller does not apply -- explicit
     * stability is a CFL constraint, not a Newton-convergence one). A
     * blow-up guard aborts cleanly with a smaller-dt hint if the solution
     * becomes non-finite (dt above the stability limit). Save cadence and
     * the t=0 emit match solve(). Requires every registered element to be
     * explicit; the caller (PDProblem) enforces that and that the job is
     * transient.
     *
     * Returns true if the run reached total_time, false on blow-up or a
     * bad configuration.
     */
    bool solveExplicit(const PDMesh &Mesh,
                       PDOperators &Operators,
                       const ElmtSystem &ElmtSys,
                       const BCSystem &BCSys,
                       PDSystem &PDSys,
                       const int &DofsPerNode,
                       VectorXd &U,
                       VectorXd &Uold,
                       VectorXd &Rate,
                       const std::function<void(const int&,const double&)> &SaveResults);

    /**
     * Run the explicit-DYNAMICS (central-difference) transient analysis for
     * a second-order equation of motion rho*u'' = L(u)+b. Matrix-free: each
     * step evaluates the nodal acceleration a(u^n)=R via
     * PDSys.formExplicitDynamicsRate, advances with the central-difference
     * (Stoermer--Verlet) rule
     *     u^{n+1} = 2 u^n - u^{n-1} + dt^2 * a(u^n),
     * then re-imposes the time-independent boundary conditions at the new
     * state. Initial velocity belongs to ICSystem and is used when the
     * history state is initialized.
     *
     * dt is fixed (CFL-limited, like solveExplicit). A blow-up guard aborts
     * cleanly if the solution diverges. Returns true on reaching total_time.
     */
    bool solveExplicitDynamics(const PDMesh &Mesh,
                               PDOperators &Operators,
                               const ElmtSystem &ElmtSys,
                               const BCSystem &BCSys,
                               PDSystem &PDSys,
                               const int &DofsPerNode,
                               VectorXd &U,
                               VectorXd &Uold,
                               VectorXd &Rate,
                               const std::function<void(const int&,const double&)> &SaveResults);

    /**
     * Run the quasi-static analysis by Adaptive Dynamic Relaxation (ADR). The
     * static equilibrium div(sigma)+b=0 is reached as the steady state of a
     * damped fictitious-mass pseudo-dynamics: with the kernel-supplied
     * fictitious acceleration a=(L+b)/lambda (lambda the ADR mass, baked into
     * the kernel's rate), each pseudo-step
     *   1. presets the (ramped) boundary displacement at the new pseudo-time,
     *   2. evaluates a(u) matrix-free via PDSys.formExplicitDynamicsRate,
     *   3. advances Underwood's central-difference DR update with an adaptive
     *      damping cn estimated from the Rayleigh quotient,
     *      v^{n+1/2} = ((2-cn dt) v^{n-1/2} + 2 dt a)/(2 + cn dt),
     *      u^{n+1} = u^n + dt v^{n+1/2},
     *   4. re-pins the boundary.
     * dt is the pseudo-time step (a fixed relaxation parameter, typically 1).
     * The load is ramped over the whole run (total_time pseudo-steps), so each
     * increment stays near equilibrium. Returns true on reaching total_time.
     */
    bool solveADR(const PDMesh &Mesh,
                  PDOperators &Operators,
                  const ElmtSystem &ElmtSys,
                  const BCSystem &BCSys,
                  PDSystem &PDSys,
                  const int &DofsPerNode,
                  VectorXd &U,
                  VectorXd &Uold,
                  VectorXd &Rate,
                  const std::function<void(const int&,const double&)> &SaveResults);


    /** time-derivative order of the explicit scheme (1 forward Euler,
     *  2 central difference). Set by PDProblem from ElmtSystem; only used
     *  for the printed scheme name. */
    void setTimeOrder(const int &n)              { m_TimeOrder=n; }
    [[nodiscard]] int getTimeOrder() const       { return m_TimeOrder; }

    /** quasi-static (ADR) scheme flag, for the printed scheme name. */
    void setQuasiStatic(const bool &v)           { m_QuasiStatic=v; }
    [[nodiscard]] bool isQuasiStatic() const     { return m_QuasiStatic; }



    /** run the matrix-free explicit assembly with the OpenMP back-end. Set by
     *  PDProblem from JobSystem (assemble==openmp). The explicit drivers forward
     *  it to the rate routines and the explicit kernels through LocalElmtInfo so a
     *  single run-wide key controls both the implicit and the explicit path; the
     *  worker count is taken from OMP_NUM_THREADS. Off by default (serial). */
    void setParallelAssemble(const bool &v)       { m_ParallelAssemble=v; }
    [[nodiscard]] bool isParallelAssemble() const { return m_ParallelAssemble; }

    /** run the matrix-free explicit assembly on the GPU (assemble==cuda). Set by
     *  PDProblem from JobSystem. Only the forward-Euler PDDO rate kernels have a
     *  device port; everything else falls back to the OpenMP/serial CPU path. */
    void setCudaAssemble(const bool &v)           { m_CudaAssemble=v; }
    [[nodiscard]] bool isCudaAssemble() const     { return m_CudaAssemble; }

    void printTimeSteppingInfo() const;

private:
    /**
     * print the registered kernels' a-priori stable-dt estimate (CFL-type
     * bound) next to the configured dt, and warn when dt exceeds it. No-op
     * when no kernel reports an estimate. Called by the fixed-dt explicit
     * drivers (forward Euler / central difference / velocity-Verlet).
     */
    void printStableDtHint(const PDMesh &Mesh,const ElmtSystem &ElmtSys) const;

    // configuration
    double m_Dt=0.0;
    double m_TotalTime=0.0;
    int    m_OutputInterval=1;
    bool   m_Verbose=true;
    bool   m_Explicit=false;   // forward Euler (set from ElmtSystem::isExplicit)
    int    m_TimeOrder=1;      // 1 forward Euler, 2 central-difference dynamics
    bool   m_QuasiStatic=false;// quasi-static via ADR (set from ElmtSystem)
    bool   m_ParallelAssemble=false;// OpenMP matrix-free explicit assembly (assemble==openmp)
    bool   m_CudaAssemble=false;    // CUDA GPU matrix-free explicit assembly (assemble==cuda)

    // adaptive controller (off by default)
    bool   m_Adaptive=false;
    int    m_OptimalIters=5;
    double m_GrowthFactor=1.25;
    double m_CutbackFactor=0.5;
    int    m_MaxCutbacks=4;
    double m_MinDt=0.0;
    double m_MaxDt=0.0;   // 0 => effective cap is the initial m_Dt
    // last-call status
    int    m_LastSteps=0;
    double m_LastTime=0.0;
    bool   m_LastConverged=false;
    int    m_LastCutbacks=0;
    int    m_LastGrows=0;
    double m_LastDt=0.0;
};
