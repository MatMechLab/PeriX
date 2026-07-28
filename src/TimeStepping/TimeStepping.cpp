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
//+++ Function: Backward-Euler transient driver implementation,
//+++           with iter-driven adaptive dt control.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "TimeStepping/TimeStepping.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Utils/MessagePrinter.h"
#include "Utils/Timer.h"

namespace {
    // One-line banner naming the matrix-free explicit assembly back-end. For
    // OpenMP (and the CPU fallback of a CUDA run) it also reports the
    // worker-thread count available at run time (OMP_NUM_THREADS). Printed once
    // at the start of each explicit driver so a run records how it was assembled.
    //   backend  : "serial" | "openmp" | "cuda"
    //   parallel : whether a CPU OpenMP team is in play (for the thread count)
    void printExplicitAssembleInfo(const char *backend,const bool parallel,const char *scheme) {
        char buf[176];
#ifdef _OPENMP
        const int nthreads = parallel ? omp_get_max_threads() : 1;
#else
        const int nthreads = 1;
#endif
        if (nthreads<=1)
            std::snprintf(buf,sizeof(buf),
                          "%s integrator: matrix-free assembly = %s",scheme,backend);
        else
            std::snprintf(buf,sizeof(buf),
                          "%s integrator: matrix-free assembly = %s (%d CPU thread%s)",
                          scheme,backend,nthreads,nthreads==1?"":"s");
        MessagePrinter::printNormalTxt(buf);
    }
}

void TimeStepping::printStableDtHint(const PDMesh &Mesh,const ElmtSystem &ElmtSys) const {
    // A-priori stable-dt estimate for the fixed-dt explicit integrators (the
    // registered kernels report a CFL-type bound from their material constants
    // and the mesh spacing; <=0 = no estimate). Users otherwise pick dt blind
    // and discover an instability only at the blow-up abort deep into the run.
    const double est=ElmtSys.estimateStableDt(Mesh);
    if (est<=0.0) return;
    char buf[176];
    std::snprintf(buf,sizeof(buf),
                  "  stable-dt estimate ~%10.3e (element CFL-type bound), configured dt=%10.3e",
                  est,m_Dt);
    MessagePrinter::printNormalTxt(buf);
    if (m_Dt>est) {
        MessagePrinter::printWarningTxt("TimeStepping: the configured dt exceeds the element's "
                                        "stable-dt estimate -- the explicit run is likely to blow up. "
                                        "Reduce TimeStepping.dt (the bound is approximate; a safety "
                                        "factor of ~0.5-0.8 of it is customary).");
    }
}

bool TimeStepping::solve(const PDMesh &Mesh,
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
                         const std::function<void(const int&,const double&)> &SaveResults) {
    constexpr int bufSize=200;
    char buffer[bufSize];

    // -------- sanity-check the configuration --------
    auto bail=[&](const string &msg)->bool {
        MessagePrinter::printErrorTxt(msg);
        m_LastConverged=false;
        m_LastSteps=0;
        m_LastTime=0.0;
        m_LastCutbacks=0;
        m_LastGrows=0;
        m_LastDt=0.0;
        return false;
    };

    if (m_Dt<=0.0 || m_TotalTime<=0.0) {
        return bail("TimeStepping::solve: dt and total_time must be positive (got dt="
                    +std::to_string(m_Dt)+", total_time="+std::to_string(m_TotalTime)+")");
    }
    if (m_OutputInterval<1) m_OutputInterval=1;
    if (m_Adaptive) {
        if (m_GrowthFactor<1.0)  return bail("TimeStepping::solve: growth_factor must be >= 1.0");
        if (m_CutbackFactor<=0.0||m_CutbackFactor>=1.0)
            return bail("TimeStepping::solve: cutback_factor must be in (0,1)");
        if (m_OptimalIters<1)    return bail("TimeStepping::solve: optimal_iters must be >= 1");
        if (m_MaxCutbacks<0)     return bail("TimeStepping::solve: max_cutbacks must be >= 0");
        if (m_MinDt<0.0)         return bail("TimeStepping::solve: min_dt must be >= 0");
        if (m_MaxDt<0.0)         return bail("TimeStepping::solve: max_dt must be >= 0");
    }

    // The effective upper cap for dt. 0 (default) means "the initial dt is the cap"
    // -- so the adaptive controller never grows past the user-stated baseline.
    const double dtCapHi=(m_MaxDt>0.0)?m_MaxDt:m_Dt;
    const double dtCapLo=m_MinDt;

    // emit the initial state (t=0) so the time series starts there.
    if (SaveResults) SaveResults(0,0.0);

    LocalElmtInfo info;
    double t       =0.0;
    double dt      =m_Dt;       // running dt (mutated by the adaptive controller)
    int    step    =0;
    int    totalCB =0;
    int    totalGrow=0;

    // Relative epsilon used to decide when t has "reached" total_time.
    const double tEps=1.0e-12*m_TotalTime;

    Timer stepTimer;

    while (t<m_TotalTime-tEps) {
        step++;

        // Clamp dt for THIS step: respect the upper cap and never overshoot total_time.
        // The end-of-time clamp guarantees the run lands exactly on total_time.
        double dtAttempt=std::min(dt,dtCapHi);
        dtAttempt=std::min(dtAttempt,m_TotalTime-t);
        double tNew=t+dtAttempt;

        int  cutbacks=0;
        bool ok=false;
        while (true) {
            info.Dt      =dtAttempt;
            info.T       =tNew;
            info.Timestep=step;

            if (m_Verbose) {
                std::snprintf(buffer,bufSize,
                              "Time=%13.5e, step=%8d, dt=%13.5e",
                              tNew,step,dtAttempt);
                MessagePrinter::printNormalTxt(buffer);
            }

            // Ramp the prescribed boundary to THIS step's load level before the
            // first assembly. Without it a quasi-static kernel (no time-derivative
            // term) would see a near-zero residual at the start of the step and
            // "converge" before the load is applied. Constant BCs are unaffected.
            BCSys.presetSolution(Mesh,DofsPerNode,U,tNew);

            stepTimer.startTimer();
            ok=Newton.solve(Mesh,Operators,ElmtSys,BCSys,PDSys,LinSolver,
                            info,DofsPerNode,K,U,Uold,dU,RHS);
            stepTimer.endTimer();

            if (ok) break;

            // ---- Newton failed for this attempt ----
            if (!m_Adaptive || cutbacks>=m_MaxCutbacks) {
                std::snprintf(buffer,bufSize,
                              "Newton did not converge at step %d (t=%12.5e, dt=%10.3e), |R|=%12.5e",
                              step,tNew,dtAttempt,Newton.getLastRNorm());
                MessagePrinter::printErrorTxt(buffer);
                m_LastConverged=false;
                m_LastSteps   =step-1;
                m_LastTime    =t;
                m_LastCutbacks=totalCB+cutbacks;
                m_LastGrows   =totalGrow;
                m_LastDt      =dtAttempt;
                return false;
            }

            cutbacks++;
            totalCB++;

            // Restore the iterate to the start-of-step state and re-pin BCs at
            // the last COMMITTED time t (not 0), so a velocity-ramped Dirichlet
            // is restored to its converged level before the smaller-dt retry.
            U=Uold;
            BCSys.presetSolution(Mesh,DofsPerNode,U,t);

            dtAttempt*=m_CutbackFactor;
            if (dtCapLo>0.0 && dtAttempt<dtCapLo) {
                std::snprintf(buffer,bufSize,
                              "TimeStepping: dt=%10.3e fell below min_dt=%10.3e at step %d, aborting",
                              dtAttempt,dtCapLo,step);
                MessagePrinter::printErrorTxt(buffer);
                m_LastConverged=false;
                m_LastSteps   =step-1;
                m_LastTime    =t;
                m_LastCutbacks=totalCB;
                m_LastGrows   =totalGrow;
                m_LastDt      =dtAttempt;
                return false;
            }
            tNew=t+dtAttempt;

            std::snprintf(buffer,bufSize,
                          " Transient solver failed, reduce dt to %13.5e (cutback %d/%d)",
                          dtAttempt,cutbacks,m_MaxCutbacks);
            MessagePrinter::printWarningTxt(buffer);
        }

        // ---- step succeeded -- commit the new state ----
        t   =tNew;
        Uold=U;

        if (m_Verbose) {
            std::snprintf(buffer,bufSize,
                          "Newton converged, iters=%3d, elapsed time=%10.3es",
                          Newton.getLastIters(),stepTimer.getDurationInSecond());
            MessagePrinter::printNormalTxt(buffer,MessageColor::BLUE);
        }

        // ---- adaptive control of the NEXT step's dt (only on success) ----
        if (m_Adaptive) {
            const int iters=Newton.getLastIters();
            if (iters<=m_OptimalIters) {
                const double dtNew=std::min(dtAttempt*m_GrowthFactor,dtCapHi);
                if (dtNew>dtAttempt) totalGrow++;
                dt=dtNew;
                if (m_Verbose && dtNew>dtAttempt) {
                    std::snprintf(buffer,bufSize,
                                  "  iters=%d <= optimal=%d, dt -> %10.3e (grow x%5.3f)",
                                  iters,m_OptimalIters,dt,m_GrowthFactor);
                    MessagePrinter::printNormalTxt(buffer);
                }
            }
            else {
                double dtNew=dtAttempt*m_CutbackFactor;
                if (dtCapLo>0.0) dtNew=std::max(dtNew,dtCapLo);
                dt=dtNew;
                if (m_Verbose) {
                    std::snprintf(buffer,bufSize,
                                  "  iters=%d >  optimal=%d, dt -> %10.3e (shrink x%5.3f)",
                                  iters,m_OptimalIters,dt,m_CutbackFactor);
                    MessagePrinter::printNormalTxt(buffer);
                }
            }
        }
        // (when adaptive is off, dt stays at the user-supplied initial value
        //  — the only mutation each step is the end-of-time clamp on dtAttempt)

        const bool isFinalStep=(t>=m_TotalTime-tEps);
        if (SaveResults && (step%m_OutputInterval==0 || isFinalStep)) {
            SaveResults(step,t);
        }

        if (m_Verbose) MessagePrinter::printStars();
    }

    m_LastConverged=true;
    m_LastSteps   =step;
    m_LastTime    =t;
    m_LastCutbacks=totalCB;
    m_LastGrows   =totalGrow;
    m_LastDt      =dt;
    return true;
}

bool TimeStepping::solveExplicit(const PDMesh &Mesh,
                                 PDOperators &Operators,
                                 const ElmtSystem &ElmtSys,
                                 const BCSystem &BCSys,
                                 PDSystem &PDSys,
                                 const int &DofsPerNode,
                                 VectorXd &U,
                                 VectorXd &Uold,
                                 VectorXd &Rate,
                                 const std::function<void(const int&,const double&)> &SaveResults) {
    constexpr int bufSize=400;
    char buffer[bufSize];

    auto bail=[&](const string &msg)->bool {
        MessagePrinter::printErrorTxt(msg);
        m_LastConverged=false; m_LastSteps=0; m_LastTime=0.0;
        m_LastCutbacks=0; m_LastGrows=0; m_LastDt=0.0;
        return false;
    };

    if (m_Dt<=0.0 || m_TotalTime<=0.0) {
        return bail("TimeStepping::solveExplicit: dt and total_time must be positive (got dt="
                    +std::to_string(m_Dt)+", total_time="+std::to_string(m_TotalTime)+")");
    }
    if (m_OutputInterval<1) m_OutputInterval=1;

    // emit the initial state (t=0) so the time series starts there.
    if (SaveResults) SaveResults(0,0.0);

    LocalElmtInfo info;
    info.UseParallel=m_ParallelAssemble||m_CudaAssemble; // CPU paths/fallback use OpenMP when CUDA
    printExplicitAssembleInfo(m_CudaAssemble?"cuda":(m_ParallelAssemble?"openmp":"serial"),
                              info.UseParallel,"Forward-Euler");
    printStableDtHint(Mesh,ElmtSys);
    double t   =0.0;
    int    step=0;
    const double tEps=1.0e-12*m_TotalTime;
    Timer stepTimer;

    const int n=U.getSize();
    double *uptr=U.getDataPtr();

    // Blow-up guard: windowed geometric-growth detector. Abort when max|U|
    // grew by more than 1e3x over the last kBlowupWindow steps -- an unstable
    // mode amplifies geometrically (orders of magnitude across any 50-step
    // window), while every legitimate BC-driven ramp grows polynomially in t
    // and can never sustain that rate, so large-amplitude runs (values >> 1 in
    // the user's units) are no longer falsely aborted the way a cap anchored
    // to the initial state alone did. Non-finite entries abort immediately and
    // are checked PER ENTRY: std::max ignores NaN (a<NaN is false), so a NaN
    // pocket that never passes through +-inf would otherwise sail to a
    // "successful" all-NaN completion.
    constexpr int kBlowupWindow=50;
    double umaxHist[kBlowupWindow]={0.0};

    while (t<m_TotalTime-tEps) {
        step++;

        // Fixed dt for explicit; only clamp so the run lands exactly on
        // total_time (no adaptive control: stability is a CFL constraint).
        const double dtAttempt=std::min(m_Dt,m_TotalTime-t);
        const double tNew=t+dtAttempt;

        info.Dt      =dtAttempt;
        info.T       =tNew;
        info.Timestep=step;

        if (m_Verbose) {
            std::snprintf(buffer,bufSize,
                          "Time=%13.5e, step=%8d, dt=%13.5e (forward Euler)",
                          tNew,step,dtAttempt);
            MessagePrinter::printNormalTxt(buffer);
        }

        stepTimer.startTimer();
        // 1) freeze the start-of-step state as U^n.
        Uold=U;
        // 2) evaluate the explicit nodal rate R(U^n) (matrix-free).
        PDSys.formExplicitRate(Mesh,Operators,ElmtSys,info,U,Uold,Rate);
        // 3) forward-Euler update: U^{n+1} = U^n + dt * R(U^n).
        const double *rptr =Rate.getDataPtr();
        const double *uoptr=Uold.getDataPtr();
        for (int k=0;k<n;k++) uptr[k]=uoptr[k]+dtAttempt*rptr[k];
        // 4) re-impose the Dirichlet and mirror conditions at t^{n+1}.
        BCSys.presetSolution(Mesh,DofsPerNode,U,tNew);
        stepTimer.endTimer();

        // ---- blow-up guard (forward-Euler CFL stability + non-finite scan) ----
        const int slot=(step-1)%kBlowupWindow;
        const double past=(step>kBlowupWindow)?umaxHist[slot]:-1.0; // umax at step-kBlowupWindow
        const double blowupCap=(past>=0.0)?1.0e3*std::max(1.0,past)
                                          :std::numeric_limits<double>::infinity();
        double umax=0.0; bool finiteU=true;
        for (int k=0;k<n;k++) {
            const double av=std::fabs(uptr[k]);
            if (!(av<=std::numeric_limits<double>::max())) finiteU=false; // NaN or inf
            else if (av>umax) umax=av;
        }
        if (!finiteU || umax>blowupCap) {
            std::snprintf(buffer,bufSize,
                "TimeStepping::solveExplicit: solution blew up at step %d (t=%12.5e, dt=%10.3e; %s "
                "max|U|=%10.3e vs %10.3e a window of %d steps ago). Forward Euler is only conditionally "
                "stable: reduce dt (diffusion CFL dt<~dx^2/(2*D*dim); a stiff reaction also limits dt) "
                "or use the implicit backward-Euler kernel.",step,tNew,dtAttempt,
                finiteU?"":"non-finite (NaN/inf) entries detected;",umax,
                (past>=0.0)?past:0.0,kBlowupWindow);
            MessagePrinter::printErrorTxt(buffer);
            m_LastConverged=false; m_LastSteps=step-1; m_LastTime=t;
            m_LastCutbacks=0; m_LastGrows=0; m_LastDt=dtAttempt;
            return false;
        }
        umaxHist[slot]=umax;

        t=tNew;

        if (m_Verbose) {
            std::snprintf(buffer,bufSize,
                          "Explicit step done, max|U|=%12.5e, elapsed time=%10.3es",
                          umax,stepTimer.getDurationInSecond());
            MessagePrinter::printNormalTxt(buffer,MessageColor::BLUE);
        }

        const bool isFinalStep=(t>=m_TotalTime-tEps);
        if (SaveResults && (step%m_OutputInterval==0 || isFinalStep)) {
            SaveResults(step,t);
        }
        if (m_Verbose) MessagePrinter::printStars();
    }

    m_LastConverged=true;
    m_LastSteps   =step;
    m_LastTime    =t;
    m_LastCutbacks=0;
    m_LastGrows   =0;
    m_LastDt      =m_Dt;
    return true;
}

bool TimeStepping::solveExplicitDynamics(const PDMesh &Mesh,
                                         PDOperators &Operators,
                                         const ElmtSystem &ElmtSys,
                                         const BCSystem &BCSys,
                                         PDSystem &PDSys,
                                         const int &DofsPerNode,
                                         VectorXd &U,
                                         VectorXd &Uold,
                                         VectorXd &Rate,
                                         const std::function<void(const int&,const double&)> &SaveResults) {
    constexpr int bufSize=400;
    char buffer[bufSize];

    auto bail=[&](const string &msg)->bool {
        MessagePrinter::printErrorTxt(msg);
        m_LastConverged=false; m_LastSteps=0; m_LastTime=0.0;
        m_LastCutbacks=0; m_LastGrows=0; m_LastDt=0.0;
        return false;
    };

    if (m_Dt<=0.0 || m_TotalTime<=0.0) {
        return bail("TimeStepping::solveExplicitDynamics: dt and total_time must be positive (got dt="
                    +std::to_string(m_Dt)+", total_time="+std::to_string(m_TotalTime)+")");
    }
    if (m_OutputInterval<1) m_OutputInterval=1;

    LocalElmtInfo info;
    info.UseParallel=m_ParallelAssemble||m_CudaAssemble; // CPU fallback uses OpenMP when CUDA
    info.UseCuda=m_CudaAssemble;                          // the explicit fracture kernel has a device port
    printExplicitAssembleInfo(m_CudaAssemble?"cuda":(m_ParallelAssemble?"openmp":"serial"),
                              info.UseParallel,"Central-difference");
    printStableDtHint(Mesh,ElmtSys);
    double t   =0.0;
    int    step=0;
    const double tEps=1.0e-12*m_TotalTime;
    Timer stepTimer;

    const int n=U.getSize();
    double *uptr =U.getDataPtr();
    double *uoptr=Uold.getDataPtr();

    // ---- consistent rest start + meaningful t=0 output ----
    // One rate evaluation at the preset initial state, BEFORE the loop:
    // (1) rest start: central difference needs u^{-1}=u^0-dt*v^0+(dt^2/2)a^0.
    //     The caller seeds Uold=u^0 (v^0=0 -- there is no velocity-IC channel),
    //     which silently drops the a^0 term: the first step then advances by
    //     dt^2*a^0 instead of (dt^2/2)*a^0, injecting a persistent spurious
    //     velocity (dt/2)|a^0| whenever the run does NOT start in equilibrium
    //     (suddenly applied body force, pre-strained IC). Add the missing term.
    // (2) the kernel's one-off caches (seeded crack health, cached operators)
    //     are built by this call, so the t=0 snapshot below shows the seeded
    //     notch damage / initial stress instead of zeros.
    // For a run starting at rest in equilibrium (a^0=0) both effects vanish
    // and the trajectory is bit-identical to the previous behaviour.
    info.Dt=m_Dt; info.T=0.0; info.Timestep=0;
    PDSys.formExplicitDynamicsRate(Mesh,Operators,ElmtSys,info,U,Uold,Rate);
    {
        const double *rptr=Rate.getDataPtr();
        const double half=0.5*m_Dt*m_Dt;
        for (int k=0;k<n;k++) uoptr[k]=uptr[k]+half*rptr[k];
    }
    // emit the initial state (t=0) so the time series starts there.
    if (SaveResults) SaveResults(0,0.0);

    // Blow-up guard: windowed geometric-growth detector + per-entry non-finite
    // scan (see solveExplicit for the rationale).
    constexpr int kBlowupWindow=50;
    double umaxHist[kBlowupWindow]={0.0};
    double dtPrev=m_Dt;   // dt of the previous step (variable-step final clamp)

    while (t<m_TotalTime-tEps) {
        step++;

        // Fixed dt (CFL-limited); clamp only so the run lands on total_time.
        const double dtAttempt=std::min(m_Dt,m_TotalTime-t);
        const double tNew=t+dtAttempt;
        const double dt2=dtAttempt*dtAttempt;

        info.Dt      =dtAttempt;
        info.T       =tNew;
        info.Timestep=step;

        if (m_Verbose) {
            std::snprintf(buffer,bufSize,
                          "Time=%13.5e, step=%8d, dt=%13.5e (central difference)",
                          tNew,step,dtAttempt);
            MessagePrinter::printNormalTxt(buffer);
        }

        stepTimer.startTimer();
        // 1) acceleration a(u^n) = R (matrix-free; bond force + damage cached
        //    in the kernel's preprocessIteration, gathered to nodes here).
        PDSys.formExplicitDynamicsRate(Mesh,Operators,ElmtSys,info,U,Uold,Rate);
        // 2) central-difference update, in place (each component depends only
        //    on its own (u^n,u^{n-1},a^n), so the shift Uold<-U, U<-Unew is
        //    fused per entry). Uniform dt: u^{n+1}=2u^n-u^{n-1}+dt^2 a^n. A
        //    CLAMPED final step (dtAttempt!=dtPrev) takes the variable-step
        //    form u^{n+1}=u^n+(dt/dtPrev)(u^n-u^{n-1})+a^n dt(dt+dtPrev)/2 --
        //    the uniform formula reduces to it when dt==dtPrev, but applied
        //    verbatim to a shorter last step it would carry the FULL previous
        //    velocity increment (u^n-u^{n-1}) over the shortened interval,
        //    overshooting the final state by ~(dtPrev-dt)|v|.
        const double *rptr=Rate.getDataPtr();
        if (dtAttempt==dtPrev) {
            for (int k=0;k<n;k++) {
                const double unew=2.0*uptr[k]-uoptr[k]+dt2*rptr[k];
                uoptr[k]=uptr[k];
                uptr[k]=unew;
            }
        }
        else {
            const double rho=dtAttempt/dtPrev;
            const double w=0.5*dtAttempt*(dtAttempt+dtPrev);
            for (int k=0;k<n;k++) {
                const double unew=uptr[k]+rho*(uptr[k]-uoptr[k])+w*rptr[k];
                uoptr[k]=uptr[k];
                uptr[k]=unew;
            }
        }
        // 3) re-impose the time-independent boundary constraints.
        BCSys.presetSolution(Mesh,DofsPerNode,U,tNew);
        stepTimer.endTimer();

        // ---- blow-up guard (central-difference CFL stability + non-finite scan) ----
        const int slot=(step-1)%kBlowupWindow;
        const double past=(step>kBlowupWindow)?umaxHist[slot]:-1.0; // umax at step-kBlowupWindow
        const double blowupCap=(past>=0.0)?1.0e3*std::max(1.0,past)
                                          :std::numeric_limits<double>::infinity();
        double umax=0.0; bool finiteU=true;
        for (int k=0;k<n;k++) {
            const double av=std::fabs(uptr[k]);
            if (!(av<=std::numeric_limits<double>::max())) finiteU=false; // NaN or inf
            else if (av>umax) umax=av;
        }
        if (!finiteU || umax>blowupCap) {
            std::snprintf(buffer,bufSize,
                "TimeStepping::solveExplicitDynamics: solution blew up at step %d (t=%12.5e, dt=%10.3e; %s "
                "max|U|=%10.3e vs %10.3e a window of %d steps ago). Central difference is only conditionally "
                "stable: reduce dt below the CFL limit dt<~dx/c with c=sqrt(E/rho).",
                step,tNew,dtAttempt,finiteU?"":"non-finite (NaN/inf) entries detected;",umax,
                (past>=0.0)?past:0.0,kBlowupWindow);
            MessagePrinter::printErrorTxt(buffer);
            m_LastConverged=false; m_LastSteps=step-1; m_LastTime=t;
            m_LastCutbacks=0; m_LastGrows=0; m_LastDt=dtAttempt;
            return false;
        }
        umaxHist[slot]=umax;

        t=tNew;
        dtPrev=dtAttempt;

        if (m_Verbose) {
            std::snprintf(buffer,bufSize,
                          "Explicit-dynamics step done, max|U|=%12.5e, elapsed time=%10.3es",
                          umax,stepTimer.getDurationInSecond());
            MessagePrinter::printNormalTxt(buffer,MessageColor::BLUE);
        }

        const bool isFinalStep=(t>=m_TotalTime-tEps);
        if (SaveResults && (step%m_OutputInterval==0 || isFinalStep)) {
            SaveResults(step,t);
        }
        if (m_Verbose) MessagePrinter::printStars();
    }

    m_LastConverged=true;
    m_LastSteps   =step;
    m_LastTime    =t;
    m_LastCutbacks=0;
    m_LastGrows   =0;
    m_LastDt      =m_Dt;
    return true;
}

bool TimeStepping::solveADR(const PDMesh &Mesh,
                            PDOperators &Operators,
                            const ElmtSystem &ElmtSys,
                            const BCSystem &BCSys,
                            PDSystem &PDSys,
                            const int &DofsPerNode,
                            VectorXd &U,
                            VectorXd &Uold,
                            VectorXd &Rate,
                            const std::function<void(const int&,const double&)> &SaveResults) {
    constexpr int bufSize=400;
    char buffer[bufSize];

    auto bail=[&](const string &msg)->bool {
        MessagePrinter::printErrorTxt(msg);
        m_LastConverged=false; m_LastSteps=0; m_LastTime=0.0;
        m_LastCutbacks=0; m_LastGrows=0; m_LastDt=0.0;
        return false;
    };

    if (m_Dt<=0.0 || m_TotalTime<=0.0) {
        return bail("TimeStepping::solveADR: dt and total_time must be positive (got dt="
                    +std::to_string(m_Dt)+", total_time="+std::to_string(m_TotalTime)+")");
    }
    if (m_OutputInterval<1) m_OutputInterval=1;

    const int n=U.getSize();

    // ADR pseudo-dynamics state: half-step velocity and the previous fictitious
    // acceleration (both per DoF), persistent across the relaxation.
    VectorXd velHalfOld; velHalfOld.resize(n); velHalfOld.setToZeros();
    VectorXd accelOld;   accelOld.resize(n);   accelOld.setToZeros();

    // Seed the boundary values at t=0 and emit the initial (undeformed) state.
    BCSys.presetSolution(Mesh,DofsPerNode,U,0.0);
    if (SaveResults) SaveResults(0,0.0);

    LocalElmtInfo info;
    info.UseParallel=m_ParallelAssemble||m_CudaAssemble; // no CUDA dynamics port -> OpenMP CPU
    printExplicitAssembleInfo((m_ParallelAssemble||m_CudaAssemble)?"openmp":"serial",
                              info.UseParallel,"ADR");
    double t   =0.0;
    int    step=0;
    const double tEps=1.0e-12*m_TotalTime;
    Timer stepTimer;

    double *uptr =U.getDataPtr();
    double *vhptr=velHalfOld.getDataPtr();
    double *aoptr=accelOld.getDataPtr();

    // windowed geometric-growth guard + per-entry non-finite scan (see
    // solveExplicit; DR relaxations legitimately grow while loading ramps, so
    // the window factor is 1e6 like the old absolute DR cap).
    constexpr int kBlowupWindow=50;
    double umaxHist[kBlowupWindow]={0.0};

    // Free-DoF mask for the adaptive damping: Underwood's Rayleigh-quotient
    // estimate is defined over FREE DoFs. Preset-driven rows (Dirichlet pins
    // and mirror ghosts) move by imposition, so
    // their u / fictitious v / a would bias cn -- a strongly ramped BC used to
    // inflate the denominator with imposed u^2 and systematically underdamp
    // the relaxation. The pattern is fixed for the run; build it once.
    const std::vector<char> presetMask=BCSys.collectPresetDofMask(Mesh,DofsPerNode);
    const char *maskPtr=presetMask.data();

    while (t<m_TotalTime-tEps) {
        step++;
        const double dt =std::min(m_Dt,m_TotalTime-t);
        const double tNew=t+dt;

        info.Dt      =dt;
        info.T       =tNew;
        info.Timestep=step;

        stepTimer.startTimer();
        // 1) ramp the prescribed boundary displacement to the new pseudo-time.
        BCSys.presetSolution(Mesh,DofsPerNode,U,tNew);
        // 2) fictitious acceleration a=(L+b)/lambda (lambda baked into the rate
        //    by the quasi-static kernel; the material history advances here).
        PDSys.formExplicitDynamicsRate(Mesh,Operators,ElmtSys,info,U,Uold,Rate);
        const double *aptr=Rate.getDataPtr();

        // 3) adaptive damping cn from the Rayleigh quotient of the local
        //    stiffness (a = -K u / lambda) over the FREE DoFs, Underwood/ADR
        //    (preset-controlled rows are excluded via the mask above).
        double cn1=0.0,cn2=0.0;
        for (int k=0;k<n;k++) {
            if (maskPtr[k]) continue;
            cn2+=uptr[k]*uptr[k];
            if (std::fabs(vhptr[k])>1.0e-12) {
                cn1-=uptr[k]*uptr[k]*(aptr[k]-aoptr[k])/(dt*vhptr[k]);
            }
        }
        double cn=0.0;
        if (std::fabs(cn2)>1.0e-12) {
            const double r=cn1/cn2;
            if (r>0.0) cn=2.0*std::sqrt(r);
        }
        if (cn>2.0) cn=1.9;

        // 4) central-difference DR update; first step uses the half-step start.
        if (step==1) {
            for (int k=0;k<n;k++) {
                const double vh=0.5*dt*aptr[k];
                uptr[k]+=vh*dt;
                vhptr[k]=vh; aoptr[k]=aptr[k];
            }
        }
        else {
            const double den=2.0+cn*dt, num=2.0-cn*dt;
            for (int k=0;k<n;k++) {
                const double vh=(num*vhptr[k]+2.0*dt*aptr[k])/den;
                uptr[k]+=vh*dt;
                vhptr[k]=vh; aoptr[k]=aptr[k];
            }
        }
        // 5) re-pin the boundary (the DR update moved it too) for clean output.
        BCSys.presetSolution(Mesh,DofsPerNode,U,tNew);
        stepTimer.endTimer();

        const int slot=(step-1)%kBlowupWindow;
        const double past=(step>kBlowupWindow)?umaxHist[slot]:-1.0; // umax at step-kBlowupWindow
        const double blowupCap=(past>=0.0)?1.0e6*std::max(1.0,past)
                                          :std::numeric_limits<double>::infinity();
        double umax=0.0; bool finiteU=true;
        for (int k=0;k<n;k++) {
            const double av=std::fabs(uptr[k]);
            if (!(av<=std::numeric_limits<double>::max())) finiteU=false; // NaN or inf
            else if (av>umax) umax=av;
        }
        if (!finiteU || umax>blowupCap) {
            std::snprintf(buffer,bufSize,
                "TimeStepping::solveADR: relaxation diverged at step %d (t=%12.5e; %s max|U|=%10.3e "
                "vs %10.3e a window of %d steps ago). The ADR fictitious mass is undersized or the "
                "load increment per step is too large; increase total_time (more, smaller "
                "increments) or check the kernel's stable-mass estimate.",
                step,tNew,finiteU?"":"non-finite (NaN/inf) entries detected;",
                umax,(past>=0.0)?past:0.0,kBlowupWindow);
            MessagePrinter::printErrorTxt(buffer);
            m_LastConverged=false; m_LastSteps=step-1; m_LastTime=t;
            m_LastCutbacks=0; m_LastGrows=0; m_LastDt=dt;
            return false;
        }
        umaxHist[slot]=umax;

        t=tNew;

        if (m_Verbose && (step%m_OutputInterval==0)) {
            std::snprintf(buffer,bufSize,
                          "ADR pseudo-step %8d/%g, cn=%8.4f, max|U|=%12.5e, elapsed=%9.3es",
                          step,m_TotalTime/m_Dt,cn,umax,stepTimer.getDurationInSecond());
            MessagePrinter::printNormalTxt(buffer,MessageColor::BLUE);
        }

        const bool isFinalStep=(t>=m_TotalTime-tEps);
        if (SaveResults && (step%m_OutputInterval==0 || isFinalStep)) {
            SaveResults(step,t);
        }
    }

    // Equilibrium-quality report: DR has no convergence test (the ramp runs a
    // fixed pseudo-step count), so at least tell the user how "static" the
    // final state is -- both norms -> 0 as the relaxation converges.
    {
        double vmax=0.0,amax=0.0;
        for (int k=0;k<n;k++) {
            vmax=std::max(vmax,std::fabs(vhptr[k]));
            amax=std::max(amax,std::fabs(aoptr[k]));
        }
        std::snprintf(buffer,bufSize,
                      "ADR finished: max|v_half|=%10.3e, max|a|=%10.3e (equilibrium quality; "
                      "increase total_time if these are not small)",vmax,amax);
        MessagePrinter::printNormalTxt(buffer);
    }

    m_LastConverged=true;
    m_LastSteps   =step;
    m_LastTime    =t;
    m_LastCutbacks=0; m_LastGrows=0; m_LastDt=m_Dt;
    return true;
}


void TimeStepping::printTimeSteppingInfo() const {
    constexpr int bufSize=200;
    char buffer[bufSize];

    const char *scheme="Time-stepping info (Backward Euler):";
    if (m_Explicit) {
        if (m_QuasiStatic)
            scheme="Time-stepping info (Adaptive Dynamic Relaxation, quasi-static, matrix-free):";
        else
            scheme=(m_TimeOrder>=2)
                ? "Time-stepping info (central difference, explicit dynamics, matrix-free):"
                : "Time-stepping info (Forward Euler, explicit, matrix-free):";
    }
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt(scheme);
    std::snprintf(buffer,bufSize,
                  "  dt=%12.5e, total_time=%12.5e",
                  m_Dt,m_TotalTime);
    MessagePrinter::printNormalTxt(buffer);
    if (m_Explicit) {
        MessagePrinter::printNormalTxt("  fixed dt (CFL-limited); adaptive control not applicable");
        MessagePrinter::printStars();
        return;
    }
    if (m_Adaptive) {
        std::snprintf(buffer,bufSize,
                      "  adaptive=on, optimal_iters=%d, growth=%5.3f, cutback=%5.3f, max_cb=%d",
                      m_OptimalIters,m_GrowthFactor,m_CutbackFactor,m_MaxCutbacks);
        MessagePrinter::printNormalTxt(buffer);
        std::snprintf(buffer,bufSize,
                      "  min_dt=%10.3e, max_dt=%10.3e (0 means use initial dt)",
                      m_MinDt,m_MaxDt);
        MessagePrinter::printNormalTxt(buffer);
    }
    else {
        MessagePrinter::printNormalTxt("  adaptive=off");
    }
    MessagePrinter::printStars();
}
