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
//+++ Date    : 2026.05.30
//+++ Function: plain-old-data bridge between the host glue
//+++           (FormResidualAndJacobianCUDA.cpp, compiled by the C++
//+++           compiler) and the CUDA device kernels
//+++           (CudaAssembleKernels.cu, compiled by nvcc). Only POD
//+++           types cross this boundary so nvcc never has to parse the
//+++           framework's C++ headers.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

namespace perix_cuda {

/** element kernels with a GPU device port (others fall back to the CPU). */
enum AssembleElmtType {
    ELMT_POISSON      = 0,
    ELMT_DIFFUSION    = 1    ///< transient: uses dt + Uold
};

/**
 * 0-based PDDO operator-slot indices for the derivatives the device
 * kernels read (resolved host-side from the multi-index table); -1 if a
 * given operator is absent for the configured order.
 */
struct OpSlots {
    int dx  = -1;  ///< d/dx     (1,0,0)
    int dy  = -1;  ///< d/dy     (0,1,0)
    int dz  = -1;  ///< d/dz     (0,0,1)   (3D)
    int dxx = -1;  ///< d2/dx2   (2,0,0)
    int dxy = -1;  ///< d2/dxdy  (1,1,0)
    int dyy = -1;  ///< d2/dy2   (0,2,0)
    int dxz = -1;  ///< d2/dxdz  (1,0,1)   (3D)
    int dyz = -1;  ///< d2/dydz  (0,1,1)   (3D)
    int dzz = -1;  ///< d2/dz2   (0,0,2)   (3D)
};

/** per-element physical parameters (each kernel uses its own subset). */
struct ElmtParams {
    int    type = ELMT_POISSON;
    int    ndof = 1;
    int    dim  = 2;                      ///< spatial dimension (2 or 3)
    double sigma = 0.0, f = 0.0;          ///< poisson: sigma*Lap u + f ; diffusion: source f
    double D = 0.0;                          ///< diffusion coefficient
    double dt = 0.0;                      ///< transient step size; 0 => static
};

/**
 * Read-only description of the PD mesh + PDDO multi-index tables that is
 * constant for the whole run. Uploaded to the device once (the kernels
 * cache it and re-upload only if the sizes change).
 */
struct AssembleMesh {
    int           N    = 0;       ///< number of PD nodes
    int           nnz  = 0;       ///< stored nonzeros of the (N*ndof) CSR matrix
    const int    *csr_off  = nullptr;  ///< (N*ndof+1) CSR row offsets (0-based)
    const int    *csr_col  = nullptr;  ///< (nnz)  CSR column indices (0-based, sorted per row)
    const int    *neigh_off= nullptr;  ///< (N+1) neighbor-list offsets
    const int    *neigh_id = nullptr;  ///< flattened 1-based neighbor node ids
    const double *coords   = nullptr;  ///< (3*N) node coordinates (x,y,z interleaved)
    const double *vols     = nullptr;  ///< (N) node volumes
    const int    *qp       = nullptr;  ///< (nop) PDDO multi-index p per operator slot
    const int    *qq       = nullptr;  ///< (nop) PDDO multi-index q per operator slot
    const int    *qr       = nullptr;  ///< (nop) PDDO multi-index r (z) per operator slot (3D)
    const double *factprod = nullptr;  ///< (nop) prod(factorial(q_k)) per slot
    int           nop      = 0;        ///< number of PDDO operator slots
    int           dim      = 2;        ///< spatial dimension (2 or 3)
    double        delta    = 0.0;      ///< base horizon radius (uniform horizon)
    double        dx_char  = 0.0;      ///< max(DX,DY,DZ) for the volume correction (uniform)
    const double *nodeHorizon = nullptr;///< (N) per-node horizon (variable horizon), or null = uniform
    const double *nodeSpacing = nullptr;///< (N) per-node spacing (variable horizon), or null = uniform
};

/**
 * Assemble the residual/Jacobian for a supported element on the GPU. The
 * mesh/structure in `m` is uploaded once and cached; `u` (and, for
 * transient kernels, `uold`) is uploaded every call; the assembled CSR
 * values and rhs are written back to the host arrays `out_Kvals` (nnz)
 * and `out_rhs` (N*ndof).
 * @param uold previous-step solution (may be null for static kernels)
 * @return true on success; false if the GPU path could not run (the caller
 *         then falls back to the CPU assembler).
 */
bool assemble(const AssembleMesh &m, const ElmtParams &p, const OpSlots &s,
              const double *u, const double *uold,
              double *out_Kvals, double *out_rhs);

/** per-element parameters for the CONSERVATIVE Cahn-Hilliard device port (device
 *  twin of the flux-conservative CahnHilliardElement). kappa/chi/dt mirror the
 *  CPU kernel; the degenerate mobility M(c)=M0 c(1-c) is Picard-frozen per node
 *  on the host and travels in CahnHilliardMesh.mnode (so it carries no Jacobian
 *  term, exactly as the CPU per-bond callback treats it). */
struct CahnHilliardParams {
    double kappa = 0.0;   ///< gradient-energy coefficient
    double chi   = 0.0;   ///< Flory-Huggins interaction (enters f'(c), f''(c))
    double dt    = 1.0;   ///< backward-Euler step (transient-only)
};

/** reference-geometry + Picard-frozen-mobility view for the conservative
 *  Cahn-Hilliard device port. Every array is HOST-built ONCE by the element in
 *  preprocessIteration(): the lambda surface-corrected per-bond PDDO Laplacian
 *  g2_ij, the reverse-bond map, node volumes and boundary-ghost flags; only the
 *  frozen nodal mobility mnode + u/uold change per Newton step. off/nbr are the
 *  element's OWN neighbour CSR (1-based nbr), aligned with lapbond/rev -- the
 *  device kernel walks these and resolves the matrix slot via devFindSlot, so
 *  the neighbour CSR need not match the matrix CSR ordering. */
struct CahnHilliardMesh {
    int N = 0, nbtot = 0;
    const int    *off     = nullptr;  ///< N+1 neighbour-CSR offsets
    const int    *nbr     = nullptr;  ///< nbtot 1-based neighbour ids
    const int    *rev     = nullptr;  ///< nbtot reverse-bond flat index (-1 if unpaired)
    const double *lapbond = nullptr;  ///< nbtot lambda-corrected Laplacian g2_ij
    const double *vol     = nullptr;  ///< N node volumes
    const char   *isghost = nullptr;  ///< N boundary-ghost flags (bond to a ghost j is dropped)
    const double *mnode   = nullptr;  ///< N Picard-frozen mobility M(c_i)=M0 c_i(1-c_i)
};

/**
 * Assemble the CONSERVATIVE Cahn-Hilliard residual + Jacobian on the GPU (device
 * twin of the flux-conservative CahnHilliardElement). One thread per node owns
 * its 2 (c,mu) rows -> row-centred, no atomics. The c-equation uses the
 * antisymmetric species flux w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji) (bonds
 * to boundary ghosts dropped); the mu-equation uses +kappa g2_ij (c_j-c_i) plus
 * the nodal mu - f'(c); the nodal block carries (c-c_old)/dt, -f''(c) and 1/dt.
 * The host-built geometry + frozen mobility are uploaded once (re-uploaded on a
 * size change); u/uold each call. Fills out_Kvals (nnz, Newton sign K=-dF/dU)
 * and out_rhs (N*2, +F).
 * @return true on success; false if the GPU path could not run (CPU fallback).
 */
bool assembleCahnHilliard(const CahnHilliardMesh &m, const CahnHilliardParams &p,
                          const int *csr_off, const int *csr_col, int nnz,
                          const double *u, const double *uold,
                          double *out_Kvals, double *out_rhs);

/** per-element parameters for the small-strain stress-coupled Cahn-Hilliard
 *  device port (device twin of StressCahnHilliardElement). C11/C12/C66 are the
 *  plane-stress/strain (or 3D) elastic constants; sigma_h = Kh*(eps_xx+eps_yy
 *  [+eps_zz]) + Ch*(c-cref) is the hydrostatic stress feeding the chemical
 *  potential; A is the chemo-mechanical body-force coefficient (-A grad c).
 *  The degenerate mobility M(c)=D c(1-c) is Picard-frozen per node on the host
 *  and travels as CahnHilliardMesh-style mnode (no Jacobian term). */
struct StressCahnHilliardParams {
    int    dim = 2, ndof = 4;
    double C11 = 0.0, C12 = 0.0, C66 = 0.0;   ///< elastic stiffness
    double A   = 0.0, Kh = 0.0, Ch = 0.0;     ///< chemo-mechanical coupling
    double Omega = 0.0, cref = 0.0;           ///< swelling
    double chi = 0.0, kappa = 0.0;            ///< CH interaction / gradient energy
    double dt  = 1.0;                          ///< backward-Euler step (transient-only)
    double kres = 1.0e-8;                      ///< residual-stiffness floor for a broken bond
    double rho  = 0.0;                          ///< >0 => backward-Euler inertia on the u rows
};

/**
 * Assemble the small-strain stress-coupled Cahn-Hilliard residual + Jacobian on
 * the GPU (device twin of StressCahnHilliardElement). One thread per node owns
 * its (c,mu,ux,uy[,uz]) rows -> row-centred, no atomics. It builds node i's PDDO
 * moment matrix on the device (shared devBuildNodeInverse/devBondOps, exactly
 * like the generic PDDO assembler) for the mechanics + chemo-mechanical operators
 * (Gx,Gy,Gxx,Gyy,Gxy...), and uses the HOST-prebuilt conservative species-flux
 * arrays for the c-equation. The lambda surface-corrected per-bond Laplacian
 * drives both the conservative species flux and the gradient-energy term.
 * `rev` is the reverse-bond map, `isghost` the boundary-ghost flags,
 * `mnode` the Picard-frozen mobility. Reuses the generic AssembleMesh (geometry +
 * matrix CSR) so the dispatch shares the same setup. Fills out_Kvals (nnz,
 * K=-dF/dU) and out_rhs (N*ndof, +F).
 *
 * Pass `health` (nbtot, per-bond in {0,1},
 * frozen/committed host-side, aligned with AssembleMesh.neigh_*) to gate the
 * MECHANICAL bond terms (elastic + eigenstress + sigma_h coupling; NOT the
 * species flux or kappa-Laplacian) by g = health>0 ? 1 : p.kres; and `vel`
 * (N*dim committed velocity) with p.rho>0 to add the backward-Euler inertia
 * R_u += -rho(u-u_old-vel*dt)/dt^2 on the displacement rows. Both null =>
 * plain stress_cahnhilliard (g==1, no inertia), bit-identical to before.
 * @return true on success; false if the GPU path could not run (CPU fallback).
 */
bool assembleStressCahnHilliard(const AssembleMesh &m, const StressCahnHilliardParams &p,
                                const OpSlots &s,
                                const int *rev, const double *lapbond,
                                const char *isghost, const double *mnode,
                                const double *health, const double *vel,
                                const double *u, const double *uold,
                                double *out_Kvals, double *out_rhs);

/** Parameters for the implicit PDDO dynamic-fracture device assembler. */
struct CachedParams {
    int    ndof = 2;
    int    dim  = 2;
    int    ncache = 3;              ///< committed velocity slots per node
    double dt = 0.0;
    double C11 = 0.0, C12 = 0.0, C66 = 0.0;
    double rho = 0.0;
    double bx = 0.0, by = 0.0, bz = 0.0;
    int    nbcache = 1;             ///< one health scalar per bond
};

/**
 * Shared row-centred "cached PDDO callback" assembler on the GPU. One thread per
 * node owns its ndof rows; it builds node i's PDDO moment matrix on-device
 * (devBuildNodeInverse/devBondOps) for the operators, reads node i's Picard-frozen
 * committed velocity cache slice `nodecache + i*ncache` and evaluates the
 * dynamic-fracture bond and nodal physics. Reuses the
 * generic AssembleMesh (geometry + matrix CSR). uold may be null for static types.
 * `bondcache` stores the frozen per-bond health.
 * Fills out_Kvals (nnz, K=-dF/dU) and out_rhs (N*ndof, +F).
 * @return true on success; false if the GPU path could not run (CPU fallback).
 */
bool assembleCached(const AssembleMesh &m, const CachedParams &p, const OpSlots &s,
                    const double *nodecache, const double *bondcache,
                    const double *u, const double *uold,
                    double *out_Kvals, double *out_rhs);

/** per-element parameters for the explicit strong-form PDDO fracture device
 *  port (resolved host-side from ExplicitPDDOFracElement). The Navier-Cauchy
 *  constants C11/C12/C66 and the critical-stretch ingredients mirror the CPU
 *  kernel; the mesh + PDDO multi-index tables travel in the AssembleMesh. */
struct ExplicitPDDOFracParams {
    int    dim = 2;
    double C11 = 0.0, C12 = 0.0, C66 = 0.0;   // Navier-Cauchy elastic constants
    double rho = 1.0;
    double bx = 0.0, by = 0.0, bz = 0.0;      // body force / reference volume
    double sc = 0.0;                          // uniform critical stretch (0 => off)
    int    tension_only = 1, damage_on = 1;
    // per-bond critical stretch under a variable horizon (else sc is used)
    int    plane_stress = 0;
    double G0 = 0.0, E = 0.0;
};

/**
 * Matrix-free EXPLICIT strong-form PDDO fracture acceleration on the GPU (the
 * device port of ExplicitPDDOFracElement). One thread per source node builds the
 * PDDO moment matrix (shared devBuildNodeInverse), evaluates the per-bond PDDO
 * operators, updates the irreversible per-bond tensile critical-stretch damage,
 * and assembles the health-gated strong-form Navier-Cauchy nodal force into the
 * acceleration a_i = (L_i + b)/rho written to accel_out (N*dim). The per-bond
 * damage health_inout (nbtot, aligned with the mesh neighbour list) is PERSISTENT
 * device state: uploaded once from the host seed, evolved in place, and copied
 * back ONLY on steps where a bond breaks (gated by a device flag; *health_changed
 * is 1 then, 0 otherwise). accel_out is copied back every call.
 * @return true on success; false if the GPU path could not run (caller falls back
 *         to the CPU preprocessIteration).
 */
bool assembleExplicitPDDOFracAccel(const AssembleMesh &m, const ExplicitPDDOFracParams &p,
                                   const OpSlots &s, const double *u,
                                   double *health_inout, double *accel_out,
                                   int *health_changed);

/** Release the cached device buffers (optional; also freed at exit). */
void shutdown();

} // namespace perix_cuda
