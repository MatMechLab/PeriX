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
//+++ Function: CUDA device kernels for the GPU residual/Jacobian
//+++           assembler. One thread per source PD node builds the PDDO
//+++           moment matrix, inverts it, evaluates the per-bond
//+++           operators, and scatters the per-element physics into the
//+++           ndof rows owned by that node. Because every contribution
//+++           of node i lands in node-i rows, threads own disjoint rows
//+++           and need no atomics. The per-element physics is selected
//+++           by an element-type id (no virtual dispatch on the GPU).
//+++           Compiled by nvcc; only POD crosses the host boundary
//+++           (see include/PDSystem/CudaAssemble.h).
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "PDSystem/CudaAssemble.h"

namespace {

// Largest PDDO operator count (2D order <= 4 => nop <= 14) and largest
// DoF/node count (stress-Cahn-Hilliard = 4) handled on the GPU path;
// anything larger falls back to the CPU assembler (checked host-side).
constexpr int kMaxNop = 15;
constexpr int kMaxDof = 5;   // up to ndof=dim+2 (aniso-frac stress-CH in 3D)

__device__ inline double devWeight(double xsi_norm, double delta) {
    const double r = 2.0 * xsi_norm / delta;
    return exp(-r * r);
}

__device__ inline double devVolCorrection(double xsi_norm, double delta, double dx_char) {
    if (dx_char <= 0.0) return 1.0;
    if (xsi_norm <= delta - 0.5 * dx_char) return 1.0;
    double vc = (delta - xsi_norm + 0.5 * dx_char) / dx_char;
    if (vc < 0.0) vc = 0.0;
    if (vc > 1.0) vc = 1.0;
    return vc;
}

__device__ inline void devMonomials(double ex, double ey, double ez, int nop,
                                    const int *qp, const int *qq, const int *qr, double *mono) {
    for (int k = 0; k < nop; ++k) {
        double m = 1.0;
        for (int t = 0; t < qp[k]; ++t) m *= ex;
        for (int t = 0; t < qq[k]; ++t) m *= ey;
        for (int t = 0; t < qr[k]; ++t) m *= ez;
        mono[k] = m;
    }
}

// In-place Gauss-Jordan inverse of nop x nop A (row-major) with partial
// pivoting; result in Inv. Returns false if singular.
__device__ inline bool devInvert(double *A, double *Inv, int nop) {
    for (int r = 0; r < nop; ++r)
        for (int c = 0; c < nop; ++c)
            Inv[r * nop + c] = (r == c) ? 1.0 : 0.0;
    for (int col = 0; col < nop; ++col) {
        int piv = col;
        double best = fabs(A[col * nop + col]);
        for (int r = col + 1; r < nop; ++r) {
            double v = fabs(A[r * nop + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best <= 0.0) return false;
        if (piv != col) {
            for (int c = 0; c < nop; ++c) {
                double t = A[col * nop + c]; A[col * nop + c] = A[piv * nop + c]; A[piv * nop + c] = t;
                t = Inv[col * nop + c]; Inv[col * nop + c] = Inv[piv * nop + c]; Inv[piv * nop + c] = t;
            }
        }
        const double invd = 1.0 / A[col * nop + col];
        for (int c = 0; c < nop; ++c) { A[col * nop + c] *= invd; Inv[col * nop + c] *= invd; }
        for (int r = 0; r < nop; ++r) {
            if (r == col) continue;
            const double fr = A[r * nop + col];
            if (fr == 0.0) continue;
            for (int c = 0; c < nop; ++c) {
                A[r * nop + c]   -= fr * A[col * nop + c];
                Inv[r * nop + c] -= fr * Inv[col * nop + c];
            }
        }
    }
    return true;
}

// Build node i's inverse PDDO moment matrix Inv and the 1/delta^|q| weights
// idpow, 3D- and variable-horizon-aware (mirrors PDOperators::calcAMatrix).
// Writes the node's own delta_i / dx_char_i for the per-bond operator eval.
// Returns false if the moment matrix is singular.
__device__ inline bool devBuildNodeInverse(int i, int dim, int nop,
        const int *qp, const int *qq, const int *qr,
        const double *coords, const double *vols,
        const int *neigh_off, const int *neigh_id,
        const double *nodeHorizon, const double *nodeSpacing,
        double delta_g, double dx_g,
        double *A, double *Inv, double *idpow, double &delta_i, double &dxc_i) {
    delta_i = nodeHorizon ? nodeHorizon[i] : delta_g;
    dxc_i   = nodeHorizon ? nodeSpacing[i] : dx_g;
    const double inv_delta = 1.0 / delta_i;
    for (int k = 0; k < nop; ++k) {
        double v = 1.0; const int tot = qp[k] + qq[k] + qr[k];
        for (int t = 0; t < tot; ++t) v *= inv_delta;
        idpow[k] = v;
    }
    for (int k = 0; k < nop*nop; ++k) A[k] = 0.0;
    const double xi0 = coords[i*3+0], yi0 = coords[i*3+1], zi0 = coords[i*3+2];
    const int nb = neigh_off[i], ne = neigh_off[i+1];
    double mono[kMaxNop];
    for (int t = nb; t < ne; ++t) {
        const int j = neigh_id[t] - 1;
        const double dx = coords[j*3+0]-xi0, dy = coords[j*3+1]-yi0;
        const double dz = (dim>=3) ? coords[j*3+2]-zi0 : 0.0;
        const double xn = sqrt(dx*dx + dy*dy + dz*dz);
        if (xn <= 0.0) continue;
        if (nodeHorizon && xn > delta_i) continue;   // node i fits its OWN horizon
        const double bw = devWeight(xn, delta_i) * devVolCorrection(xn, delta_i, dxc_i) * vols[j];
        if (bw <= 0.0) continue;
        devMonomials(dx*inv_delta, dy*inv_delta, dz*inv_delta, nop, qp, qq, qr, mono);
        for (int a = 0; a < nop; ++a)
            for (int b = 0; b < nop; ++b) A[a*nop+b] += bw * mono[a] * mono[b];
    }
    return devInvert(A, Inv, nop);
}

// Evaluate the per-bond PDDO operators op[k] for bond (i->j) given node i's
// inverse moment matrix Inv and weights idpow (from devBuildNodeInverse).
__device__ inline void devBondOps(double dx, double dy, double dz,
        int dim, int nop, const int *qp, const int *qq, const int *qr,
        const double *factprod, const double *Inv, const double *idpow,
        double delta_i, double dxc_i, double volJ, double *op) {
    const double inv_delta = 1.0 / delta_i;
    const double xn = sqrt(dx*dx + dy*dy + (dim>=3?dz*dz:0.0));
    const double bw = devWeight(xn, delta_i) * devVolCorrection(xn, delta_i, dxc_i) * volJ;
    double mono[kMaxNop];
    devMonomials(dx*inv_delta, dy*inv_delta, (dim>=3?dz*inv_delta:0.0), nop, qp, qq, qr, mono);
    for (int k = 0; k < nop; ++k) {
        double sum = 0.0;
        for (int a = 0; a < nop; ++a) sum += Inv[a*nop+k] * mono[a];
        op[k] = factprod[k] * sum * bw * idpow[k];
    }
}

__device__ inline int devFindSlot(const int *col, int begin, int end, int target) {
    int lo = begin, hi = end - 1;
    while (lo <= hi) {
        // overflow-safe midpoint: lo+hi wraps once global CSR offsets pass
        // INT_MAX/2 (~1.07e9 nnz), turning mid negative -> OOB col[mid].
        const int mid = lo + ((hi - lo) >> 1);
        const int c = col[mid];
        if (c == target) return mid;
        if (c < target) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

// ---- device-resident PDDO operator cache -------------------------------------
// The per-bond operators depend only on the reference geometry (coordinates,
// volumes, families, horizon, volume correction), all frozen for the run, yet
// the implicit assembly kernels used to rebuild the per-node moment matrix and
// re-evaluate every bond operator on every launch. Build them ONCE here with
// the exact same device functions and replay them afterwards -- the stored
// values are the bits the kernels would have computed, so the assembled system
// is bit-identical. This generalises the pattern the explicit-PDDO-fracture
// path already uses (explicitPDDOFracOpBuildKernel), storing the FULL op[nop]
// vector so one cache serves every consumer kernel. Disable (recompute every
// launch, the original behaviour) with PERIX_CUDA_OPCACHE=0.
// Default policy (PERIX_CUDA_OPCACHE unset): cache when the rebuild is the
// expensive side. Measured on an RTX 4090 (3-run means, byte-identical
// results either way): the matrix-free explicit path gains x1.7 wall (many
// small per-step launches dominated by the moment build); implicit 3D kinds
// (nop>=9) gain 5-9% of the assemble bucket; implicit 2D kinds (nop<=6) LOSE
// ~7% of it, because for a small op vector the register-resident recompute is
// cheaper than the extra global-memory reads. PERIX_CUDA_OPCACHE=1 forces the
// cache everywhere, =0 disables it everywhere (the original behaviour).
inline bool devOpCacheEnabled(int nop, bool isExplicitPath) {
    static const int mode = [] {
        const char *e = std::getenv("PERIX_CUDA_OPCACHE");
        if (e && e[0] == '0') return 0;   // force off
        if (e && e[0] == '1') return 1;   // force on
        return 2;                         // heuristic default
    }();
    if (mode == 0) return false;
    if (mode == 1) return true;
    return isExplicitPath || nop >= 9;
}

// One thread per node: moment matrix + inverse once, then every bond's full
// operator vector. Layout: ops[bond*nop + k], bond in global family order.
// opsValid[i]=0 marks a singular moment system: the consumer kernels replay
// their existing failure branch (zero the node's rows) without recomputing.
// A zero-length/far/coincident bond stores ZEROS -- exactly the value the
// zero-op branch of the flux kernels uses; the op-only kernels `continue`
// before reading it, so the stored zeros are never consumed there.
__global__ void opsBuildKernel(int N, int dim, int nop,
        const int *qp, const int *qq, const int *qr, const double *factprod,
        const double *coords, const double *vols,
        const int *neigh_off, const int *neigh_id,
        const double *nodeHorizon, const double *nodeSpacing,
        double delta_g, double dx_g,
        double *ops, unsigned char *opsValid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int nb = neigh_off[i], ne = neigh_off[i + 1];
    double A[kMaxNop * kMaxNop], Inv[kMaxNop * kMaxNop], idpow[kMaxNop];
    double delta_i, dxc_i;
    if (!devBuildNodeInverse(i, dim, nop, qp, qq, qr, coords, vols, neigh_off, neigh_id,
                             nodeHorizon, nodeSpacing, delta_g, dx_g,
                             A, Inv, idpow, delta_i, dxc_i)) {
        opsValid[i] = 0;
        for (int t = nb; t < ne; ++t)
            for (int k = 0; k < nop; ++k) ops[static_cast<size_t>(t) * nop + k] = 0.0;
        return;
    }
    opsValid[i] = 1;
    const double xi0 = coords[i*3+0], yi0 = coords[i*3+1], zi0 = coords[i*3+2];
    for (int t = nb; t < ne; ++t) {
        const int j = neigh_id[t] - 1;
        const double dx = coords[j*3+0]-xi0, dy = coords[j*3+1]-yi0;
        const double dz = (dim>=3) ? coords[j*3+2]-zi0 : 0.0;
        const double xn = sqrt(dx*dx + dy*dy + (dim>=3?dz*dz:0.0));
        double *oc = &ops[static_cast<size_t>(t) * nop];
        if (xn <= 0.0 || (nodeHorizon && xn > delta_i)) {
            for (int k = 0; k < nop; ++k) oc[k] = 0.0;
            continue;
        }
        devBondOps(dx, dy, dz, dim, nop, qp, qq, qr, factprod, Inv, idpow,
                   delta_i, dxc_i, vols[j], oc);
    }
}

// Cahn-Hilliard double-well derivatives (device port; matches the CPU
// helpers, including the (eps,1-eps) clamp that keeps log / 1/c(1-c) finite).
__device__ inline double devClampC(double c) {
    const double eps = 1.0e-8;
    return fmax(eps, fmin(1.0 - eps, c));
}
__device__ inline double devFPrime(double c, double chi) {
    const double cc = devClampC(c);
    return log(cc / (1.0 - cc)) + chi * (1.0 - 2.0 * cc);
}
__device__ inline double devFDoublePrime(double c, double chi) {
    const double cc = devClampC(c);
    return 1.0 / (cc * (1.0 - cc)) - 2.0 * chi;
}
// ---- per-element physics (device ports of the CPU element kernels) ----
__device__ inline void physBond(const perix_cuda::ElmtParams &p,
                                 const perix_cuda::OpSlots &s,
                                 const double *op, const double *uI, const double *uJ,
                                 double *R, double *KII, double *KIJ) {
    const bool d3 = (p.dim >= 3);
    if (p.type == perix_cuda::ELMT_POISSON) {
        const double lap = op[s.dxx] + op[s.dyy] + (d3 ? op[s.dzz] : 0.0);
        const double coeff = p.sigma * lap;
        R[0] = coeff * (uJ[0] - uI[0]); KII[0] = -coeff; KIJ[0] = coeff;
    }
    else if (p.type == perix_cuda::ELMT_DIFFUSION) {
        const double lap = op[s.dxx] + op[s.dyy] + (d3 ? op[s.dzz] : 0.0);
        const double coeff = -p.D * lap;
        R[0] = coeff * (uJ[0] - uI[0]); KII[0] = -coeff; KIJ[0] = coeff;
    }
}

__device__ inline void physNodal(const perix_cuda::ElmtParams &p,
                                 const double *uI, const double *uoldI,
                                 double *Rn, double *Kn) {
    const int ndof = p.ndof;
    for (int k = 0; k < ndof * ndof; ++k) Kn[k] = 0.0;   // dim-agnostic zero seed
    if (p.type == perix_cuda::ELMT_POISSON) {
        Rn[0] = p.f;
    }
    else if (p.type == perix_cuda::ELMT_DIFFUSION) {
        Rn[0] = (uI[0] - uoldI[0]) / p.dt - p.f;
        Kn[0] = 1.0 / p.dt;
    }
}

__global__ void assembleKernel(int N, perix_cuda::ElmtParams p, perix_cuda::OpSlots s,
                               const int *csr_off, const int *csr_col,
                               const int *neigh_off, const int *neigh_id,
                               const double *coords, const double *vols,
                               const int *qp, const int *qq, const int *qr, const double *factprod,
                               int nop, double delta, double dx_char,
                               const double *nodeHorizon, const double *nodeSpacing,
                               const double *opsCache, const unsigned char *opsValid,
                               const double *u, const double *uold,
                               double *Kvals, double *rhs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int ndof = p.ndof;
    const int dim  = p.dim;
    const int rowBase = i * ndof;

    double A[kMaxNop * kMaxNop];
    double Inv[kMaxNop * kMaxNop];
    double op[kMaxNop];
    double idpow[kMaxNop];
    double delta_i, dxc_i;

    // ---- node i's inverse PDDO moment matrix + 1/delta^|q| weights (3D + varH);
    //      with the operator cache the moment solve already ran once in
    //      opsBuildKernel -- only its validity flag and delta_i are needed ----
    if (opsCache ? (opsValid[i] == 0)
                 : !devBuildNodeInverse(i, dim, nop, qp, qq, qr, coords, vols, neigh_off, neigh_id,
                                        nodeHorizon, nodeSpacing, delta, dx_char,
                                        A, Inv, idpow, delta_i, dxc_i)) {
        for (int a = 0; a < ndof; ++a) {
            for (int sl = csr_off[rowBase + a]; sl < csr_off[rowBase + a + 1]; ++sl) Kvals[sl] = 0.0;
            rhs[rowBase + a] = 0.0;
        }
        return;
    }
    if (opsCache) {
        delta_i = nodeHorizon ? nodeHorizon[i] : delta;   // far-bond guard only
        dxc_i   = 0.0;
    }

    const double xi0 = coords[i*3+0], yi0 = coords[i*3+1], zi0 = coords[i*3+2];
    const int nb = neigh_off[i];
    const int ne = neigh_off[i + 1];

    // gather nodal solution (+ previous step for transient)
    double uI[kMaxDof], uoldI[kMaxDof];
    for (int a = 0; a < ndof; ++a) {
        uI[a] = u[rowBase + a];
        uoldI[a] = uold ? uold[rowBase + a] : 0.0;
    }

    // nodal contribution seeds the diagonal block + rhs
    double Rn[kMaxDof], Kn[kMaxDof * kMaxDof];
    physNodal(p, uI, uoldI, Rn, Kn);
    double rhs_loc[kMaxDof], kii[kMaxDof * kMaxDof];
    for (int a = 0; a < ndof; ++a) {
        rhs_loc[a] = Rn[a];
        for (int b = 0; b < ndof; ++b) kii[a * ndof + b] = -Kn[a * ndof + b];
    }

    // zero the ndof rows owned by node i
    for (int a = 0; a < ndof; ++a)
        for (int sl = csr_off[rowBase + a]; sl < csr_off[rowBase + a + 1]; ++sl) Kvals[sl] = 0.0;

    // ---- per-bond operators + physics, scatter off-diagonal blocks ----
    for (int t = nb; t < ne; ++t) {
        const int j = neigh_id[t] - 1;
        const double dx = coords[j*3+0]-xi0, dy = coords[j*3+1]-yi0;
        const double dz = (dim>=3) ? coords[j*3+2]-zi0 : 0.0;
        const double xn = sqrt(dx*dx + dy*dy + (dim>=3?dz*dz:0.0));
        if (xn <= 0.0) continue;
        if (nodeHorizon && xn > delta_i) continue;   // varH: node i's own family
        if (opsCache) {
            const double *oc = &opsCache[static_cast<size_t>(t) * nop];
            for (int k = 0; k < nop; ++k) op[k] = oc[k];
        } else {
            devBondOps(dx, dy, dz, dim, nop, qp, qq, qr, factprod, Inv, idpow,
                       delta_i, dxc_i, vols[j], op);
        }

        double uJ[kMaxDof];
        const int colBase = j * ndof;
        for (int b = 0; b < ndof; ++b) uJ[b] = u[colBase + b];

        double R[kMaxDof], KII[kMaxDof * kMaxDof], KIJ[kMaxDof * kMaxDof];
        physBond(p, s, op, uI, uJ, R, KII, KIJ);

        for (int a = 0; a < ndof; ++a) {
            rhs_loc[a] += R[a];
            for (int b = 0; b < ndof; ++b) kii[a * ndof + b] += -KII[a * ndof + b];
            const int row = rowBase + a;
            for (int b = 0; b < ndof; ++b) {
                const int sl = devFindSlot(csr_col, csr_off[row], csr_off[row + 1], colBase + b);
                if (sl >= 0) Kvals[sl] = -KIJ[a * ndof + b];
            }
        }
    }

    // diagonal block + rhs
    for (int a = 0; a < ndof; ++a) {
        const int row = rowBase + a;
        for (int b = 0; b < ndof; ++b) {
            const int sl = devFindSlot(csr_col, csr_off[row], csr_off[row + 1], rowBase + b);
            if (sl >= 0) Kvals[sl] = kii[a * ndof + b];
        }
        rhs[row] = rhs_loc[a];
    }
}

// ===== explicit strong-form PDDO fracture (matrix-free) device kernel =====
// One thread/node: build the PDDO moment matrix (devBuildNodeInverse), evolve the
// irreversible per-bond tensile critical-stretch damage, and assemble the
// health-gated strong-form Navier-Cauchy nodal force -> acceleration. Mirrors
// ExplicitPDDOFracElement::preprocessIteration.
__device__ inline double devPDDOCritStretch(const perix_cuda::ExplicitPDDOFracParams &p,double delta) {
    if (p.G0<=0.0 || p.E<=0.0 || delta<=0.0) return 0.0;
    if (p.dim>=3)       return sqrt(5.0*p.G0/(6.0*p.E*delta));
    if (p.plane_stress) return sqrt(4.0*M_PI*p.G0/(9.0*p.E*delta));
    return sqrt(5.0*M_PI*p.G0/(12.0*p.E*delta));   // plane strain
}

// PASS A (ONCE): build + cache the per-bond Navier PDDO operators (the reference
// geometry is constant, so the expensive moment-matrix inverse runs only once).
// opcache layout per bond: 2D {Gxx,Gyy,Gxy}; 3D {Gxx,Gyy,Gzz,Gxy,Gxz,Gyz}.
template <int DIM>
__global__ void explicitPDDOFracOpBuildKernel(int N, perix_cuda::OpSlots sl,
        const double *coords, const double *vols,
        const double *nodeHorizon, const double *nodeSpacing,
        const int *neigh_off, const int *neigh_id,
        const int *qp, const int *qq, const int *qr, const double *factprod, int nop,
        double delta_g, double dx_g, int nopb, double *opcache) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    const int nb=neigh_off[i],ne=neigh_off[i+1];
    double A[kMaxNop*kMaxNop],Inv[kMaxNop*kMaxNop],idpow[kMaxNop],op[kMaxNop];
    double delta_i,dxc_i;
    if (!devBuildNodeInverse(i,DIM,nop,qp,qq,qr,coords,vols,neigh_off,neigh_id,
                             nodeHorizon,nodeSpacing,delta_g,dx_g,A,Inv,idpow,delta_i,dxc_i)) {
        for (int t=nb;t<ne;t++) for (int k=0;k<nopb;k++) opcache[static_cast<size_t>(t)*nopb+k]=0.0;
        return;
    }
    const double xi0=coords[i*3+0],yi0=coords[i*3+1],zi0=(DIM>=3?coords[i*3+2]:0.0);
    const bool varH=(nodeHorizon!=nullptr);
    for (int t=nb;t<ne;t++) {
        const int j=neigh_id[t]-1;
        const double dx=coords[j*3+0]-xi0,dy=coords[j*3+1]-yi0,dz=(DIM>=3?coords[j*3+2]-zi0:0.0);
        const double r2=dx*dx+dy*dy+(DIM>=3?dz*dz:0.0);
        double *oc=&opcache[static_cast<size_t>(t)*nopb];
        if (r2<=0.0 || (varH && r2>delta_i*delta_i)) { for (int k=0;k<nopb;k++) oc[k]=0.0; continue; }
        devBondOps(dx,dy,dz,DIM,nop,qp,qq,qr,factprod,Inv,idpow,delta_i,dxc_i,vols[j],op);
        if (DIM>=3) { oc[0]=op[sl.dxx];oc[1]=op[sl.dyy];oc[2]=op[sl.dzz];oc[3]=op[sl.dxy];oc[4]=op[sl.dxz];oc[5]=op[sl.dyz]; }
        else        { oc[0]=op[sl.dxx];oc[1]=op[sl.dyy];oc[2]=op[sl.dxy]; }
    }
}

// PASS B (PER STEP): irreversible bond failure + health-gated Navier force ->
// acceleration, reading the cached operators (no moment-matrix work).
template <int DIM>
__global__ void explicitPDDOFracForceKernel(int N, perix_cuda::ExplicitPDDOFracParams p,
        const double *coords, const double *nodeHorizon,
        const int *neigh_off, const int *neigh_id, const double *u,
        int nopb, const double *opcache,
        double *health, double *accel, int *anyBroke) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    const int rowI=i*DIM;
    const double xi0=coords[i*3+0],yi0=coords[i*3+1],zi0=(DIM>=3?coords[i*3+2]:0.0);
    const bool varH=(nodeHorizon!=nullptr);
    const double di=varH?nodeHorizon[i]:0.0;
    const bool damage=(p.damage_on!=0);
    const int nb=neigh_off[i],ne=neigh_off[i+1];
    double L[3]={0.0,0.0,0.0};
    for (int t=nb;t<ne;t++) {
        const int j=neigh_id[t]-1; const int rowJ=j*DIM;
        const double dx=coords[j*3+0]-xi0, dy=coords[j*3+1]-yi0, dz=(DIM>=3?coords[j*3+2]-zi0:0.0);
        const double r2=dx*dx+dy*dy+(DIM>=3?dz*dz:0.0);
        if (r2<=0.0) continue;
        const double eta0=u[rowJ+0]-u[rowI+0], eta1=u[rowJ+1]-u[rowI+1], eta2=(DIM>=3?u[rowJ+2]-u[rowI+2]:0.0);
        double &hp=health[t];
        // Failure runs over EVERY directed bond -- including varH dual-family
        // bonds beyond node i's own horizon -- exactly like the CPU damage loop
        // (which has no r>di gate), so device health stays symmetric with the
        // host's. Only the force below skips far bonds (their cached ops are
        // zero anyway); gating damage on the same skip left far bonds
        // unbreakable on the GPU and forked the damage projection under a
        // variable horizon.
        if (damage && hp>0.0) {                               // irreversible failure
            const double r=sqrt(r2);
            const double yx=dx+eta0,yy=dy+eta1,yz=dz+eta2;
            const double s=(sqrt(yx*yx+yy*yy+(DIM>=3?yz*yz:0.0))-r)/r;
            const double scl=varH?devPDDOCritStretch(p,0.5*(di+nodeHorizon[j])):p.sc;
            if (scl>0.0 && (p.tension_only?(s>scl):(fabs(s)>scl))) { hp=0.0; anyBroke[0]=1; }
        }
        if (varH && r2>di*di) continue;                       // node i's own family (force only)
        if (hp<=0.0) continue;
        const double *oc=&opcache[static_cast<size_t>(t)*nopb];
        if (DIM>=3) {
            const double Gxx=oc[0],Gyy=oc[1],Gzz=oc[2],Gxy=oc[3],Gxz=oc[4],Gyz=oc[5];
            const double p11=p.C11*Gxx+p.C66*(Gyy+Gzz),p22=p.C11*Gyy+p.C66*(Gxx+Gzz),p33=p.C11*Gzz+p.C66*(Gxx+Gyy);
            const double a=p.C12+p.C66; const double p12=a*Gxy,p13=a*Gxz,p23=a*Gyz;
            L[0]+=hp*(p11*eta0+p12*eta1+p13*eta2);
            L[1]+=hp*(p12*eta0+p22*eta1+p23*eta2);
            L[2]+=hp*(p13*eta0+p23*eta1+p33*eta2);
        } else {
            const double Gxx=oc[0],Gyy=oc[1],Gxy=oc[2];
            const double p11=p.C11*Gxx+p.C66*Gyy,p22=p.C66*Gxx+p.C11*Gyy,pxy=(p.C12+p.C66)*Gxy;
            L[0]+=hp*(p11*eta0+pxy*eta1);
            L[1]+=hp*(pxy*eta0+p22*eta1);
        }
    }
    const double bvec[3]={p.bx,p.by,p.bz};
    for (int a=0;a<DIM;a++) accel[rowI+a]=(L[a]+bvec[a])/p.rho;
}

// ----- device-resident context (constant structure uploaded once) -----
struct Ctx {
    int N = 0, nnz = 0, nop = 0, nbtot = 0, ndof = 0, rows = 0;
    int *csr_off=nullptr,*csr_col=nullptr,*neigh_off=nullptr,*neigh_id=nullptr;
    int *qp=nullptr,*qq=nullptr,*qr=nullptr;
    double *coords=nullptr,*vols=nullptr,*factprod=nullptr,*u=nullptr,*uold=nullptr,*Kvals=nullptr,*rhs=nullptr;
    double *nodeHorizon=nullptr,*nodeSpacing=nullptr;   ///< per-node horizon/spacing (variable horizon), or null
    double *bondOps=nullptr;          ///< geometry-operator cache [bond*nop+k] (null => recompute)
    unsigned char *opsValid=nullptr;  ///< per-node moment-system validity for the cache
    bool opsBuilt=false;              ///< one build attempt per upload generation
    bool uploaded=false, hasHorizon=false;
};
Ctx g_ctx;    ///< implicit residual/Jacobian assembler context
Ctx g_ectx;   ///< matrix-free explicit rate assembler context (g.rhs reused as the rate buffer)

// device-resident context for the explicit strong-form PDDO fracture port.
// health is PERSISTENT (uploaded once, evolved on device); the PDDO operators
// are recomputed each step from the cached reference geometry.
struct EpfCtx {
    int N=0, nbtot=0, nop=0, dim=2, rows=0;
    double *coords=nullptr,*vols=nullptr,*factprod=nullptr,*nodeHorizon=nullptr,*nodeSpacing=nullptr;
    int    *neigh_off=nullptr,*neigh_id=nullptr,*qp=nullptr,*qq=nullptr,*qr=nullptr;
    double *u=nullptr,*health=nullptr,*accel=nullptr,*opcache=nullptr;
    int    *anyBroke=nullptr, nopb=3;
    bool uploaded=false, hasHorizon=false;
};
EpfCtx g_epfctx;


bool cudaOk(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[PeriX CUDA assemble] %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}
template <class T> bool devAlloc(T **p, size_t n) {
    return cudaOk(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(T)), "cudaMalloc");
}
template <class T> bool devCopyH2D(T *d, const T *h, size_t n) {
    return cudaOk(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice), "H2D");
}

// conservative Cahn-Hilliard (2-DoF) bespoke assembler context. Geometry
// (neighbour CSR, lambda-corrected Laplacian, reverse-bond map, node volumes,
// ghost flags) + matrix CSR uploaded once; mnode (Picard-frozen mobility) and
// u/uold uploaded per Newton step.
struct ChCtx {
    int N=0,nbtot=0,nnz=0,rows=0;
    int    *off=nullptr,*nbr=nullptr,*rev=nullptr,*csr_off=nullptr,*csr_col=nullptr;
    double *lapbond=nullptr,*vol=nullptr,*mnode=nullptr;
    char   *isghost=nullptr;
    double *u=nullptr,*uold=nullptr,*Kvals=nullptr,*rhs=nullptr;
    bool uploaded=false;
};
ChCtx g_chctx;

// ===== conservative Cahn-Hilliard (2-DoF) row-centred assembler =====
// One thread per node owns its (c,mu) rows (disjoint CSR rows -> no atomics).
// Device twin of the flux-conservative CahnHilliardElement: the nodal block is
// computeNodalResidualAndJacobian, the bond loop is computeBondResidualAndJacobian
// with the antisymmetric species flux w_ij = 0.5(M_i g2_ij + (V_j/V_i) M_j g2_ji).
// The prebuilt lambda-corrected Laplacian g2 (lapbond), reverse-bond map (rev),
// node volumes (vol), ghost flags (isghost) and Picard-frozen mobility M(c)
// (mnode) are the SAME host arrays the CPU element caches, so parity is exact by
// construction. Sign convention matches the generic assembleKernel: K=-dF/dU,
// rhs=+F.
__global__ void cahnHilliardKernel(int N, perix_cuda::CahnHilliardParams p,
                                   const int *off, const int *nbr, const int *rev,
                                   const double *lapbond, const double *vol,
                                   const char *isghost, const double *mnode,
                                   const int *csr_off, const int *csr_col,
                                   const double *u, const double *uold,
                                   double *Kvals, double *rhs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int ndof = 2;
    const int rowBase = i * ndof;

    // zero the 2 rows owned by node i. Off-diagonal slots no bond touches (e.g.
    // symmetrised-sparsity entries) correctly stay 0, matching the CPU element
    // which only writes the bonds in node i's own family.
    for (int a = 0; a < ndof; ++a)
        for (int sl = csr_off[rowBase + a]; sl < csr_off[rowBase + a + 1]; ++sl) Kvals[sl] = 0.0;

    // ---- nodal: R_c=(c-c_old)/dt, R_mu=mu-f'(c); kii accumulates -dR/dU ----
    const double c    = u[rowBase + 0];
    const double mu   = u[rowBase + 1];
    const double cold = uold ? uold[rowBase + 0] : 0.0;
    const double fp   = devFPrime(c, p.chi);
    const double fpp  = devFDoublePrime(c, p.chi);
    double rc  = (c - cold) / p.dt;
    double rmu = mu - fp;
    double kii[4];
    kii[0] = -(1.0 / p.dt);   // (c ,c )
    kii[1] = 0.0;             // (c ,mu)  bonds add coeff_c
    kii[2] = fpp;             // (mu,c )  = -(-f''(c)); bonds add coeff_mu
    kii[3] = -1.0;            // (mu,mu)

    // ---- per-bond conservative flux ----
    const double Vi = vol[i];
    const double Mi = mnode[i];
    const int bo = off[i], eo = off[i + 1];
    for (int t = bo; t < eo; ++t) {
        const int j = nbr[t] - 1;
        const bool ghostJ = (isghost[j] != 0);
        const double g2ij = lapbond[t];
        double coeff_c = 0.0;                       // = -w_ij (0 on dropped ghost bonds)
        if (!ghostJ) {
            const int r = rev[t];
            double w_ij;
            if (r >= 0 && Vi > 0.0)
                w_ij = 0.5 * (Mi * g2ij + (vol[j] / Vi) * mnode[j] * lapbond[r]);
            else
                w_ij = Mi * g2ij;                   // unpaired bulk bond: forward only
            coeff_c = -w_ij;
        }
        const double coeff_mu = ghostJ ? 0.0 : (p.kappa * g2ij);

        const int colBase = j * ndof;
        rc  += coeff_c  * (u[colBase + 1] - mu);    // R_c  += coeff_c *(mu_j-mu_i)
        rmu += coeff_mu * (u[colBase + 0] - c);     // R_mu += coeff_mu*(c_j -c_i)

        // diagonal-block accumulation of -dR/dU (self derivative at node i)
        kii[1] += coeff_c;                          // -(dR_c /dmu_i)
        kii[2] += coeff_mu;                         // -(dR_mu/dc_i )
        // off-diagonal block (column j): Kvals = -dR/dU_j
        int sl;
        sl = devFindSlot(csr_col, csr_off[rowBase + 0], csr_off[rowBase + 1], colBase + 1);
        if (sl >= 0) Kvals[sl] = -coeff_c;          // (c , mu_j)
        sl = devFindSlot(csr_col, csr_off[rowBase + 1], csr_off[rowBase + 2], colBase + 0);
        if (sl >= 0) Kvals[sl] = -coeff_mu;         // (mu, c_j )
    }

    // ---- diagonal block + rhs ----
    for (int a = 0; a < ndof; ++a) {
        const int row = rowBase + a;
        for (int b = 0; b < ndof; ++b) {
            const int sl = devFindSlot(csr_col, csr_off[row], csr_off[row + 1], rowBase + b);
            if (sl >= 0) Kvals[sl] = kii[a * ndof + b];
        }
    }
    rhs[rowBase + 0] = rc;
    rhs[rowBase + 1] = rmu;
}

// small-strain stress-coupled Cahn-Hilliard bespoke assembler context. Geometry
// (matrix CSR, neighbour CSR, PDDO multi-index tables, coords/vols, optional
// variable-horizon) + the 4 precomputed conservative-flux arrays uploaded once;
// mnode (Picard-frozen mobility) + u/uold uploaded per Newton step.
struct StressChCtx {
    int N=0,nbtot=0,nnz=0,nop=0,rows=0,dim=2;
    int    *csr_off=nullptr,*csr_col=nullptr,*neigh_off=nullptr,*neigh_id=nullptr;
    int    *qp=nullptr,*qq=nullptr,*qr=nullptr,*rev=nullptr;
    double *coords=nullptr,*vols=nullptr,*factprod=nullptr;
    double *nodeHorizon=nullptr,*nodeSpacing=nullptr;
    double *lapbond=nullptr,*mnode=nullptr;
    char   *isghost=nullptr;
    double *health=nullptr,*vel=nullptr;   // frac: per-bond health (nbtot) + committed velocity (N*dim)
    double *u=nullptr,*uold=nullptr,*Kvals=nullptr,*rhs=nullptr;
    double *bondOps=nullptr;          ///< geometry-operator cache [bond*nop+k] (null => recompute)
    unsigned char *opsValid=nullptr;  ///< per-node moment-system validity for the cache
    bool opsBuilt=false;              ///< one build attempt per upload generation
    bool uploaded=false;
};
StressChCtx g_stresschctx;

// ===== small-strain stress-coupled Cahn-Hilliard row-centred assembler =====
// Device twin of StressCahnHilliardElement. One thread per node owns its
// (c,mu,ux,uy[,uz]) rows (disjoint CSR rows -> no atomics). It builds node i's
// PDDO moment matrix on-device (shared devBuildNodeInverse/devBondOps) for the
// mechanics + chemo-mechanical operators (Gx,Gy,Gxx,Gyy,Gxy[,Gz,Gzz,Gxz,Gyz]),
// and uses the HOST-prebuilt conservative-flux arrays (lambda-corrected Laplacian
// lapbond, reverse-bond rev, ghost flags isghost, Picard-frozen mobility mnode)
// for the c-equation -- so it matches the CPU element exactly. The physics is a
// direct transcription of computeNodalResidualAndJacobian (regular-solution free energy +
// swelling) and computeBondResidualAndJacobian; the assembly (row zeroing, -dR/dU
// accumulation, devFindSlot scatter) is identical to the generic assembleKernel,
// so K=-dF/dU, rhs=+F.
__global__ void stressCahnHilliardKernel(int N, perix_cuda::StressCahnHilliardParams p,
        perix_cuda::OpSlots s,
        const int *csr_off, const int *csr_col,
        const int *neigh_off, const int *neigh_id,
        const double *coords, const double *vols,
        const int *qp, const int *qq, const int *qr, const double *factprod,
        int nop, double delta, double dx_char,
        const double *nodeHorizon, const double *nodeSpacing,
        const int *rev, const double *lapbond,
        const char *isghost, const double *mnode,
        const double *health, const double *vel,
        const double *opsCache, const unsigned char *opsValid,
        const double *u, const double *uold,
        double *Kvals, double *rhs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    constexpr int MD = 5;                 // max DoFs (3D stress-CH = dim+2 = 5)
    const int ndof = p.ndof;
    const int dim  = p.dim;
    const int rowBase = i * ndof;

    double A[kMaxNop*kMaxNop], Inv[kMaxNop*kMaxNop], op[kMaxNop], idpow[kMaxNop];
    double delta_i, dxc_i;
    if (opsCache ? (opsValid[i]==0)
                 : !devBuildNodeInverse(i, dim, nop, qp, qq, qr, coords, vols, neigh_off, neigh_id,
                                        nodeHorizon, nodeSpacing, delta, dx_char, A, Inv, idpow, delta_i, dxc_i)) {
        for (int a=0;a<ndof;++a) {
            for (int sl=csr_off[rowBase+a]; sl<csr_off[rowBase+a+1]; ++sl) Kvals[sl]=0.0;
            rhs[rowBase+a]=0.0;
        }
        return;
    }
    if (opsCache) {
        delta_i = nodeHorizon ? nodeHorizon[i] : delta;   // far-bond zero-op branch only
        dxc_i   = 0.0;
    }

    const double xi0=coords[i*3+0], yi0=coords[i*3+1], zi0=coords[i*3+2];
    const int nb=neigh_off[i], ne=neigh_off[i+1];

    double uI[MD];
    for (int a=0;a<ndof;++a) uI[a]=u[rowBase+a];
    const double c=uI[0], mu=uI[1];
    const double cold = uold ? uold[rowBase+0] : 0.0;

    // ---- nodal contribution (regular-solution free energy) ----
    const double fp=devFPrime(c,p.chi);
    const double fpp=devFDoublePrime(c,p.chi);
    double rhs_loc[MD]; double kii[MD*MD];
    for (int a=0;a<ndof*ndof;++a) kii[a]=0.0;
    for (int a=0;a<ndof;++a) rhs_loc[a]=0.0;
    rhs_loc[0] = (c-cold)/p.dt;
    rhs_loc[1] = mu - fp + p.Omega*p.Ch*(c - p.cref);
    kii[0*ndof+0] = -(1.0/p.dt);                 // -(dR_c /dc )
    kii[1*ndof+0] = -(-fpp + p.Omega*p.Ch);      // -(dR_mu/dc ) nodal
    kii[1*ndof+1] = -(1.0);                       // -(dR_mu/dmu)

    // frac_stress_cahnhilliard optional inertia on the displacement rows:
    // R_u += -rho (u - u_old - vel*dt)/dt^2 ; dR_u/du_u = -rho/dt^2 -> kii += +rho/dt^2.
    if (p.rho>0.0 && vel) {
        const double c0 = p.rho/(p.dt*p.dt);
        for (int d=0; d<dim; ++d) {
            const int ur = 2+d;                  // rows: c=0, mu=1, ux=2, ...
            const double uold_ur = uold ? uold[rowBase+ur] : 0.0;
            rhs_loc[ur] += -c0*(uI[ur] - uold_ur - vel[i*dim+d]*p.dt);
            kii[ur*ndof+ur] += c0;
        }
    }

    // zero my ndof rows
    for (int a=0;a<ndof;++a)
        for (int sl=csr_off[rowBase+a]; sl<csr_off[rowBase+a+1]; ++sl) Kvals[sl]=0.0;

    const double Vi=vols[i], Mi=mnode[i];

    for (int t=nb; t<ne; ++t) {
        const int j=neigh_id[t]-1;
        const double dx=coords[j*3+0]-xi0, dy=coords[j*3+1]-yi0;
        const double dz=(dim>=3)?coords[j*3+2]-zi0:0.0;
        const double xn=sqrt(dx*dx+dy*dy+(dim>=3?dz*dz:0.0));
        if (xn<=0.0) continue;
        // Variable horizon: a bond outside node i's OWN horizon contributes ZERO
        // PDDO operators (the CPU calcOperators returns setToZeros there), so the
        // mechanics / chemo-mechanical / kappa-Laplacian terms vanish -- BUT the
        // bond is NOT skipped: its conservative species flux still fires through
        // the reverse weight g2_ji (this bond is near for node j). Skipping it (as
        // a pure-operator kernel would) drops that flux and breaks varH parity.
        if (opsCache) {
            // cache stores ZEROS for xn>delta_i bonds -- identical to the branch below
            const double *oc=&opsCache[static_cast<size_t>(t)*nop];
            for (int k=0;k<nop;++k) op[k]=oc[k];
        }
        else if (nodeHorizon && xn>delta_i) { for (int k=0;k<nop;++k) op[k]=0.0; }
        else devBondOps(dx,dy,dz, dim, nop, qp,qq,qr, factprod, Inv, idpow, delta_i, dxc_i, vols[j], op);

        // Conservative species-flux weight for the c-equation (dropped on
        // ghost bonds). The same surface-corrected Laplacian drives the
        // gradient-energy term.
        const bool ghostJ = (isghost[j]!=0);
        const double g2A = lapbond[t];
        double coeff_c = 0.0;
        if (!ghostJ) {
            const int r = rev[t];
            const double w_ij = (r>=0 && Vi>0.0)
                ? 0.5*(Mi*g2A + (vols[j]/Vi)*mnode[j]*lapbond[r])
                : Mi*g2A;
            coeff_c = -w_ij;
        }
        const double kappaLap = ghostJ ? 0.0 : (p.kappa*lapbond[t]);

        double uJ[MD];
        const int colBase=j*ndof;
        for (int b=0;b<ndof;++b) uJ[b]=u[colBase+b];

        double R[MD], KII[MD*MD], KIJ[MD*MD];
        for (int a=0;a<ndof*ndof;++a){ KII[a]=0.0; KIJ[a]=0.0; }

        // frac: gate the MECHANICAL operators by the bond health g. g==1.0 exactly
        // for the non-fracture path (health==nullptr) -> bit-identical. Scaling the
        // operators scales every mechanical term (cmu_u*, p*, c*) by g, which is
        // exactly frac_stress_cahnhilliard's per-term g factor; the species flux
        // (coeff_c) and kappa-Laplacian (kappaLap) use lapbond (NOT these
        // operators), so they stay UNgated, as required.
        const double g = health ? (health[t]>0.0 ? 1.0 : p.kres) : 1.0;
        const double Gx=g*op[s.dx], Gy=g*op[s.dy];
        const double Gxx=g*op[s.dxx], Gyy=g*op[s.dyy], Gxy=g*op[s.dxy];
        const double cmu_ux=p.Omega*p.Kh*Gx, cmu_uy=p.Omega*p.Kh*Gy;
        const double cx=-p.A*Gx, cy=-p.A*Gy;
        const double dC=uJ[0]-uI[0], dMu=uJ[1]-uI[1];
        if (dim<3) {
            const double p11=p.C11*Gxx+p.C66*Gyy, p22=p.C66*Gxx+p.C11*Gyy, pxy=(p.C12+p.C66)*Gxy;
            const double dux=uJ[2]-uI[2], duy=uJ[3]-uI[3];
            R[0]=coeff_c*dMu;
            R[1]=kappaLap*dC + cmu_ux*dux + cmu_uy*duy;
            R[2]=cx*dC + p11*dux + pxy*duy;
            R[3]=cy*dC + pxy*dux + p22*duy;
            KII[0*4+1]=-coeff_c;
            KII[1*4+0]=-kappaLap; KII[1*4+2]=-cmu_ux; KII[1*4+3]=-cmu_uy;
            KII[2*4+0]=-cx;       KII[2*4+2]=-p11;    KII[2*4+3]=-pxy;
            KII[3*4+0]=-cy;       KII[3*4+2]=-pxy;    KII[3*4+3]=-p22;
            KIJ[0*4+1]= coeff_c;
            KIJ[1*4+0]= kappaLap; KIJ[1*4+2]= cmu_ux; KIJ[1*4+3]= cmu_uy;
            KIJ[2*4+0]= cx;       KIJ[2*4+2]= p11;    KIJ[2*4+3]= pxy;
            KIJ[3*4+0]= cy;       KIJ[3*4+2]= pxy;    KIJ[3*4+3]= p22;
        } else {
            const double Gz=g*op[s.dz], Gzz=g*op[s.dzz], Gxz=g*op[s.dxz], Gyz=g*op[s.dyz];
            const double a=p.C12+p.C66;
            const double cmu_uz=p.Omega*p.Kh*Gz, cz=-p.A*Gz;
            const double p11=p.C11*Gxx+p.C66*(Gyy+Gzz);
            const double p22=p.C11*Gyy+p.C66*(Gxx+Gzz);
            const double p33=p.C11*Gzz+p.C66*(Gxx+Gyy);
            const double p12=a*Gxy, p13=a*Gxz, p23=a*Gyz;
            const double dux=uJ[2]-uI[2], duy=uJ[3]-uI[3], duz=uJ[4]-uI[4];
            R[0]=coeff_c*dMu;
            R[1]=kappaLap*dC + cmu_ux*dux + cmu_uy*duy + cmu_uz*duz;
            R[2]=cx*dC + p11*dux + p12*duy + p13*duz;
            R[3]=cy*dC + p12*dux + p22*duy + p23*duz;
            R[4]=cz*dC + p13*dux + p23*duy + p33*duz;
            KII[0*5+1]=-coeff_c;
            KII[1*5+0]=-kappaLap; KII[1*5+2]=-cmu_ux; KII[1*5+3]=-cmu_uy; KII[1*5+4]=-cmu_uz;
            KII[2*5+0]=-cx; KII[2*5+2]=-p11; KII[2*5+3]=-p12; KII[2*5+4]=-p13;
            KII[3*5+0]=-cy; KII[3*5+2]=-p12; KII[3*5+3]=-p22; KII[3*5+4]=-p23;
            KII[4*5+0]=-cz; KII[4*5+2]=-p13; KII[4*5+3]=-p23; KII[4*5+4]=-p33;
            KIJ[0*5+1]= coeff_c;
            KIJ[1*5+0]= kappaLap; KIJ[1*5+2]= cmu_ux; KIJ[1*5+3]= cmu_uy; KIJ[1*5+4]= cmu_uz;
            KIJ[2*5+0]= cx; KIJ[2*5+2]= p11; KIJ[2*5+3]= p12; KIJ[2*5+4]= p13;
            KIJ[3*5+0]= cy; KIJ[3*5+2]= p12; KIJ[3*5+3]= p22; KIJ[3*5+4]= p23;
            KIJ[4*5+0]= cz; KIJ[4*5+2]= p13; KIJ[4*5+3]= p23; KIJ[4*5+4]= p33;
        }

        for (int a=0;a<ndof;++a) {
            rhs_loc[a]+=R[a];
            for (int b=0;b<ndof;++b) kii[a*ndof+b] += -KII[a*ndof+b];
            const int row=rowBase+a;
            for (int b=0;b<ndof;++b) {
                const int sl=devFindSlot(csr_col, csr_off[row], csr_off[row+1], colBase+b);
                if (sl>=0) Kvals[sl]=-KIJ[a*ndof+b];
            }
        }
    }

    for (int a=0;a<ndof;++a) {
        const int row=rowBase+a;
        for (int b=0;b<ndof;++b) {
            const int sl=devFindSlot(csr_col, csr_off[row], csr_off[row+1], rowBase+b);
            if (sl>=0) Kvals[sl]=kii[a*ndof+b];
        }
        rhs[row]=rhs_loc[a];
    }
}

// ===== implicit PDDO dynamic-fracture assembler =====
// The element carries committed nodal velocity in nodecache and frozen bond
// health in bondcache. The device kernel uses the same PDDO operators and
// row-centred residual/Jacobian convention as the CPU implementation.
__device__ inline void physNodalCached(const perix_cuda::CachedParams &p,
        const double *uI, const double *uoldI, const double *nc,
        double *Rn, double *Kn) {
    const int ndof=p.ndof;
    for (int k=0;k<ndof*ndof;++k) Kn[k]=0.0;
    const double c0=p.rho/(p.dt*p.dt);
    const double b[3]={p.bx,p.by,p.bz};
    for (int a=0;a<ndof;++a) {
        Rn[a]=b[a]-c0*(uI[a]-uoldI[a]-nc[a]*p.dt);
        Kn[a*ndof+a]=-c0;
    }
}

struct BondGeomView { double dx,dy,dz,xn,di,dj,vi,vj; int varh; };

__device__ inline void physBondCached(const perix_cuda::CachedParams &p,
        const perix_cuda::OpSlots &s, const double *op,
        const double *uI, const double *uJ, const double *nc, const double *ncj,
        const double *bc, const BondGeomView &geom,
        double *R, double *KII, double *KIJ) {
    const int ndof=p.ndof;
    for (int k=0;k<ndof*ndof;++k){ KII[k]=0.0; KIJ[k]=0.0; }
    for (int a=0;a<ndof;++a) R[a]=0.0;
    (void)nc;
    (void)ncj;
    (void)geom;

    const double health=bc[0];
    if (health<=0.0) return;
    const bool d3=(p.dim>=3);
    const double Gxx=op[s.dxx], Gyy=op[s.dyy], Gxy=op[s.dxy];
    if (!d3) {
        const double p11=health*(p.C11*Gxx+p.C66*Gyy);
        const double p22=health*(p.C66*Gxx+p.C11*Gyy);
        const double pxy=health*(p.C12+p.C66)*Gxy;
        const double dux=uJ[0]-uI[0], duy=uJ[1]-uI[1];
        R[0]=p11*dux+pxy*duy;
        R[1]=pxy*dux+p22*duy;
        KII[0]=-p11; KII[1]=-pxy; KII[2]=-pxy; KII[3]=-p22;
        KIJ[0]= p11; KIJ[1]= pxy; KIJ[2]= pxy; KIJ[3]= p22;
    } else {
        const double Gzz=op[s.dzz], Gxz=op[s.dxz], Gyz=op[s.dyz];
        const double a=p.C12+p.C66;
        const double p11=health*(p.C11*Gxx+p.C66*(Gyy+Gzz));
        const double p22=health*(p.C11*Gyy+p.C66*(Gxx+Gzz));
        const double p33=health*(p.C11*Gzz+p.C66*(Gxx+Gyy));
        const double p12=health*a*Gxy, p13=health*a*Gxz, p23=health*a*Gyz;
        const double dux=uJ[0]-uI[0], duy=uJ[1]-uI[1], duz=uJ[2]-uI[2];
        R[0]=p11*dux+p12*duy+p13*duz;
        R[1]=p12*dux+p22*duy+p23*duz;
        R[2]=p13*dux+p23*duy+p33*duz;
        KII[0]=-p11; KII[1]=-p12; KII[2]=-p13;
        KII[3]=-p12; KII[4]=-p22; KII[5]=-p23;
        KII[6]=-p13; KII[7]=-p23; KII[8]=-p33;
        KIJ[0]= p11; KIJ[1]= p12; KIJ[2]= p13;
        KIJ[3]= p12; KIJ[4]= p22; KIJ[5]= p23;
        KIJ[6]= p13; KIJ[7]= p23; KIJ[8]= p33;
    }
}

struct CachedCtx {
    int N=0,nbtot=0,nnz=0,nop=0,rows=0,dim=2,ncache=0,nbcache=0;
    int    *csr_off=nullptr,*csr_col=nullptr,*neigh_off=nullptr,*neigh_id=nullptr;
    int    *qp=nullptr,*qq=nullptr,*qr=nullptr;
    double *coords=nullptr,*vols=nullptr,*factprod=nullptr,*nodeHorizon=nullptr,*nodeSpacing=nullptr;
    double *nodecache=nullptr,*bondcache=nullptr,*u=nullptr,*uold=nullptr,*Kvals=nullptr,*rhs=nullptr;
    double *bondOps=nullptr;          ///< geometry-operator cache [bond*nop+k] (null => recompute)
    unsigned char *opsValid=nullptr;  ///< per-node moment-system validity for the cache
    bool opsBuilt=false;              ///< one build attempt per upload generation
    bool uploaded=false;
};
CachedCtx g_cachedctx;

__global__ void cachedAssembleKernel(int N, perix_cuda::CachedParams p, perix_cuda::OpSlots s,
        const int *csr_off, const int *csr_col, const int *neigh_off, const int *neigh_id,
        const double *coords, const double *vols,
        const int *qp, const int *qq, const int *qr, const double *factprod,
        int nop, double delta, double dx_char, const double *nodeHorizon, const double *nodeSpacing,
        const double *opsCache, const unsigned char *opsValid,
        const double *nodecache, const double *bondcache, const double *u, const double *uold,
        double *Kvals, double *rhs) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    const int ndof=p.ndof, dim=p.dim, ncache=p.ncache, rowBase=i*ndof;
    double A[kMaxNop*kMaxNop], Inv[kMaxNop*kMaxNop], op[kMaxNop], idpow[kMaxNop];
    double delta_i, dxc_i;
    if (opsCache ? (opsValid[i]==0)
                 : !devBuildNodeInverse(i, dim, nop, qp,qq,qr, coords, vols, neigh_off, neigh_id,
                                        nodeHorizon, nodeSpacing, delta, dx_char, A, Inv, idpow, delta_i, dxc_i)) {
        for (int a=0;a<ndof;++a){ for (int sl=csr_off[rowBase+a]; sl<csr_off[rowBase+a+1]; ++sl) Kvals[sl]=0.0; rhs[rowBase+a]=0.0; }
        return;
    }
    if (opsCache) {
        delta_i = nodeHorizon ? nodeHorizon[i] : delta;
        dxc_i   = 0.0;
    }
    const double xi0=coords[i*3+0], yi0=coords[i*3+1], zi0=coords[i*3+2];
    const int nb=neigh_off[i], ne=neigh_off[i+1];
    double uI[kMaxDof], uoldI[kMaxDof];
    for (int a=0;a<ndof;++a){ uI[a]=u[rowBase+a]; uoldI[a]=uold?uold[rowBase+a]:0.0; }
    const double *nc = nodecache + (size_t)i*ncache;

    double Rn[kMaxDof], Kn[kMaxDof*kMaxDof];
    physNodalCached(p, uI, uoldI, nc, Rn, Kn);
    double rhs_loc[kMaxDof], kii[kMaxDof*kMaxDof];
    for (int a=0;a<ndof;++a){ rhs_loc[a]=Rn[a]; for (int b=0;b<ndof;++b) kii[a*ndof+b]=-Kn[a*ndof+b]; }
    for (int a=0;a<ndof;++a) for (int sl=csr_off[rowBase+a]; sl<csr_off[rowBase+a+1]; ++sl) Kvals[sl]=0.0;

    for (int t=nb; t<ne; ++t) {
        const int j=neigh_id[t]-1;
        const double dx=coords[j*3+0]-xi0, dy=coords[j*3+1]-yi0;
        const double dz=(dim>=3)?coords[j*3+2]-zi0:0.0;
        const double xn=sqrt(dx*dx+dy*dy+(dim>=3?dz*dz:0.0));
        if (xn<=0.0) continue;
        if (opsCache) {
            const double *oc=&opsCache[static_cast<size_t>(t)*nop];
            for (int k=0;k<nop;++k) op[k]=oc[k];
        }
        else if (nodeHorizon && xn>delta_i) { for (int k=0;k<nop;++k) op[k]=0.0; }
        else devBondOps(dx,dy,dz, dim, nop, qp,qq,qr, factprod, Inv, idpow, delta_i, dxc_i, vols[j], op);
        double uJ[kMaxDof]; const int colBase=j*ndof;
        for (int b=0;b<ndof;++b) uJ[b]=u[colBase+b];
        BondGeomView geom;
        geom.dx=dx; geom.dy=dy; geom.dz=dz; geom.xn=xn;
        geom.di=delta_i; geom.dj=nodeHorizon?nodeHorizon[j]:delta;
        geom.vi=vols[i]; geom.vj=vols[j];
        geom.varh=nodeHorizon?1:0;
        const double defbc[3]={1.0,1.0,1.0};
        double R[kMaxDof], KII[kMaxDof*kMaxDof], KIJ[kMaxDof*kMaxDof];
        physBondCached(p, s, op, uI, uJ, nc, nodecache + (size_t)j*ncache,
                       bondcache ? bondcache + (size_t)t*p.nbcache : defbc,
                       geom, R, KII, KIJ);
        for (int a=0;a<ndof;++a) {
            rhs_loc[a]+=R[a];
            for (int b=0;b<ndof;++b) kii[a*ndof+b] += -KII[a*ndof+b];
            const int row=rowBase+a;
            for (int b=0;b<ndof;++b){ const int sl=devFindSlot(csr_col, csr_off[row], csr_off[row+1], colBase+b); if (sl>=0) Kvals[sl]=-KIJ[a*ndof+b]; }
        }
    }
    for (int a=0;a<ndof;++a) {
        const int row=rowBase+a;
        for (int b=0;b<ndof;++b){ const int sl=devFindSlot(csr_col, csr_off[row], csr_off[row+1], rowBase+b); if (sl>=0) Kvals[sl]=kii[a*ndof+b]; }
        rhs[row]=rhs_loc[a];
    }
}

// One build attempt per context upload generation: allocate and fill the
// geometry-operator cache with opsBuildKernel, or leave the pointers null (the
// per-launch recompute path, i.e. the original behaviour) on any failure or
// when disabled via PERIX_CUDA_OPCACHE=0. All pointers are DEVICE pointers
// already resident from the context upload.
inline void buildDeviceOpsCache(int N, int nbtot, int nop, int dim,
        const int *d_qp, const int *d_qq, const int *d_qr, const double *d_factprod,
        const double *d_coords, const double *d_vols,
        const int *d_neigh_off, const int *d_neigh_id,
        const double *d_nodeHorizon, const double *d_nodeSpacing,
        double delta, double dx_char,
        double **bondOps, unsigned char **opsValid, bool &opsBuilt,
        bool isExplicitPath = false) {
    opsBuilt = true;
    *bondOps = nullptr;
    *opsValid = nullptr;
    if (!devOpCacheEnabled(nop, isExplicitPath) || N <= 0 || nop <= 0 || nbtot <= 0) return;
    double *ops = nullptr;
    unsigned char *valid = nullptr;
    bool ok = devAlloc(&ops, static_cast<size_t>(nbtot) * static_cast<size_t>(nop))
           && devAlloc(&valid, static_cast<size_t>(N));
    if (ok) {
        const int blk = 128, grd = (N + blk - 1) / blk;
        opsBuildKernel<<<grd, blk>>>(N, dim, nop, d_qp, d_qq, d_qr, d_factprod,
                                     d_coords, d_vols, d_neigh_off, d_neigh_id,
                                     d_nodeHorizon, d_nodeSpacing, delta, dx_char,
                                     ops, valid);
        ok = cudaOk(cudaGetLastError(), "opsBuild launch")
          && cudaOk(cudaDeviceSynchronize(), "opsBuild sync");
    }
    if (!ok) { cudaFree(ops); cudaFree(valid); return; }
    *bondOps = ops;
    *opsValid = valid;
    std::fprintf(stderr, "[PeriX CUDA] geometry-operator cache resident: %d bonds x %d ops (%.1f MB)\n",
                 nbtot, nop,
                 static_cast<double>(nbtot) * nop * sizeof(double) / (1024.0 * 1024.0));
}

} // namespace

namespace perix_cuda {

void shutdown() {
    for (Ctx *gp : {&g_ctx, &g_ectx}) {
        Ctx &g = *gp;
        cudaFree(g.csr_off); cudaFree(g.csr_col); cudaFree(g.neigh_off); cudaFree(g.neigh_id);
        cudaFree(g.qp); cudaFree(g.qq); cudaFree(g.qr); cudaFree(g.coords); cudaFree(g.vols);
        cudaFree(g.factprod); cudaFree(g.u); cudaFree(g.uold); cudaFree(g.Kvals); cudaFree(g.rhs);
        cudaFree(g.nodeHorizon); cudaFree(g.nodeSpacing);
        cudaFree(g.bondOps); cudaFree(g.opsValid);
        g = Ctx{};
    }
    EpfCtx &e = g_epfctx;
    cudaFree(e.coords); cudaFree(e.vols); cudaFree(e.factprod); cudaFree(e.nodeHorizon); cudaFree(e.nodeSpacing);
    cudaFree(e.neigh_off); cudaFree(e.neigh_id); cudaFree(e.qp); cudaFree(e.qq); cudaFree(e.qr);
    cudaFree(e.u); cudaFree(e.health); cudaFree(e.accel); cudaFree(e.anyBroke); cudaFree(e.opcache);
    e = EpfCtx{};

    ChCtx &ch = g_chctx;
    cudaFree(ch.off); cudaFree(ch.nbr); cudaFree(ch.rev); cudaFree(ch.csr_off); cudaFree(ch.csr_col);
    cudaFree(ch.lapbond); cudaFree(ch.vol); cudaFree(ch.mnode); cudaFree(ch.isghost);
    cudaFree(ch.u); cudaFree(ch.uold); cudaFree(ch.Kvals); cudaFree(ch.rhs);
    ch = ChCtx{};

    StressChCtx &sc = g_stresschctx;
    cudaFree(sc.csr_off); cudaFree(sc.csr_col); cudaFree(sc.neigh_off); cudaFree(sc.neigh_id);
    cudaFree(sc.qp); cudaFree(sc.qq); cudaFree(sc.qr); cudaFree(sc.coords); cudaFree(sc.vols);
    cudaFree(sc.factprod); cudaFree(sc.nodeHorizon); cudaFree(sc.nodeSpacing);
    cudaFree(sc.rev); cudaFree(sc.lapbond); cudaFree(sc.isghost); cudaFree(sc.mnode);
    cudaFree(sc.health); cudaFree(sc.vel);
    cudaFree(sc.u); cudaFree(sc.uold); cudaFree(sc.Kvals); cudaFree(sc.rhs);
    cudaFree(sc.bondOps); cudaFree(sc.opsValid);
    sc = StressChCtx{};

    CachedCtx &cc = g_cachedctx;
    cudaFree(cc.csr_off); cudaFree(cc.csr_col); cudaFree(cc.neigh_off); cudaFree(cc.neigh_id);
    cudaFree(cc.qp); cudaFree(cc.qq); cudaFree(cc.qr); cudaFree(cc.coords); cudaFree(cc.vols);
    cudaFree(cc.factprod); cudaFree(cc.nodeHorizon); cudaFree(cc.nodeSpacing);
    cudaFree(cc.nodecache); cudaFree(cc.bondcache); cudaFree(cc.u); cudaFree(cc.uold); cudaFree(cc.Kvals); cudaFree(cc.rhs);
    cudaFree(cc.bondOps); cudaFree(cc.opsValid);
    cc = CachedCtx{};
}

bool assemble(const AssembleMesh &m, const ElmtParams &p, const OpSlots &s,
              const double *u, const double *uold,
              double *out_Kvals, double *out_rhs) {
    if (m.nop > kMaxNop || m.N <= 0 || p.ndof < 1 || p.ndof > kMaxDof) return false;
    if (p.type == ELMT_POISSON && (s.dxx < 0 || s.dyy < 0)) return false;
    if (p.type == ELMT_DIFFUSION && (s.dxx < 0 || s.dyy < 0 || s.dx < 0 || s.dy < 0)) return false;

    Ctx &g = g_ctx;
    const int nbtot = m.neigh_off[m.N];
    const int rows  = m.N * p.ndof;
    const bool wantHorizon = (m.nodeHorizon != nullptr && m.nodeSpacing != nullptr);

    if (!g.uploaded || g.N != m.N || g.nnz != m.nnz || g.nbtot != nbtot
        || g.nop != m.nop || g.ndof != p.ndof || g.hasHorizon != wantHorizon) {
        shutdown();
        bool ok = true;
        ok = ok && devAlloc(&g.csr_off, rows + 1)  && devCopyH2D(g.csr_off, m.csr_off, rows + 1);
        ok = ok && devAlloc(&g.csr_col, m.nnz)     && devCopyH2D(g.csr_col, m.csr_col, m.nnz);
        ok = ok && devAlloc(&g.neigh_off, m.N + 1) && devCopyH2D(g.neigh_off, m.neigh_off, m.N + 1);
        ok = ok && devAlloc(&g.neigh_id, nbtot)    && devCopyH2D(g.neigh_id, m.neigh_id, nbtot);
        ok = ok && devAlloc(&g.coords, 3 * m.N)    && devCopyH2D(g.coords, m.coords, 3 * m.N);
        ok = ok && devAlloc(&g.vols, m.N)          && devCopyH2D(g.vols, m.vols, m.N);
        ok = ok && devAlloc(&g.qp, m.nop)          && devCopyH2D(g.qp, m.qp, m.nop);
        ok = ok && devAlloc(&g.qq, m.nop)          && devCopyH2D(g.qq, m.qq, m.nop);
        ok = ok && devAlloc(&g.qr, m.nop)          && devCopyH2D(g.qr, m.qr, m.nop);
        ok = ok && devAlloc(&g.factprod, m.nop)    && devCopyH2D(g.factprod, m.factprod, m.nop);
        if (wantHorizon) {
            ok = ok && devAlloc(&g.nodeHorizon, m.N) && devCopyH2D(g.nodeHorizon, m.nodeHorizon, m.N);
            ok = ok && devAlloc(&g.nodeSpacing, m.N) && devCopyH2D(g.nodeSpacing, m.nodeSpacing, m.N);
        }
        ok = ok && devAlloc(&g.u, rows);
        ok = ok && devAlloc(&g.uold, rows);
        ok = ok && devAlloc(&g.Kvals, m.nnz);
        ok = ok && devAlloc(&g.rhs, rows);
        if (!ok) { shutdown(); return false; }
        g.N = m.N; g.nnz = m.nnz; g.nop = m.nop; g.nbtot = nbtot; g.ndof = p.ndof; g.rows = rows;
        g.hasHorizon = wantHorizon; g.uploaded = true;
    }

    if (!g.opsBuilt) {
        buildDeviceOpsCache(m.N, nbtot, m.nop, p.dim, g.qp, g.qq, g.qr, g.factprod,
                            g.coords, g.vols, g.neigh_off, g.neigh_id,
                            wantHorizon ? g.nodeHorizon : nullptr,
                            wantHorizon ? g.nodeSpacing : nullptr,
                            m.delta, m.dx_char, &g.bondOps, &g.opsValid, g.opsBuilt);
    }

    if (!devCopyH2D(g.u, u, rows)) return false;
    if (uold && !devCopyH2D(g.uold, uold, rows)) return false;

    static bool announced = false;
    if (!announced) {
        std::fprintf(stderr, "[PeriX CUDA] GPU assembler active: elmt-type=%d ndof=%d N=%d nnz=%d\n",
                     p.type, p.ndof, m.N, m.nnz);
        announced = true;
    }

    const int block = 128;
    const int grid  = (m.N + block - 1) / block;
    assembleKernel<<<grid, block>>>(m.N, p, s, g.csr_off, g.csr_col, g.neigh_off, g.neigh_id,
                                    g.coords, g.vols, g.qp, g.qq, g.qr, g.factprod,
                                    m.nop, m.delta, m.dx_char,
                                    wantHorizon ? g.nodeHorizon : nullptr,
                                    wantHorizon ? g.nodeSpacing : nullptr,
                                    g.bondOps, g.opsValid,
                                    g.u, uold ? g.uold : nullptr,
                                    g.Kvals, g.rhs);
    if (!cudaOk(cudaGetLastError(), "kernel launch")) return false;
    if (!cudaOk(cudaDeviceSynchronize(), "kernel sync")) return false;

    if (!cudaOk(cudaMemcpy(out_Kvals, g.Kvals, static_cast<size_t>(m.nnz) * sizeof(double),
                           cudaMemcpyDeviceToHost), "D2H(Kvals)")) return false;
    if (!cudaOk(cudaMemcpy(out_rhs, g.rhs, static_cast<size_t>(rows) * sizeof(double),
                           cudaMemcpyDeviceToHost), "D2H(rhs)")) return false;
    return true;
}




bool assembleExplicitPDDOFracAccel(const AssembleMesh &m, const ExplicitPDDOFracParams &p,
                                   const OpSlots &s, const double *u,
                                   double *health_inout, double *accel_out, int *health_changed) {
    if (health_changed) *health_changed=0;
    if (m.N<=0 || m.nop>kMaxNop || (p.dim!=2 && p.dim!=3)) return false;
    EpfCtx &e=g_epfctx;
    const int N=m.N, nbtot=m.neigh_off[m.N], dim=p.dim, rows=N*dim;
    const bool wantHorizon=(m.nodeHorizon!=nullptr && m.nodeSpacing!=nullptr);

    if (!e.uploaded || e.N!=N || e.nbtot!=nbtot || e.nop!=m.nop || e.dim!=dim || e.hasHorizon!=wantHorizon) {
        shutdown();
        bool ok=true;
        ok=ok && devAlloc(&e.coords,3*N)    && devCopyH2D(e.coords,m.coords,3*N);
        ok=ok && devAlloc(&e.vols,N)        && devCopyH2D(e.vols,m.vols,N);
        ok=ok && devAlloc(&e.neigh_off,N+1) && devCopyH2D(e.neigh_off,m.neigh_off,N+1);
        ok=ok && devAlloc(&e.neigh_id,nbtot)&& devCopyH2D(e.neigh_id,m.neigh_id,nbtot);
        ok=ok && devAlloc(&e.qp,m.nop)      && devCopyH2D(e.qp,m.qp,m.nop);
        ok=ok && devAlloc(&e.qq,m.nop)      && devCopyH2D(e.qq,m.qq,m.nop);
        ok=ok && devAlloc(&e.qr,m.nop)      && devCopyH2D(e.qr,m.qr,m.nop);
        ok=ok && devAlloc(&e.factprod,m.nop)&& devCopyH2D(e.factprod,m.factprod,m.nop);
        if (wantHorizon) {
            ok=ok && devAlloc(&e.nodeHorizon,N) && devCopyH2D(e.nodeHorizon,m.nodeHorizon,N);
            ok=ok && devAlloc(&e.nodeSpacing,N) && devCopyH2D(e.nodeSpacing,m.nodeSpacing,N);
        }
        ok=ok && devAlloc(&e.health,nbtot)  && devCopyH2D(e.health,health_inout,nbtot);
        e.nopb=(dim>=3)?6:3;
        ok=ok && devAlloc(&e.opcache,static_cast<size_t>(nbtot)*e.nopb);
        ok=ok && devAlloc(&e.u,rows) && devAlloc(&e.accel,rows) && devAlloc(&e.anyBroke,1);
        if (!ok) { shutdown(); return false; }
        e.N=N; e.nbtot=nbtot; e.nop=m.nop; e.dim=dim; e.rows=rows;
        e.hasHorizon=wantHorizon;

        // PASS A: build the cached per-bond Navier operators ONCE.
        const int blk=128, grd=(N+blk-1)/blk;
        const double *nh0=wantHorizon?e.nodeHorizon:nullptr, *ns0=wantHorizon?e.nodeSpacing:nullptr;
        if (dim==3)
            explicitPDDOFracOpBuildKernel<3><<<grd,blk>>>(N,s,e.coords,e.vols,nh0,ns0,e.neigh_off,e.neigh_id,
                e.qp,e.qq,e.qr,e.factprod,m.nop,m.delta,m.dx_char,e.nopb,e.opcache);
        else
            explicitPDDOFracOpBuildKernel<2><<<grd,blk>>>(N,s,e.coords,e.vols,nh0,ns0,e.neigh_off,e.neigh_id,
                e.qp,e.qq,e.qr,e.factprod,m.nop,m.delta,m.dx_char,e.nopb,e.opcache);
        if (!cudaOk(cudaGetLastError(),"epf op-build launch")) { shutdown(); return false; }
        if (!cudaOk(cudaDeviceSynchronize(),"epf op-build sync")) { shutdown(); return false; }
        // Commit the cache only now: flagging `uploaded` before the op-build
        // succeeded would let a failed build pass the cache guard on every
        // later call and run PASS B on uninitialized operators, silently.
        e.uploaded=true;

        static bool announced=false;
        if (!announced) {
            std::fprintf(stderr,"[PeriX CUDA] GPU explicit PDDO-fracture assembler active: "
                         "N=%d bonds=%d dim=%d %s (cached operators)\n",N,nbtot,dim,
                         wantHorizon?"variable-horizon":"uniform-horizon");
            announced=true;
        }
    }

    if (!devCopyH2D(e.u,u,rows)) return false;
    if (!cudaOk(cudaMemset(e.anyBroke,0,sizeof(int)),"memset(anyBroke)")) return false;

    // PASS B: per-step force + failure from the cached operators.
    const int block=128, grid=(N+block-1)/block;
    const double *nh=wantHorizon?e.nodeHorizon:nullptr;
    if (dim==3)
        explicitPDDOFracForceKernel<3><<<grid,block>>>(N,p,e.coords,nh,e.neigh_off,e.neigh_id,e.u,
            e.nopb,e.opcache,e.health,e.accel,e.anyBroke);
    else
        explicitPDDOFracForceKernel<2><<<grid,block>>>(N,p,e.coords,nh,e.neigh_off,e.neigh_id,e.u,
            e.nopb,e.opcache,e.health,e.accel,e.anyBroke);
    if (!cudaOk(cudaGetLastError(),"epf force launch")) return false;
    if (!cudaOk(cudaDeviceSynchronize(),"epf force sync")) return false;

    if (!cudaOk(cudaMemcpy(accel_out,e.accel,static_cast<size_t>(rows)*sizeof(double),
                           cudaMemcpyDeviceToHost),"D2H(accel)")) return false;
    int broke=0;
    if (!cudaOk(cudaMemcpy(&broke,e.anyBroke,sizeof(int),cudaMemcpyDeviceToHost),"D2H(anyBroke)")) return false;
    if (broke) {
        if (!cudaOk(cudaMemcpy(health_inout,e.health,static_cast<size_t>(nbtot)*sizeof(double),
                               cudaMemcpyDeviceToHost),"D2H(health)")) return false;
        if (health_changed) *health_changed=1;
    }
    return true;
}





bool assembleCahnHilliard(const CahnHilliardMesh &m, const CahnHilliardParams &p,
                          const int *csr_off, const int *csr_col, int nnz,
                          const double *u, const double *uold,
                          double *out_Kvals, double *out_rhs) {
    if (m.N<=0 || p.dt<=0.0) return false;      // transient-only
    ChCtx &g=g_chctx;
    const int N=m.N, nbtot=m.nbtot, ndof=2, rows=N*ndof;

    if (!g.uploaded || g.N!=N || g.nbtot!=nbtot || g.nnz!=nnz) {
        cudaFree(g.off); cudaFree(g.nbr); cudaFree(g.rev); cudaFree(g.csr_off); cudaFree(g.csr_col);
        cudaFree(g.lapbond); cudaFree(g.vol); cudaFree(g.mnode); cudaFree(g.isghost);
        cudaFree(g.u); cudaFree(g.uold); cudaFree(g.Kvals); cudaFree(g.rhs);
        g=ChCtx{};
        bool ok=true;
        ok=ok && devAlloc(&g.off,N+1)        && devCopyH2D(g.off,m.off,N+1);
        ok=ok && devAlloc(&g.nbr,nbtot)      && devCopyH2D(g.nbr,m.nbr,nbtot);
        ok=ok && devAlloc(&g.rev,nbtot)      && devCopyH2D(g.rev,m.rev,nbtot);
        ok=ok && devAlloc(&g.lapbond,nbtot)  && devCopyH2D(g.lapbond,m.lapbond,nbtot);
        ok=ok && devAlloc(&g.vol,N)          && devCopyH2D(g.vol,m.vol,N);
        ok=ok && devAlloc(&g.csr_off,rows+1) && devCopyH2D(g.csr_off,csr_off,rows+1);
        ok=ok && devAlloc(&g.csr_col,nnz)    && devCopyH2D(g.csr_col,csr_col,nnz);
        ok=ok && devAlloc(&g.mnode,N);
        ok=ok && devAlloc(&g.u,rows) && devAlloc(&g.uold,rows) && devAlloc(&g.Kvals,nnz) && devAlloc(&g.rhs,rows);
        // boundary-ghost flags (fall back to a zeroed array = no ghosts if absent)
        ok=ok && devAlloc(&g.isghost,N);
        if (ok) { if (m.isghost) ok=ok && devCopyH2D(g.isghost,m.isghost,N);
                  else ok=ok && (cudaMemset(g.isghost,0,(size_t)N*sizeof(char))==cudaSuccess); }
        if (!ok) { g=ChCtx{}; return false; }
        g.N=N; g.nbtot=nbtot; g.nnz=nnz; g.rows=rows; g.uploaded=true;
        static bool announced=false;
        if (!announced) { std::fprintf(stderr,"[PeriX CUDA] GPU conservative Cahn-Hilliard assembler active: "
                          "N=%d nnz=%d\n",N,nnz); announced=true; }
    }
    // Picard-frozen mobility + solution (re)uploaded each Newton step
    if (!devCopyH2D(g.mnode,m.mnode,N)) return false;
    if (!devCopyH2D(g.u,u,rows)) return false;
    if (!devCopyH2D(g.uold,uold,rows)) return false;

    const int blk=128, grd=(N+blk-1)/blk;
    cahnHilliardKernel<<<grd,blk>>>(N,p,g.off,g.nbr,g.rev,g.lapbond,g.vol,g.isghost,g.mnode,
                                    g.csr_off,g.csr_col,g.u,g.uold,g.Kvals,g.rhs);
    if (!cudaOk(cudaGetLastError(),"cahnhilliard launch")) return false;
    if (!cudaOk(cudaDeviceSynchronize(),"cahnhilliard sync")) return false;
    if (!cudaOk(cudaMemcpy(out_Kvals,g.Kvals,(size_t)nnz*sizeof(double),cudaMemcpyDeviceToHost),"D2H(Kvals)")) return false;
    if (!cudaOk(cudaMemcpy(out_rhs,g.rhs,(size_t)rows*sizeof(double),cudaMemcpyDeviceToHost),"D2H(rhs)")) return false;
    return true;
}

bool assembleStressCahnHilliard(const AssembleMesh &m, const StressCahnHilliardParams &p,
                                const OpSlots &s,
                                const int *rev, const double *lapbond,
                                const char *isghost, const double *mnode,
                                const double *health, const double *vel,
                                const double *u, const double *uold,
                                double *out_Kvals, double *out_rhs) {
    if (m.N<=0 || p.dt<=0.0 || (p.dim!=2 && p.dim!=3)) return false;
    StressChCtx &g=g_stresschctx;
    const int N=m.N, nnz=m.nnz, nop=m.nop, ndof=p.ndof, rows=N*ndof;
    const int nbtot=m.neigh_off[N];
    const bool wantVel=(p.rho>0.0 && vel);

    if (!g.uploaded || g.N!=N || g.nbtot!=nbtot || g.nnz!=nnz || g.nop!=nop || g.dim!=p.dim
        || ((g.health!=nullptr)!=(health!=nullptr)) || ((g.vel!=nullptr)!=wantVel)) {
        cudaFree(g.csr_off); cudaFree(g.csr_col); cudaFree(g.neigh_off); cudaFree(g.neigh_id);
        cudaFree(g.qp); cudaFree(g.qq); cudaFree(g.qr); cudaFree(g.coords); cudaFree(g.vols);
        cudaFree(g.factprod); cudaFree(g.nodeHorizon); cudaFree(g.nodeSpacing);
        cudaFree(g.rev); cudaFree(g.lapbond); cudaFree(g.isghost); cudaFree(g.mnode);
        cudaFree(g.health); cudaFree(g.vel);
        cudaFree(g.u); cudaFree(g.uold); cudaFree(g.Kvals); cudaFree(g.rhs);
        cudaFree(g.bondOps); cudaFree(g.opsValid);
        g=StressChCtx{};
        bool ok=true;
        ok=ok && devAlloc(&g.csr_off,rows+1)  && devCopyH2D(g.csr_off,m.csr_off,rows+1);
        ok=ok && devAlloc(&g.csr_col,nnz)     && devCopyH2D(g.csr_col,m.csr_col,nnz);
        ok=ok && devAlloc(&g.neigh_off,N+1)   && devCopyH2D(g.neigh_off,m.neigh_off,N+1);
        ok=ok && devAlloc(&g.neigh_id,nbtot)  && devCopyH2D(g.neigh_id,m.neigh_id,nbtot);
        ok=ok && devAlloc(&g.coords,3*N)      && devCopyH2D(g.coords,m.coords,3*N);
        ok=ok && devAlloc(&g.vols,N)          && devCopyH2D(g.vols,m.vols,N);
        ok=ok && devAlloc(&g.qp,nop)          && devCopyH2D(g.qp,m.qp,nop);
        ok=ok && devAlloc(&g.qq,nop)          && devCopyH2D(g.qq,m.qq,nop);
        ok=ok && devAlloc(&g.qr,nop)          && devCopyH2D(g.qr,m.qr,nop);
        ok=ok && devAlloc(&g.factprod,nop)    && devCopyH2D(g.factprod,m.factprod,nop);
        ok=ok && devAlloc(&g.rev,nbtot)       && devCopyH2D(g.rev,rev,nbtot);
        ok=ok && devAlloc(&g.lapbond,nbtot)   && devCopyH2D(g.lapbond,lapbond,nbtot);
        ok=ok && devAlloc(&g.mnode,N);        // (re)uploaded per Newton step below
        ok=ok && devAlloc(&g.u,rows) && devAlloc(&g.uold,rows) && devAlloc(&g.Kvals,nnz) && devAlloc(&g.rhs,rows);
        // boundary-ghost flags (fall back to a zeroed array = no ghosts if absent)
        ok=ok && devAlloc(&g.isghost,N);
        if (ok) { if (isghost) ok=ok && devCopyH2D(g.isghost,isghost,N);
                  else ok=ok && (cudaMemset(g.isghost,0,(size_t)N*sizeof(char))==cudaSuccess); }
        // variable horizon (uniform if the host passed null)
        if (ok && m.nodeHorizon && m.nodeSpacing) {
            ok=ok && devAlloc(&g.nodeHorizon,N) && devCopyH2D(g.nodeHorizon,m.nodeHorizon,N);
            ok=ok && devAlloc(&g.nodeSpacing,N) && devCopyH2D(g.nodeSpacing,m.nodeSpacing,N);
        }
        // frac: per-bond health (nbtot) + committed velocity (N*dim); (re)uploaded per call
        if (ok && health)  ok=ok && devAlloc(&g.health,nbtot);
        if (ok && wantVel) ok=ok && devAlloc(&g.vel,(size_t)N*p.dim);
        if (!ok) { g=StressChCtx{}; return false; }
        g.N=N; g.nbtot=nbtot; g.nnz=nnz; g.nop=nop; g.rows=rows; g.dim=p.dim; g.uploaded=true;
        static bool announced=false;
        if (!announced) { std::fprintf(stderr,"[PeriX CUDA] GPU small-strain stress-CH assembler active: "
                          "N=%d nnz=%d ndof=%d%s\n",N,nnz,ndof, health?" (+fracture)":""); announced=true; }
    }
    if (!g.opsBuilt) {
        buildDeviceOpsCache(N, nbtot, nop, p.dim, g.qp, g.qq, g.qr, g.factprod,
                            g.coords, g.vols, g.neigh_off, g.neigh_id,
                            g.nodeHorizon, g.nodeSpacing,
                            m.delta, m.dx_char, &g.bondOps, &g.opsValid, g.opsBuilt);
    }

    if (!devCopyH2D(g.mnode,mnode,N)) return false;
    if (health  && !devCopyH2D(g.health,health,nbtot)) return false;
    if (wantVel && !devCopyH2D(g.vel,vel,(size_t)N*p.dim)) return false;
    if (!devCopyH2D(g.u,u,rows)) return false;
    if (!devCopyH2D(g.uold,uold,rows)) return false;

    const int blk=128, grd=(N+blk-1)/blk;
    stressCahnHilliardKernel<<<grd,blk>>>(N, p, s, g.csr_off, g.csr_col, g.neigh_off, g.neigh_id,
        g.coords, g.vols, g.qp, g.qq, g.qr, g.factprod, nop, m.delta, m.dx_char,
        g.nodeHorizon, g.nodeSpacing, g.rev, g.lapbond, g.isghost, g.mnode,
        health?g.health:nullptr, wantVel?g.vel:nullptr,
        g.bondOps, g.opsValid,
        g.u, g.uold, g.Kvals, g.rhs);
    if (!cudaOk(cudaGetLastError(),"stress-CH launch")) return false;
    if (!cudaOk(cudaDeviceSynchronize(),"stress-CH sync")) return false;
    if (!cudaOk(cudaMemcpy(out_Kvals,g.Kvals,(size_t)nnz*sizeof(double),cudaMemcpyDeviceToHost),"D2H(Kvals)")) return false;
    if (!cudaOk(cudaMemcpy(out_rhs,g.rhs,(size_t)rows*sizeof(double),cudaMemcpyDeviceToHost),"D2H(rhs)")) return false;
    return true;
}

bool assembleCached(const AssembleMesh &m, const CachedParams &p, const OpSlots &s,
                    const double *nodecache, const double *bondcache,
                    const double *u, const double *uold,
                    double *out_Kvals, double *out_rhs) {
    if (m.N<=0 || (p.dim!=2 && p.dim!=3) || p.ncache<=0) return false;
    CachedCtx &g=g_cachedctx;
    const int N=m.N, nnz=m.nnz, nop=m.nop, ndof=p.ndof, rows=N*ndof, ncache=p.ncache;
    const int nbtot=m.neigh_off[N];

    if (!g.uploaded || g.N!=N || g.nbtot!=nbtot || g.nnz!=nnz || g.nop!=nop
        || g.dim!=p.dim || g.ncache!=ncache || g.rows!=rows || g.nbcache!=p.nbcache
        || ((g.bondcache!=nullptr)!=(bondcache!=nullptr))) {
        cudaFree(g.csr_off); cudaFree(g.csr_col); cudaFree(g.neigh_off); cudaFree(g.neigh_id);
        cudaFree(g.qp); cudaFree(g.qq); cudaFree(g.qr); cudaFree(g.coords); cudaFree(g.vols);
        cudaFree(g.factprod); cudaFree(g.nodeHorizon); cudaFree(g.nodeSpacing);
        cudaFree(g.nodecache); cudaFree(g.bondcache); cudaFree(g.u); cudaFree(g.uold); cudaFree(g.Kvals); cudaFree(g.rhs);
        cudaFree(g.bondOps); cudaFree(g.opsValid);
        g=CachedCtx{};
        bool ok=true;
        ok=ok && devAlloc(&g.csr_off,rows+1)  && devCopyH2D(g.csr_off,m.csr_off,rows+1);
        ok=ok && devAlloc(&g.csr_col,nnz)     && devCopyH2D(g.csr_col,m.csr_col,nnz);
        ok=ok && devAlloc(&g.neigh_off,N+1)   && devCopyH2D(g.neigh_off,m.neigh_off,N+1);
        ok=ok && devAlloc(&g.neigh_id,nbtot)  && devCopyH2D(g.neigh_id,m.neigh_id,nbtot);
        ok=ok && devAlloc(&g.coords,3*N)      && devCopyH2D(g.coords,m.coords,3*N);
        ok=ok && devAlloc(&g.vols,N)          && devCopyH2D(g.vols,m.vols,N);
        ok=ok && devAlloc(&g.qp,nop)          && devCopyH2D(g.qp,m.qp,nop);
        ok=ok && devAlloc(&g.qq,nop)          && devCopyH2D(g.qq,m.qq,nop);
        ok=ok && devAlloc(&g.qr,nop)          && devCopyH2D(g.qr,m.qr,nop);
        ok=ok && devAlloc(&g.factprod,nop)    && devCopyH2D(g.factprod,m.factprod,nop);
        ok=ok && devAlloc(&g.nodecache,(size_t)N*ncache);
        if (bondcache) ok=ok && devAlloc(&g.bondcache,(size_t)nbtot*p.nbcache);   // (re)uploaded per call
        ok=ok && devAlloc(&g.u,rows) && devAlloc(&g.Kvals,nnz) && devAlloc(&g.rhs,rows);
        if (p.dt>0.0) ok=ok && devAlloc(&g.uold,rows);   // transient types only
        if (ok && m.nodeHorizon && m.nodeSpacing) {
            ok=ok && devAlloc(&g.nodeHorizon,N) && devCopyH2D(g.nodeHorizon,m.nodeHorizon,N);
            ok=ok && devAlloc(&g.nodeSpacing,N) && devCopyH2D(g.nodeSpacing,m.nodeSpacing,N);
        }
        if (!ok) { g=CachedCtx{}; return false; }
        g.N=N; g.nbtot=nbtot; g.nnz=nnz; g.nop=nop; g.rows=rows; g.dim=p.dim; g.ncache=ncache; g.nbcache=p.nbcache; g.uploaded=true;
        static bool announced=false;
        if (!announced) { std::fprintf(stderr,"[PeriX CUDA] GPU cached PDDO-callback assembler active: "
                          "N=%d nnz=%d ndof=%d ncache=%d\n",N,nnz,ndof,ncache); announced=true; }
    }
    if (!g.opsBuilt) {
        buildDeviceOpsCache(N, nbtot, nop, p.dim, g.qp, g.qq, g.qr, g.factprod,
                            g.coords, g.vols, g.neigh_off, g.neigh_id,
                            g.nodeHorizon, g.nodeSpacing,
                            m.delta, m.dx_char, &g.bondOps, &g.opsValid, g.opsBuilt);
    }

    // nodecache (Picard-frozen) + optional bondcache + solution (re)uploaded each Newton step
    if (!devCopyH2D(g.nodecache,nodecache,(size_t)N*ncache)) return false;
    if (bondcache && !devCopyH2D(g.bondcache,bondcache,(size_t)nbtot*p.nbcache)) return false;
    if (!devCopyH2D(g.u,u,rows)) return false;
    if (uold && g.uold) { if (!devCopyH2D(g.uold,uold,rows)) return false; }

    const int blk=128, grd=(N+blk-1)/blk;
    cachedAssembleKernel<<<grd,blk>>>(N, p, s, g.csr_off, g.csr_col, g.neigh_off, g.neigh_id,
        g.coords, g.vols, g.qp, g.qq, g.qr, g.factprod, nop, m.delta, m.dx_char,
        g.nodeHorizon, g.nodeSpacing, g.bondOps, g.opsValid,
        g.nodecache, bondcache?g.bondcache:nullptr,
        g.u, (p.dt>0.0)?g.uold:nullptr, g.Kvals, g.rhs);
    if (!cudaOk(cudaGetLastError(),"cached launch")) return false;
    if (!cudaOk(cudaDeviceSynchronize(),"cached sync")) return false;
    if (!cudaOk(cudaMemcpy(out_Kvals,g.Kvals,(size_t)nnz*sizeof(double),cudaMemcpyDeviceToHost),"D2H(Kvals)")) return false;
    if (!cudaOk(cudaMemcpy(out_rhs,g.rhs,(size_t)rows*sizeof(double),cudaMemcpyDeviceToHost),"D2H(rhs)")) return false;
    return true;
}

} // namespace perix_cuda
