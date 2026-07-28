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
//+++ Function: PDDO moment-matrix assembly and per-bond operator
//+++           evaluation. Bond vectors are normalized to
//+++             eta = xi / delta
//+++           before forming the moment matrix and the per-bond
//+++           monomials. This keeps every A entry O(1) regardless
//+++           of the operator order — without normalization the
//+++           xi^|q| weights at high orders underflow to ~delta^|q|
//+++           and the |q| x |q| solve loses precision (this matters
//+++           for orders >= 5). The output operator value is rescaled
//+++           by 1/delta^|q_k| so the on-the-wire derivative is in
//+++           original x-coordinates:
//+++             ops_k(i,j) = ( sum_a a_mat[a,k] * eta^{q_a} )
//+++                          * w * vc * V_j / delta^|q_k|
//+++           where the bond weight is
//+++             bondWeight = w(|xi|) * vc(|xi|) * V_j
//+++           with the Madenci partial-volume correction
//+++             vc = 1                                if |xi| <= delta - dx/2
//+++                = (delta - |xi| + dx/2)/dx         if |xi| in [delta-dx/2, delta+dx/2]
//+++                = 0                                otherwise
//+++           (matches `vc` in the reference Matlab regFastNewHorizon /
//+++           main_pddo_*.m files). The rim correction is gated by
//+++           PDMeshData.VolumeCorrection (input: PDMesh.VolumeCorrection,
//+++           default true); when OFF every in-horizon bond carries its
//+++           full cell volume, which is the plain-volume-sum convention
//+++           of the Kalthoff--Winkler reference drivers.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDOperators/PDOperators.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

inline double computeWeight(const double &xsi_norm, const double &delta) {
    const double r = 2.0 * xsi_norm / delta;
    return std::exp(-r * r);
}

inline double partialVolumeCorrection(const double &xsi_norm,
                                      const double &delta,
                                      const double &dx_char) {
    if (dx_char <= 0.0) return 1.0;
    if (xsi_norm <= delta - 0.5 * dx_char) return 1.0;
    const double vc = (delta - xsi_norm + 0.5 * dx_char) / dx_char;
    return std::clamp(vc, 0.0, 1.0);
}

// Opt-in replay verification (PERIX_OPCACHE_VERIFY=1): every replayed
// operator vector is bit-compared against a live recomputation in the same
// run. Any mismatch is reported (first few loudly) — the expected result is
// ZERO mismatches over the whole run.
inline bool opcacheVerifyOn() {
    static const bool on = [] {
        const char *e=std::getenv("PERIX_OPCACHE_VERIFY");
        return e && e[0]=='1';
    }();
    return on;
}

} // namespace

void PDOperators::computeXsiMonomials(const double &xsi_x, const double &xsi_y, const double &xsi_z) {
    for (int k = 0; k < m_OperatorsVecSize; ++k) {
        const int p = m_QIndices[static_cast<std::size_t>(k)][0];
        const int q = m_QIndices[static_cast<std::size_t>(k)][1];
        const int r = m_QIndices[static_cast<std::size_t>(k)][2];
        double mono = 1.0;
        for (int t = 0; t < p; ++t) mono *= xsi_x;
        for (int t = 0; t < q; ++t) mono *= xsi_y;
        for (int t = 0; t < r; ++t) mono *= xsi_z;
        m_XsiOperatorsVec(k + 1) = mono;
    }
}

void PDOperators::calcAMatrix(const int &NodeI, const PDMeshData &Data) {
    // Replay mode: the per-bond operators for this node come verbatim from the
    // geometry cache (built once with calcAMatrixCompute/the calcOperators tail
    // below), so the moment system was already assembled, solved and
    // degenerate-checked with production semantics. Just park the family
    // cursor for the sequential calcOperators replay.
    if (cacheServes(NodeI)) {
        m_CursorNode=NodeI;
        m_CursorPos=0;
        if (opcacheVerifyOn()) {
            // verification mode recomputes each replayed bond live, which
            // needs this node's moment solution in m_Amat.
            calcAMatrixCompute(NodeI,Data);
        }
        return;
    }
    calcAMatrixCompute(NodeI,Data);
}

void PDOperators::calcAMatrixCompute(const int &NodeI, const PDMeshData &Data) {
    m_AMATRIX.setToZeros();
    m_Amat.setToZeros();

    // Per-node (variable) horizon: node i's moment matrix and operators use its
    // own delta_i / spacing. NodeHorizon is empty unless VariableHorizon is on,
    // so the uniform path reads the global HorizonRadius / max(DX,DY,DZ) exactly
    // as before.
    const bool   varHorizon = !Data.NodeHorizon.empty();
    const double delta   = varHorizon ? Data.NodeHorizon[NodeI - 1] : Data.HorizonRadius;
    const double dx_char = varHorizon ? Data.NodeSpacing[NodeI - 1]
                                      : std::max(Data.DX, std::max(Data.DY, Data.DZ));
    if (delta <= 0.0) {
        MessagePrinter::printErrorTxt("PDOperators::calcAMatrix: HorizonRadius must be > 0");
        MessagePrinter::exitPeriX();
    }
    const double inv_delta = 1.0 / delta;

    // Cache 1/delta^|q_k| per operator slot for use in calcOperators.
    // Computed every calcAMatrix call so that a runtime change of
    // delta (between nodes, in principle non-uniform PD setups) is
    // picked up automatically.
    if (static_cast<int>(m_OpInvDeltaPow.size()) != m_OperatorsVecSize) {
        m_OpInvDeltaPow.assign(static_cast<std::size_t>(m_OperatorsVecSize), 1.0);
    }
    for (int k = 0; k < m_OperatorsVecSize; ++k) {
        const int total = m_QIndices[static_cast<std::size_t>(k)][0]
                        + m_QIndices[static_cast<std::size_t>(k)][1]
                        + m_QIndices[static_cast<std::size_t>(k)][2];
        double v = 1.0;
        for (int t = 0; t < total; ++t) v *= inv_delta;
        m_OpInvDeltaPow[static_cast<std::size_t>(k)] = v;
    }

    const double x1 = Data.NodeCoords[(NodeI - 1) * 3 + 0];
    const double y1 = Data.NodeCoords[(NodeI - 1) * 3 + 1];
    const double z1 = Data.NodeCoords[(NodeI - 1) * 3 + 2];

    const auto &neighbors = Data.NodesNeighNodesID[NodeI - 1];
    int usableBonds = 0;
    for (const int NodeJ : neighbors) {
        // Crack-respecting mode (BC rows): a bond severed by a pre-existing
        // crack is excluded from the moment system so the operator is strictly
        // one-sided at the notch even when the family spans it (force-only).
        if (m_RespectInitialCracks && initialCrackCutsBond(Data, NodeI, NodeJ)) continue;
        const double x2 = Data.NodeCoords[(NodeJ - 1) * 3 + 0];
        const double y2 = Data.NodeCoords[(NodeJ - 1) * 3 + 1];
        const double z2 = Data.NodeCoords[(NodeJ - 1) * 3 + 2];
        const double xsi_x = x2 - x1;
        const double xsi_y = y2 - y1;
        const double xsi_z = z2 - z1;
        const double xsi_norm = std::sqrt(xsi_x * xsi_x + xsi_y * xsi_y + xsi_z * xsi_z);
        if (xsi_norm <= 0.0) continue;
        // With a variable horizon the neighbour list is the DUAL family, which
        // can contain j with |xi| > delta_i (i is inside H_j but j is outside
        // H_i). Node i's operator only fits its OWN horizon H_i, so skip those.
        if (varHorizon && xsi_norm > delta) continue;

        const double w  = computeWeight(xsi_norm, delta);
        const double vc = Data.VolumeCorrection
                        ? partialVolumeCorrection(xsi_norm, delta, dx_char) : 1.0;
        const double VolJ = Data.NodeVolumes[NodeJ - 1];
        const double bondWeight = w * vc * VolJ;
        if (bondWeight <= 0.0) continue;
        ++usableBonds;

        // Work in normalized bond coordinates eta = xi / delta so the
        // moment matrix entries stay O(1) for all orders.
        computeXsiMonomials(xsi_x * inv_delta, xsi_y * inv_delta, xsi_z * inv_delta);
        for (int ii = 1; ii <= m_OperatorsVecSize; ++ii) {
            const double xi = m_XsiOperatorsVec(ii);
            for (int jj = 1; jj <= m_OperatorsVecSize; ++jj) {
                m_AMATRIX(ii, jj) += bondWeight * xi * m_XsiOperatorsVec(jj);
            }
        }
    }

    // Solve A * a = diag(b) for ALL operator columns with ONE full-pivot LU
    // factorization (the old code re-factored A once per column), and verify
    // the moment system is CONSISTENT. A rank-deficient A is acceptable when
    // diag(b) is still in range(A) -- the symmetric-family / high-order case,
    // where the orthogonality constraints are met exactly and the null-space
    // part of a is irrelevant -- but an INCONSISTENT system means the family
    // genuinely cannot support the requested Order (a crack-isolated point, a
    // boundary ghost with a too-small horizon, a collinear neighbour layout).
    // The old per-column solve() silently returned garbage operators there.
    // The solve always runs (it fills m_Amat with the LU particular solution,
    // matching the pre-check behaviour when the abort is disabled).
    const bool consistent = m_AMATRIX.solveMatrixChecked(m_BDiag, m_Amat, 1.0e-8);
    const bool degenerate = (usableBonds < m_OperatorsVecSize || !consistent);
    m_LastFamilyDegenerate = degenerate;
    // A degenerate family is a genuine error for a BULK material point or for the
    // single wall (layer-1) ghost a strong-form BC reconstructs on. A DEEPER
    // mirror ghost (PDMesh ghost_layer>1, GhostLayerIndex>1) is only a
    // family-COMPLETION reflection node: its own row is always BC-slaved (the
    // Dirichlet/Neumann/flux/strong-form-traction reflection ties it to its bulk)
    // and bulk neighbours use its reflected DoF VALUE, never its operator. So when
    // such a ghost lands beyond a domain corner with a one-sided family, ZERO its
    // operators -- a singular moment system would otherwise return NaN/huge LU
    // entries that leak into the assembled (then BC-overwritten) ghost row and
    // poison the diagonal bcRowScale reads. Do NOT abort.
    bool deepMirrorGhost = false;
    if (NodeI>=1 && NodeI<=static_cast<int>(Data.GhostLayerIndex.size()))
        deepMirrorGhost = (Data.GhostLayerIndex[static_cast<std::size_t>(NodeI-1)] > 1);
    if (degenerate && deepMirrorGhost) {
        m_Amat.setToZeros();
    }
    else if (degenerate && m_AbortOnDegenerate) {
        MessagePrinter::printErrorTxt(
            "PDOperators::calcAMatrix: degenerate moment matrix at node "
            + std::to_string(NodeI) + " (usable bonds=" + std::to_string(usableBonds)
            + ", monomials=" + std::to_string(m_OperatorsVecSize)
            + " for Order=" + std::to_string(m_Order)
            + "). The family cannot support this Order: increase "
              "HorizonRadiusFactor, lower PDMesh Order, or check for a "
              "crack-isolated / degenerate point.");
        MessagePrinter::exitPeriX();
    }
}

int PDOperators::cacheFamilyPos(const int &NodeI,const int &NodeJ,const PDMeshData &Data) {
    const auto &fam=Data.NodesNeighNodesID[static_cast<std::size_t>(NodeI-1)];
    if (m_CursorNode!=NodeI) { m_CursorNode=NodeI; m_CursorPos=0; }
    if (m_CursorPos<static_cast<int>(fam.size())
        && fam[static_cast<std::size_t>(m_CursorPos)]==NodeJ) {
        const int p=m_CursorPos;
        ++m_CursorPos;
        return p;
    }
    // out-of-order caller: linear scan of the family, then resume sequentially
    for (int t=0;t<static_cast<int>(fam.size());++t) {
        if (fam[static_cast<std::size_t>(t)]==NodeJ) {
            m_CursorPos=t+1;
            return t;
        }
    }
    return -1;
}

namespace {
    long long g_VerifyChecked=0;
    long long g_VerifyMismatch=0;
}

void PDOperators::calcOperators(const int &NodeI, const int &NodeJ, const PDMeshData &Data) {
    // Replay mode: copy the cached per-bond operator vector (bitwise identical
    // to the live computation below — it IS that computation, stored). The
    // plain array serves the crack-respecting mode too when the mesh has no
    // preset cracks (the crack gate is then a no-op).
    if (cacheServes(NodeI)) {
        const int pos=cacheFamilyPos(NodeI,NodeJ,Data);
        if (pos>=0) {
            const bool cracked = m_RespectInitialCracks && m_Cache->hasCrackVariant;
            const std::vector<double> &src = cracked ? m_Cache->opsCracked : m_Cache->ops;
            const double *v = src.data()
                + (static_cast<std::size_t>(m_Cache->nodeOff[static_cast<std::size_t>(NodeI-1)])
                   + static_cast<std::size_t>(pos)) * static_cast<std::size_t>(m_OperatorsVecSize);
            if (opcacheVerifyOn()) {
                // recompute live (m_Amat is valid: verify mode also runs the
                // full calcAMatrixCompute in the calcAMatrix wrapper) and
                // demand bit equality with the stored vector.
                calcOperatorsCompute(NodeI,NodeJ,Data);
                long long bad=0;
                for (int k=1;k<=m_OperatorsVecSize;++k) {
                    if (m_OperatorsVec(k)!=v[k-1]
                        && !(m_OperatorsVec(k)!=m_OperatorsVec(k) && v[k-1]!=v[k-1])) {
                        ++bad;
                    }
                }
#pragma omp atomic
                g_VerifyChecked+=1;
                if (bad>0) {
#pragma omp atomic
                    g_VerifyMismatch+=1;
                    if (g_VerifyMismatch<=10) {
                        std::printf("[opcache-verify] MISMATCH bond (%d,%d): %lld slots differ\n",
                                    NodeI,NodeJ,bad);
                        std::fflush(stdout);
                    }
                }
            }
            for (int k=1;k<=m_OperatorsVecSize;++k) {
                m_OperatorsVec(k)=v[k-1];
            }
            return;
        }
        // NodeJ is not in NodeI's family (an arbitrary-pair caller): fall back
        // to the live path. The cache-mode calcAMatrix skipped the moment
        // solve, so rebuild it first — m_Amat must belong to NodeI here.
        calcAMatrixCompute(NodeI,Data);
        m_CursorNode=0;
        m_CursorPos=0;
    }
    calcOperatorsCompute(NodeI,NodeJ,Data);
}

long long PDOperators::getVerifyChecked()  { return g_VerifyChecked; }
long long PDOperators::getVerifyMismatch() { return g_VerifyMismatch; }

void PDOperators::calcOperatorsCompute(const int &NodeI, const int &NodeJ, const PDMeshData &Data) {
    // Use node i's own horizon (the operator belongs to node i); see calcAMatrix.
    const bool   varHorizon = !Data.NodeHorizon.empty();
    const double delta   = varHorizon ? Data.NodeHorizon[NodeI - 1] : Data.HorizonRadius;
    const double dx_char = varHorizon ? Data.NodeSpacing[NodeI - 1]
                                      : std::max(Data.DX, std::max(Data.DY, Data.DZ));
    const double inv_delta = 1.0 / delta;

    const double x1 = Data.NodeCoords[(NodeI - 1) * 3 + 0];
    const double y1 = Data.NodeCoords[(NodeI - 1) * 3 + 1];
    const double z1 = Data.NodeCoords[(NodeI - 1) * 3 + 2];
    const double x2 = Data.NodeCoords[(NodeJ - 1) * 3 + 0];
    const double y2 = Data.NodeCoords[(NodeJ - 1) * 3 + 1];
    const double z2 = Data.NodeCoords[(NodeJ - 1) * 3 + 2];
    const double xsi_x = x2 - x1;
    const double xsi_y = y2 - y1;
    const double xsi_z = z2 - z1;
    const double xsi_norm = std::sqrt(xsi_x * xsi_x + xsi_y * xsi_y + xsi_z * xsi_z);

    m_OperatorsVec.setToZeros();
    if (xsi_norm <= 0.0) return;
    // Dual family may include j beyond node i's own horizon; its operator is zero
    // there (it only fits H_i). See calcAMatrix.
    if (varHorizon && xsi_norm > delta) return;
    // Crack-respecting mode: a preset-crack bond carries no operator (it was
    // excluded from the moment system in calcAMatrix as well).
    if (m_RespectInitialCracks && initialCrackCutsBond(Data, NodeI, NodeJ)) return;

    const double w  = computeWeight(xsi_norm, delta);
    const double vc = Data.VolumeCorrection
                    ? partialVolumeCorrection(xsi_norm, delta, dx_char) : 1.0;
    const double VolJ = Data.NodeVolumes[NodeJ - 1];
    const double bondWeight = w * vc * VolJ;
    if (bondWeight <= 0.0) return;

    // Match the normalization used in calcAMatrix.
    computeXsiMonomials(xsi_x * inv_delta, xsi_y * inv_delta, xsi_z * inv_delta);

    // op_k(i,j) = ( sum_a a_mat[a,k] * eta^{q_a} ) * bondWeight / delta^|q_k|
    for (int k = 1; k <= m_OperatorsVecSize; ++k) {
        double sum = 0.0;
        for (int a = 1; a <= m_OperatorsVecSize; ++a) {
            sum += m_Amat(a, k) * m_XsiOperatorsVec(a);
        }
        m_OperatorsVec(k) = sum * bondWeight
                          * m_OpInvDeltaPow[static_cast<std::size_t>(k - 1)];
    }
}

void PDOperators::buildGeometryCache(const PDMeshData &Data, const double &MaxGB) {
    // Drop any previous cache first so every computation below runs the live
    // path (m_Cache empty => cacheServes() false).
    m_Cache.reset();
    m_CursorNode=0;
    m_CursorPos=0;

    const int N=Data.NodesNum;
    if (N<=0 || m_OperatorsVecSize<=0) return;

    auto cache=std::make_shared<GeometryOpCache>();
    cache->nodeOff.assign(static_cast<std::size_t>(N)+1,0);
    long long nb=0;
    for (int i=0;i<N;++i) {
        nb+=static_cast<long long>(Data.NodesNeighNodesID[static_cast<std::size_t>(i)].size());
        cache->nodeOff[static_cast<std::size_t>(i)+1]=nb;
    }
    if (nb<=0) return;

    const bool wantCracked=Data.hasInitialCracks();
    const double gib=static_cast<double>(nb)*m_OperatorsVecSize*sizeof(double)
                     *(wantCracked?2.0:1.0)/(1024.0*1024.0*1024.0);
    if (MaxGB>0.0 && gib>MaxGB) {
        constexpr int bufSize=200;
        char buffer[bufSize];
        std::snprintf(buffer,bufSize,
            "PDOperators: geometry-operator cache skipped (%.2f GiB > %.1f GiB limit); "
            "operators are recomputed per call",gib,MaxGB);
        MessagePrinter::printNormalTxt(buffer);
        return;
    }

    const std::size_t nop=static_cast<std::size_t>(m_OperatorsVecSize);
    cache->ops.resize(static_cast<std::size_t>(nb)*nop);
    cache->hasCrackVariant=wantCracked;
    if (wantCracked) {
        cache->opsCracked.resize(static_cast<std::size_t>(nb)*nop);
        cache->crackedValid.assign(static_cast<std::size_t>(N),0);
    }

    // Fill in parallel: every (i, j) slice is independent and each thread uses
    // its own PDOperators copy, so the stored values are identical for any
    // thread count — each is exactly what a serial live call would produce.
    // The plain pass keeps the production degenerate-family abort (this build
    // runs where the first assembly used to hit it); the crack-respecting pass
    // must NOT abort, because production only evaluates flagged operators on
    // the BC-touched ghost rows — a degenerate flagged family elsewhere is
    // simply marked invalid and served by live recomputation on demand.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        PDOperators local=*this;   // m_Cache empty -> live compute path
#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
        for (int i=1;i<=N;++i) {
            local.calcAMatrixCompute(i,Data);
            const auto &fam=Data.NodesNeighNodesID[static_cast<std::size_t>(i-1)];
            double *dst=cache->ops.data()
                +static_cast<std::size_t>(cache->nodeOff[static_cast<std::size_t>(i-1)])*nop;
            for (std::size_t t=0;t<fam.size();++t) {
                local.calcOperators(i,fam[t],Data);
                for (std::size_t k=0;k<nop;++k) {
                    dst[t*nop+k]=local.m_OperatorsVec(static_cast<int>(k)+1);
                }
            }
        }
    }

    if (wantCracked) {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            PDOperators local=*this;
            local.setRespectInitialCracks(true);
            local.setAbortOnDegenerate(false);
#ifdef _OPENMP
#pragma omp for schedule(guided)
#endif
            for (int i=1;i<=N;++i) {
                local.calcAMatrixCompute(i,Data);
                if (local.m_LastFamilyDegenerate) continue;   // stays invalid
                const auto &fam=Data.NodesNeighNodesID[static_cast<std::size_t>(i-1)];
                double *dst=cache->opsCracked.data()
                    +static_cast<std::size_t>(cache->nodeOff[static_cast<std::size_t>(i-1)])*nop;
                for (std::size_t t=0;t<fam.size();++t) {
                    local.calcOperators(i,fam[t],Data);
                    for (std::size_t k=0;k<nop;++k) {
                        dst[t*nop+k]=local.m_OperatorsVec(static_cast<int>(k)+1);
                    }
                }
                cache->crackedValid[static_cast<std::size_t>(i-1)]=1;
            }
        }
    }

    m_Cache=std::move(cache);

    constexpr int bufSize=200;
    char buffer[bufSize];
    std::snprintf(buffer,bufSize,
        "      geometry-operator cache: %lld bonds x %d ops%s (%.1f MB)",
        nb,m_OperatorsVecSize,wantCracked?" x2 (crack-respecting variant)":"",
        gib*1024.0);
    MessagePrinter::printNormalTxt(buffer);
}
