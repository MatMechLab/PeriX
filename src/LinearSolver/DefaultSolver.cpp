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
//+++ Date    : 2026.05.25
//+++ Function: implementation of PeriX's default in-house direct
//+++           sparse solver. Builds a column-wise profile from the
//+++           CSR pattern, runs an LDU factorization without
//+++           pivoting, and uses forward/back substitution to solve
//+++           A x = b. Profile (skyline) data layout with separate
//+++           upper (AU) and lower (AL) packed arrays; 0-based C++
//+++           indexing throughout.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "LinearSolver/DefaultSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <string>

#include "Utils/MessagePrinter.h"

namespace {
    bool getBoolParameter(const nlohmann::ordered_json &params,
                          const char *key,
                          const bool defaultValue) {
        if (!params.contains(key)) {
            return defaultValue;
        }
        return params.at(key).get<bool>();
    }
    double getDoubleParameter(const nlohmann::ordered_json &params,
                              const char *key,
                              const double defaultValue) {
        if (!params.contains(key)) {
            return defaultValue;
        }
        return params.at(key).get<double>();
    }
} // namespace

DefaultSolver::DefaultSolver() = default;

void DefaultSolver::computeOrdering(SparseMatrix &A) {
    const int n = A.getSize();
    m_Perm.assign(n, 0);
    m_InvPerm.assign(n, 0);

    if (!m_UseOrdering || n == 0) {
        for (int i = 0; i < n; ++i) {
            m_Perm[i]    = i;
            m_InvPerm[i] = i;
        }
        return;
    }

    const int *row_ptr = A.getCSRRowsIndexPtr();
    const int *col_idx = A.getCSRColsIndexPtr();

    // Degree of each vertex = number of CSR nonzeros in its row,
    // discounting the self-loop on the diagonal. For PeriX-built
    // matrices the CSR pattern is structurally symmetric (PD families
    // are reciprocal), so the row-based adjacency is enough.
    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        const int row_begin = row_ptr[i];
        const int row_end   = row_ptr[i + 1];
        int d = 0;
        for (int p = row_begin; p < row_end; ++p) {
            if (col_idx[p] != i) ++d;
        }
        degree[i] = d;
    }

    // Cuthill-McKee BFS, breaking ties by ascending degree. We then
    // reverse to get Reverse Cuthill-McKee. Multiple components are
    // handled by restarting at the next lowest-degree unvisited
    // vertex.
    std::vector<char> visited(n, 0);
    std::vector<int>  order;
    order.reserve(n);

    auto pickComponentStart = [&](int hint) {
        int best = -1;
        int best_deg = std::numeric_limits<int>::max();
        for (int v = hint; v < n; ++v) {
            if (!visited[v] && degree[v] < best_deg) {
                best = v;
                best_deg = degree[v];
                if (best_deg == 0) break;
            }
        }
        return best;
    };

    int component_hint = 0;
    while (true) {
        const int start = pickComponentStart(component_hint);
        if (start < 0) break;
        component_hint = start + 1;

        std::queue<int> q;
        q.push(start);
        visited[start] = 1;

        std::vector<int> nbrs; // reused per pop to avoid allocation churn
        while (!q.empty()) {
            const int v = q.front();
            q.pop();
            order.push_back(v);

            nbrs.clear();
            const int row_begin = row_ptr[v];
            const int row_end   = row_ptr[v + 1];
            for (int p = row_begin; p < row_end; ++p) {
                const int u = col_idx[p];
                if (u != v && !visited[u]) {
                    visited[u] = 1; // mark on enqueue, not on dequeue
                    nbrs.push_back(u);
                }
            }
            std::sort(nbrs.begin(), nbrs.end(),
                      [&degree](int a, int b) { return degree[a] < degree[b]; });
            for (int u : nbrs) q.push(u);
        }
    }

    // Reverse for RCM.
    std::reverse(order.begin(), order.end());

    // Defensive: if anything went wrong (shouldn't), fall back to identity.
    if (static_cast<int>(order.size()) != n) {
        MessagePrinter::printWarningTxt(
            "DefaultSolver: RCM ordering incomplete, falling back to identity");
        for (int i = 0; i < n; ++i) {
            m_Perm[i]    = i;
            m_InvPerm[i] = i;
        }
        return;
    }

    for (int new_idx = 0; new_idx < n; ++new_idx) {
        const int old_idx = order[new_idx];
        m_Perm[new_idx]    = old_idx;
        m_InvPerm[old_idx] = new_idx;
    }
}

bool DefaultSolver::patternMatches(SparseMatrix &A) const {
    if (!m_IsInitialized) return false;
    if (m_CachedRows != A.getSize()) return false;
    if (m_CachedNNZ  != A.getNNZNum()) return false;
    return true; // (rows, nnz) match is sufficient inside PeriX: the
                 // CSR pattern is set up once via initFromPDMesh and
                 // never structurally edited between solve() calls.
}

long long DefaultSolver::buildProfile(SparseMatrix &A) {
    const int n = A.getSize();
    m_N = n;

    // Compute ordering (RCM or identity) before reading sparsity.
    computeOrdering(A);

    const int *row_ptr = A.getCSRRowsIndexPtr();
    const int *col_idx = A.getCSRColsIndexPtr();
    const int  nnz = A.getNNZNum();

    // For each column/row j (in the permuted ordering), find the
    // smallest index k<j such that A(perm[k], perm[j]) != 0 OR
    // A(perm[j], perm[k]) != 0. The unified profile width is
    // K_j = j - first_off[j].
    std::vector<int> first_off(n, -1);
    for (int i_old = 0; i_old < n; ++i_old) {
        const int i = m_InvPerm[i_old]; // new row index
        const int row_begin = row_ptr[i_old];
        const int row_end   = row_ptr[i_old + 1];
        for (int p = row_begin; p < row_end; ++p) {
            const int j_old = col_idx[p];
            const int j     = m_InvPerm[j_old]; // new col index
            if (j < i) {
                if (first_off[i] < 0 || j < first_off[i]) first_off[i] = j;
            } else if (j > i) {
                if (first_off[j] < 0 || i < first_off[j]) first_off[j] = i;
            }
        }
    }

    // Build cumulative profile pointer m_JP[0..n].
    m_JP.assign(n + 1, 0);
    for (int j = 0; j < n; ++j) {
        const int Kj = (first_off[j] < 0) ? 0 : (j - first_off[j]);
        m_JP[j + 1] = m_JP[j] + static_cast<long long>(Kj);
    }
    const long long profile_size = m_JP[n];

    // Size guard BEFORE the allocation: the skyline stores the FULL envelope
    // (AU+AL, one double each per profile entry), i.e. O(n * bandwidth) --
    // super-linear in the mesh, unlike the O(nnz) CSR. On a large 3D system
    // this eager allocation is an OOM (a 200x100x8 PD grid at 3 DoF/node
    // needs ~64 GiB of profile) and the serial LDU would take hours anyway;
    // such runs are almost always better served by PARDISO/cuDSS. Refuse
    // loudly with the numbers instead of taking down the machine.
    const double projected_gib =
        (static_cast<double>(profile_size) * 2.0 + static_cast<double>(n))
        * static_cast<double>(sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    if (m_MaxProfileGB > 0.0 && projected_gib > m_MaxProfileGB) {
        long long max_K = 0;
        for (int j = 0; j < n; ++j) {
            const long long Kj = m_JP[j + 1] - m_JP[j];
            if (Kj > max_K) max_K = Kj;
        }
        const double mean_K = (n > 0) ? (static_cast<double>(profile_size) / n) : 0.0;
        constexpr int bufSize = 256;
        char buffer[bufSize];
        std::snprintf(buffer, bufSize,
                      "DefaultSolver: the skyline profile needs ~%.1f GiB (n=%d, profile=%lld, "
                      "mean/max column width=%.0f/%lld, ordering=%s), exceeding max_profile_gb=%g.",
                      projected_gib, n, profile_size, mean_K, max_K,
                      m_UseOrdering ? "RCM" : "identity", m_MaxProfileGB);
        MessagePrinter::printErrorTxt(std::string(buffer));
        MessagePrinter::printErrorTxt(
            "the in-house profile-LDU solver stores the full skyline envelope (O(n*bandwidth)) and "
            "factors it serially, so a system this size needs a sparse-direct backend instead: set "
            "\"LinearSolver\":{\"type\":\"pardiso\"} (CPU, build with -DUSE_ONEAPI=on) or "
            "{\"type\":\"cudss\"} (GPU, build with -DUSE_CUDSS=on). To insist on the default solver, "
            "raise \"params\":{\"max_profile_gb\":...} or set it <= 0 to disable this guard.");
        MessagePrinter::exitPeriX();
    }

    // Allocate storage; zero-initialized.
    m_AU.assign(static_cast<std::size_t>(profile_size), 0.0);
    m_AL.assign(static_cast<std::size_t>(profile_size), 0.0);
    m_AD.assign(static_cast<std::size_t>(n), 0.0);

    // Update structural fingerprint for fast pattern-match check
    // on subsequent solves.
    m_CachedRows = n;
    m_CachedNNZ  = nnz;

    return profile_size;
}

void DefaultSolver::copyValuesFromCSR(SparseMatrix &A) {
    std::fill(m_AU.begin(), m_AU.end(), 0.0);
    std::fill(m_AL.begin(), m_AL.end(), 0.0);
    std::fill(m_AD.begin(), m_AD.end(), 0.0);

    const int *row_ptr = A.getCSRRowsIndexPtr();
    const int *col_idx = A.getCSRColsIndexPtr();
    const double *values = A.getCSRValuesPtr();

    for (int i_old = 0; i_old < m_N; ++i_old) {
        const int i = m_InvPerm[i_old]; // permuted row
        const int row_begin = row_ptr[i_old];
        const int row_end   = row_ptr[i_old + 1];
        for (int p = row_begin; p < row_end; ++p) {
            const int j_old = col_idx[p];
            const int j     = m_InvPerm[j_old]; // permuted col
            const double v  = values[p];
            if (i == j) {
                m_AD[i] = v;
            } else if (j > i) {
                // Upper: A(i,j) -> AU at col j position (i - (j - K_j))
                const long long Kj = m_JP[j + 1] - m_JP[j];
                const int first_row = j - static_cast<int>(Kj);
                const long long pos = m_JP[j] + static_cast<long long>(i - first_row);
                m_AU[static_cast<std::size_t>(pos)] = v;
            } else { // j < i: lower triangle, entry stored in row i's AL block
                const long long Ki = m_JP[i + 1] - m_JP[i];
                const int first_col = i - static_cast<int>(Ki);
                const long long pos = m_JP[i] + static_cast<long long>(j - first_col);
                m_AL[static_cast<std::size_t>(pos)] = v;
            }
        }
    }
}

bool DefaultSolver::factorize() {
    // LDU factorization for the all-unsymmetric case. After this loop:
    //   m_AU stores U (unit upper triangle entries, divided by D)
    //   m_AL stores L (unit lower triangle entries, divided by D)
    //   m_AD stores 1/D (reciprocal of diagonal factor)
    // During processing of column j, before the diagonal-update step,
    // m_AU[m_JP[j]..] holds the partially-reduced values V_U(k,j) = U(k,j)*D(k)
    // and m_AL[m_JP[j]..] holds V_L(j,k) = L(j,k)*D(k). This is the key
    // identity that lets the inner dot product avoid an explicit D
    // multiplication (the D cancels because previous columns/rows have
    // their values divided by D already, while the current column/row
    // values still carry the D factor).

    const int n = m_N;

    // Raw pointers so the compiler can hoist the array bases out of
    // the hot loop and (with __restrict__) prove non-aliasing for SIMD.
    double *__restrict__ AU = m_AU.data();
    double *__restrict__ AL = m_AL.data();
    double *__restrict__ AD = m_AD.data();
    const long long *__restrict__ JP = m_JP.data();

    for (int j = 0; j < n; ++j) {
        const long long jp_j  = JP[j];
        const long long jp_j1 = JP[j + 1];
        const int Kj = static_cast<int>(jp_j1 - jp_j);

        if (Kj > 0) {
            const int is = j - Kj;

            // Phase 1: reduce off-diagonals using previous (already
            // factored) columns/rows. Single fused loop -- it does
            // half the memory traffic vs splitting into two dot
            // products, which matters because ih is small on average.
            for (int i = is; i < j; ++i) {
                const long long jp_i1 = JP[i + 1];
                const int Ki = static_cast<int>(jp_i1 - JP[i]);
                const int r  = i - is; // 0..Kj-1
                const int ih = (r < Ki) ? r : Ki;
                if (ih <= 0) continue;

                const double *__restrict__ au_j_part = AU + (jp_j + r - ih);
                const double *__restrict__ au_i_fin  = AU + (jp_i1 - ih);
                const double *__restrict__ al_j_part = AL + (jp_j + r - ih);
                const double *__restrict__ al_i_fin  = AL + (jp_i1 - ih);

                double sum_u = 0.0;
                double sum_l = 0.0;
                for (int t = 0; t < ih; ++t) {
                    sum_u += au_j_part[t] * al_i_fin[t];
                    sum_l += al_j_part[t] * au_i_fin[t];
                }
                AU[jp_j + r] -= sum_u;
                AL[jp_j + r] -= sum_l;
            }

            // Phase 2: reduce diagonal and convert V_U->U, V_L->L by dividing
            // by D(k) (we have 1/D stored already for all k<j).
            double diag_sub = 0.0;
            for (int r = 0; r < Kj; ++r) {
                const int k_idx = is + r;
                const double D_inv = AD[k_idx];
                const std::size_t pos = static_cast<std::size_t>(jp_j + r);
                const double V_U = AU[pos];
                const double V_L = AL[pos];
                const double U_kj = V_U * D_inv;
                diag_sub += V_L * U_kj;       // V_L * U_final = L*D*U (V_L = L*D)
                AU[pos] = U_kj;               // store U(k,j)
                AL[pos] = V_L * D_inv;        // store L(j,k)
            }

            AD[j] -= diag_sub;
        }

        // Diagonal pivot check + reciprocal.
        if (AD[j] == 0.0) {
            MessagePrinter::printErrorTxt(
                "DefaultSolver: zero diagonal pivot at equation "
                + std::to_string(j + 1) + " during LDU factorization");
            return false;
        }
        AD[j] = 1.0 / AD[j];
    }

    return true;
}

void DefaultSolver::solveFactored(const VectorXd &b, VectorXd &x) const {
    const int n = m_N;

    // Ensure the scratch buffer exists.
    m_BPerm.assign(static_cast<std::size_t>(n), 0.0);

    // Permute b into m_BPerm: b_perm[new] = b[m_Perm[new]] = b[old].
    {
        const double *__restrict__ bp = b.getDataPtr();
        for (int new_idx = 0; new_idx < n; ++new_idx) {
            m_BPerm[new_idx] = bp[m_Perm[new_idx]];
        }
    }
    double *__restrict__ yp = m_BPerm.data(); // working vector in permuted space

    const double      *__restrict__ AU = m_AU.data();
    const double      *__restrict__ AL = m_AL.data();
    const double      *__restrict__ AD = m_AD.data();
    const long long   *__restrict__ JP = m_JP.data();

    // Forward substitution: solve L y = b (in permuted space).
    for (int j = 0; j < n; ++j) {
        const long long jp_j  = JP[j];
        const int Kj = static_cast<int>(JP[j + 1] - jp_j);
        if (Kj > 0) {
            const int is = j - Kj;
            const double *__restrict__ al_j = AL + jp_j;
            const double *__restrict__ y_in = yp + is;
            double sum = 0.0;
            for (int r = 0; r < Kj; ++r) sum += al_j[r] * y_in[r];
            yp[j] -= sum;
        }
    }

    // Diagonal solve: z = D^-1 y.
    for (int j = 0; j < n; ++j) yp[j] *= AD[j];

    // Backward substitution: solve U x = z. Process columns in
    // descending order so x(j) is final when we subtract its
    // contribution from earlier rows.
    for (int j = n - 1; j > 0; --j) {
        const long long jp_j = JP[j];
        const int Kj = static_cast<int>(JP[j + 1] - jp_j);
        if (Kj > 0) {
            const int is = j - Kj;
            const double xj = yp[j];
            const double *__restrict__ au_j = AU + jp_j;
            double *__restrict__ y_out = yp + is;
            for (int r = 0; r < Kj; ++r) y_out[r] -= au_j[r] * xj;
        }
    }
    // Now yp = m_BPerm contains the solution in permuted space.

    // Permute back: x[old] = y_perm[new] = m_BPerm[m_InvPerm[old]].
    {
        double *__restrict__ xp = x.getDataPtr();
        for (int old_idx = 0; old_idx < n; ++old_idx) {
            xp[old_idx] = m_BPerm[m_InvPerm[old_idx]];
        }
    }
}

void DefaultSolver::initSolver(SparseMatrix &K, nlohmann::ordered_json &Parameters) {
    m_Verbose      = getBoolParameter(Parameters, "verbose", false);
    m_UseOrdering  = getBoolParameter(Parameters, "use_rcm", true);
    // Storage guard threshold in GiB for the skyline arrays (see buildProfile);
    // <= 0 disables the guard. Parsed BEFORE buildProfile so the guard sees it.
    m_MaxProfileGB = getDoubleParameter(Parameters, "max_profile_gb", 16.0);

    const long long profile_size = buildProfile(K);
    m_IsInitialized = true;

    if (m_Verbose) {
        const int n = m_N;
        long long max_K = 0;
        long long sum_K = 0;
        for (int j = 0; j < n; ++j) {
            const long long Kj = m_JP[j + 1] - m_JP[j];
            if (Kj > max_K) max_K = Kj;
            sum_K += Kj;
        }
        const double mean_K = (n > 0) ? (static_cast<double>(sum_K) / n) : 0.0;
        constexpr int bufSize = 160;
        char buffer[bufSize];
        std::snprintf(buffer, bufSize,
                      "      DefaultSolver: n=%d  profile=%lld  mean_K=%.2f  max_K=%lld  ordering=%s",
                      n, profile_size, mean_K, max_K,
                      m_UseOrdering ? "RCM" : "identity");
        MessagePrinter::printNormalTxt(std::string(buffer));
    }
}

bool DefaultSolver::solveLinearSystem(SparseMatrix &A, VectorXd &b, VectorXd &x) {
    if (!m_IsInitialized || !patternMatches(A)) {
        // First call, or sparsity pattern changed: rebuild profile.
        buildProfile(A);
        m_IsInitialized = true;
    }

    if (b.getSize() != m_N || x.getSize() != m_N) {
        MessagePrinter::printErrorTxt(
            "DefaultSolver: size mismatch (A is " + std::to_string(m_N)
            + ", b is " + std::to_string(b.getSize())
            + ", x is " + std::to_string(x.getSize()) + ")");
        return false;
    }

    copyValuesFromCSR(A);
    if (!factorize()) return false;
    solveFactored(b, x);
    return true;
}
