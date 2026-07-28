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
//+++ Function: PDDO (Peridynamic Differential Operator) class.
//+++           For a source node i the moment matrix
//+++             A(a,b) = sum_{j in H(i)} w * vc * V_j * X^{q_a+q_b}
//+++           is solved column-by-column against diag(b_a) where
//+++             b_a = prod_k factorial(q_a[k])
//+++           giving the per-bond operator values
//+++             ops_k(i,j) = ( sum_a a_mat[a,k] * X^{q_a} )
//+++                          * w * vc * V_j
//+++           so that the discrete derivative reads
//+++             d^|q_k| u / dx^{q_k} | i = sum_j ops_k(i,j) * (u_j-u_i)
//+++           i.e. V_j is baked into the operator value (matching
//+++           the reference Matlab `g10 = gf .* wfvj` convention).
//+++
//+++           A single integer Order=N selects every multi-index:
//+++           (p,q) in 2D with 1 <= p+q <= N, or (p,q,r) in 3D with
//+++           1 <= p+q+r <= N. The dimension is set with setDim(2|3)
//+++           before setup(); 2D is the default and reproduces the 2D
//+++           multi-index order bit-for-bit (the third exponent is 0).
//+++           The class therefore covers gradient (N=1), Laplacian /
//+++           Navier-Cauchy (N=2) and arbitrary higher-order PDE kernels
//+++           up to N=kMaxOrder, in both 2D and 3D.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "MathUtils/MatrixXd.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMeshData.h"
#include "Utils/MessagePrinter.h"

class PDOperators {
public:
    /** maximum total derivative order supported by setup().
     *  The cap is set by what the bond-normalized moment matrix can
     *  reproduce to better than 1e-6 on a regular grid with a
     *  reasonable horizon (verified by the pddo_benchmark). */
    static constexpr int kMaxOrder = 6;

    PDOperators();

    void setOrder(const int &order)        { m_Order = order; }
    [[nodiscard]] int getOrder() const     { return m_Order; }

    /** spatial dimension of the operator basis (2 or 3). Must be set
     *  before setup(); default 2. In 2D every multi-index has r=0 and
     *  the z monomial is identically 1, so the 2D path is unchanged. */
    void setDim(const int &dim)            { m_Dim = dim; }
    [[nodiscard]] int getDim() const       { return m_Dim; }

    /**
     * build the multi-index list and (re)allocate all per-call workspace
     * for the requested Order. Must be called after the order has been
     * set and before any calcAMatrix / calcOperators call.
     */
    void setup();

    /**
     * look up an operator value by name. Names follow the convention
     *   "d/dx", "d/dy", "d/dz", "d2/dx2", "d2/dxdy", "d2/dxdz",
     *   "d2/dy2", "d2/dydz", "d2/dz2", "d3/dx3", ...
     * Every multi-index (p,q,r) with 1 <= p+q+r <= Order is populated
     * (r is always 0 in 2D). Querying an unsupported name aborts.
     */
    double operator()(const std::string &op) const {
        const auto it = m_OpNameToIndexMap.find(op);
        if (it == m_OpNameToIndexMap.end()) {
            // The overwhelmingly common cause is a PDMesh "Order" too low for
            // the kernel in use: a second-derivative operator simply is not
            // generated at Order 1, and the kernel only discovers that here,
            // deep inside the bond loop. Say so, with the derivative order the
            // name implies, instead of just reporting a missing key.
            const std::size_t d=(op.rfind("d2/",0)==0)?2:((op.rfind("d3/",0)==0)?3:1);
            MessagePrinter::printErrorTxt("op="+op+" is invalid or not found in current operators list "
                "(PDMesh Order="+std::to_string(m_Order)+" generates derivatives up to order "
                +std::to_string(m_Order)+", but this operator is order "+std::to_string(d)
                +"). If the name is spelled correctly, raise \"PDMesh\":{\"Order\":"
                +std::to_string(d)+"} -- the element in use needs it.");
            MessagePrinter::exitPeriX();
        }
        return m_OperatorsVec(it->second);
    }

    [[nodiscard]] bool hasOperator(const std::string &op) const {
        return m_OpNameToIndexMap.find(op) != m_OpNameToIndexMap.end();
    }

    /**
     * resolve an operator name to its slot once, then read by index inside
     * hot per-bond loops via getOperatorByIndex -- same doubles as
     * operator(), without the per-call string hash. Returns -1 when the
     * name is not populated (caller decides whether that is an error).
     * The name->index map is fixed at setup, so a resolved index stays
     * valid for the whole run (and across PDOperators copies).
     */
    [[nodiscard]] int getOperatorIndex(const std::string &op) const {
        const auto it=m_OpNameToIndexMap.find(op);
        return (it==m_OpNameToIndexMap.end())?-1:it->second;
    }
    [[nodiscard]] double getOperatorByIndex(const int &idx) const {
        return m_OperatorsVec(idx);
    }

    /**
     * build (factor) the per-node moment matrix at NodeI. Must be
     * called once per source node before any calcOperators(i,j,...)
     * for that node.
     *
     * Aborts with a clear message when the node's moment system is
     * DEGENERATE -- fewer usable bonds than monomials, or an inconsistent
     * system (the family cannot meet the orthogonality constraints, e.g. a
     * crack-isolated point or a boundary ghost with a too-small horizon) --
     * instead of silently producing garbage operators. A rank-deficient but
     * CONSISTENT system (symmetric family at high order) is fine and passes.
     */
    void calcAMatrix(const int &NodeI, const PDMeshData &Data);

    /** disable the hard abort on a degenerate moment system (default ON).
     *  Only meant for diagnostic sweeps (pddo_benchmark probes degenerate
     *  order/horizon combos on purpose and reports them as FAIL rows);
     *  production solves should keep the abort. */
    void setAbortOnDegenerate(const bool &flag) { m_AbortOnDegenerate = flag; }

    /**
     * Crack-respecting operator mode (default OFF). When ON, calcAMatrix and
     * calcOperators skip every bond severed by a pre-existing crack
     * (initialCrackCutsBond), making the operators strictly ONE-SIDED at an
     * initial notch even when the family still spans it (force-only crack
     * treatment). The two faces of a preset notch are free surfaces to each
     * other, so anything that imposes a PHYSICAL condition through the
     * operators -- the pdtraction sigma.n rows in particular -- must not
     * couple across the slit: without this gate a traction-free row at a
     * ghost just outside a notch mouth drags that flank with the opposite
     * side's motion (the Kalthoff-Winkler impactor!), nucleating spurious
     * cracks along the outer flank. The fracture KERNELS deliberately leave
     * the mode OFF: the force-only smeared operator near the notch root is
     * the Matlab-reference behaviour that drives propagation, and severed
     * bonds carry no force there anyway (health=0). With
     * treatment=delete_crossing_bonds the gate is a no-op (those bonds are
     * not in the family at all).
     */
    void setRespectInitialCracks(const bool &flag) { m_RespectInitialCracks = flag; }
    [[nodiscard]] bool getRespectInitialCracks() const { return m_RespectInitialCracks; }

    /**
     * evaluate the per-bond operator values for bond (NodeI -> NodeJ).
     * calcAMatrix(NodeI, ...) must have been called for the current
     * source node first. The values include the bond weight, the
     * partial-volume correction vc, and V_j (volume of NodeJ).
     *
     * noinline: the geometry-cache builder (same translation unit) must call
     * the EXACT machine code the assembly drivers call — inlining this into
     * the build loop would let the optimizer contract the FP arithmetic
     * differently and store epsilon-different operator values, breaking the
     * bitwise identity between the replayed and the live-computed operators.
     */
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    void calcOperators(const int &NodeI, const int &NodeJ, const PDMeshData &Data);

    [[nodiscard]] int getOperatorsVecSize() const { return m_OperatorsVecSize; }

    /**
     * multi-index list q in the same order as the operator vector.
     * Index k (0-based) corresponds to operator slot k+1.
     */
    [[nodiscard]] const std::vector<std::array<int,3>>& getQIndices() const { return m_QIndices; }

    /** canonical operator name for multi-index (p,q,r):
     *  (1,0,0)->"d/dx", (0,0,1)->"d/dz", (2,1,0)->"d3/dx2dy",
     *  (1,0,1)->"d2/dxdz", etc. r defaults to 0 so existing 2D callers
     *  keep working. Exposed so external callers (tests, benchmarks) can
     *  construct names without duplicating the convention. */
    static std::string operatorName(int p, int q, int r = 0);

    /** overwrite the whole per-bond operator vector with externally cached
     *  values (length getOperatorsVecSize(), 0-based). The matrix-free explicit
     *  rate path uses this to REPLAY operators that were computed once: the PDDO
     *  operators depend only on the (fixed) reference geometry, so for a
     *  transient explicit run they are constant across time steps and need not
     *  be re-solved every step. After this call operator()/getOperatorValue
     *  return the cached values; calcAMatrix is not required. The implicit /
     *  serial assembly paths do not use it (they re-solve every Newton step). */
    void loadOperatorValues(const double *vals) {
        for (int k = 1; k <= m_OperatorsVecSize; k++) m_OperatorsVec(k) = vals[k-1];
    }

    /**
     * Build the shared geometry-operator cache: the per-bond operator vector
     * op_k(i,j) for every node i and every j in its (fixed) family, computed
     * ONCE with the exact calcAMatrix/calcOperators code below and replayed by
     * every later call. This is legal because the operators depend only on the
     * reference geometry (coordinates, volumes, horizon, volume correction,
     * preset cracks), all of which are frozen once the PD mesh exists — the
     * same fact the matrix-free explicit path already exploits through
     * loadOperatorValues(). Replayed values are BITWISE identical to a live
     * recomputation (they ARE that computation, stored).
     *
     * Two variants are stored when the mesh carries preset cracks: the plain
     * assembly operators and the crack-respecting ones (the strong-form
     * traction/flux BC rows switch setRespectInitialCracks(true) per call).
     * Without presets the two are identical and one array serves both. A
     * flagged family that is DEGENERATE is marked invalid instead of aborting
     * at build time (production may never evaluate it flagged); a replay
     * request for it falls back to the live computation, which preserves the
     * existing abort semantics exactly.
     *
     * Copies of this instance share the (immutable) cache. The estimated
     * footprint is checked against MaxGB first; an oversized cache is skipped
     * with a note and every call keeps the live-recompute path.
     */
    void buildGeometryCache(const PDMeshData &Data, const double &MaxGB);
    [[nodiscard]] bool hasGeometryCache() const { return m_Cache!=nullptr; }

    /** replay-verification counters (PERIX_OPCACHE_VERIFY=1): number of
     *  replayed bonds bit-compared against a live recomputation, and how many
     *  mismatched (expected: zero). */
    static long long getVerifyChecked();
    static long long getVerifyMismatch();

    /** look up an operator value by its 1-based slot index in the
     *  multi-index list returned by getQIndices(). Useful for tight
     *  per-bond loops that already know the slot. */
    [[nodiscard]] double getOperatorValue(const int &slot_one_based) const {
        if (slot_one_based < 1 || slot_one_based > m_OperatorsVecSize) {
            MessagePrinter::printErrorTxt("PDOperators::getOperatorValue: slot="
                                          +std::to_string(slot_one_based)
                                          +" is out of [1,"+std::to_string(m_OperatorsVecSize)+"]");
            MessagePrinter::exitPeriX();
        }
        return m_OperatorsVec(slot_one_based);
    }

private:
    /** generate the multi-indices with 1 <= |q| <= Order, sorted by total
     *  order then by decreasing p (then decreasing q). In 2D r is always 0
     *  and the 2D ordering is reproduced exactly. */
    void buildMultiIndices();

    /** populate m_XsiOperatorsVec with the bond monomials X^{q_k}
     *  for the current (xsi_x, xsi_y, xsi_z). In 2D xsi_z is 0 and every
     *  r is 0, so the z factor is identically 1. */
    void computeXsiMonomials(const double &xsi_x, const double &xsi_y, const double &xsi_z);

    /** factorial(n) — n is bounded by kMaxOrder (small). */
    static int factorial(int n);

private:
    /** immutable geometry-operator cache shared by all copies (see
     *  buildGeometryCache). ops holds the plain per-bond operator vectors in
     *  family order; opsCracked the crack-respecting variant (empty when the
     *  mesh has no preset cracks — the plain array then serves both modes);
     *  crackedValid marks flagged families that built cleanly (a degenerate
     *  one is replayed by live recomputation instead). */
    struct GeometryOpCache {
        std::vector<long long> nodeOff;   /**< N+1 prefix offsets (bond index) */
        std::vector<double>    ops;       /**< [bond*nop + k], plain mode */
        std::vector<double>    opsCracked;/**< [bond*nop + k], respect-cracks mode */
        std::vector<char>      crackedValid;/**< per node, flagged family non-degenerate */
        bool hasCrackVariant=false;
    };

    /** locate NodeJ inside NodeI's family (O(1) for in-order replay via the
     *  cursor, linear scan otherwise); -1 if not a family member. */
    int cacheFamilyPos(const int &NodeI,const int &NodeJ,const PDMeshData &Data);

    /** the full moment-matrix build + solve (the pre-cache calcAMatrix body);
     *  the public calcAMatrix dispatches here whenever the replay cache does
     *  not apply. Sets m_LastFamilyDegenerate. noinline for the same bitwise
     *  build-equals-live guarantee as calcOperatorsCompute. */
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    void calcAMatrixCompute(const int &NodeI, const PDMeshData &Data);

    /** the live per-bond operator evaluation (the pre-cache calcOperators
     *  body); called by the public calcOperators when the cache does not
     *  apply, by the cache builder, and by the PERIX_OPCACHE_VERIFY bit
     *  comparison. noinline so every caller uses the same machine code. */
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    void calcOperatorsCompute(const int &NodeI, const int &NodeJ, const PDMeshData &Data);

    /** true when the replay cache serves the CURRENT mode (plain, or flagged
     *  with a valid stored variant) for this node. */
    [[nodiscard]] bool cacheServes(const int &NodeI) const {
        if (!m_Cache) return false;
        if (!m_RespectInitialCracks) return true;
        if (!m_Cache->hasCrackVariant) return true;   // no preset cracks: identical
        return NodeI>=1 && NodeI<=static_cast<int>(m_Cache->crackedValid.size())
               && m_Cache->crackedValid[static_cast<std::size_t>(NodeI-1)]!=0;
    }

    int m_Order = -1;
    int m_Dim   = 2;   /**< spatial dimension of the basis (2 or 3) */
    bool m_AbortOnDegenerate = true; /**< hard-abort on a degenerate moment system */
    bool m_RespectInitialCracks = false; /**< skip preset-crack bonds in the operators */
    bool m_LastFamilyDegenerate = false; /**< set by calcAMatrixCompute (cache builder reads it) */

    std::vector<std::array<int,3>> m_QIndices; /**< multi-indices (p,q,r), 0-based */
    std::vector<int> m_FactProducts;           /**< prod(factorial(q_i)) per index */
    std::vector<double> m_OpInvDeltaPow;       /**< 1/delta^|q_k| per slot, set by calcAMatrix */

    int m_OperatorsVecSize = 0;
    VectorXd m_OperatorsVec;        /**< per-bond operator values (1-based) */
    VectorXd m_XsiOperatorsVec;     /**< bond monomials X^{q_k} (1-based) */

    MatrixXd m_AMATRIX;             /**< moment matrix at the current source node */
    MatrixXd m_Amat;                /**< A^{-1} * diag(b), shape NxN */
    MatrixXd m_BDiag;               /**< constant RHS diag(b_k), b_k = prod(factorial(q_k)) */

    std::shared_ptr<const GeometryOpCache> m_Cache; /**< shared replay cache (may be null) */
    int m_CursorNode=0;             /**< family cursor: node of the last replayed bond */
    int m_CursorPos =0;             /**< family cursor: next expected family position */

    std::unordered_map<std::string, int> m_OpNameToIndexMap;
};
