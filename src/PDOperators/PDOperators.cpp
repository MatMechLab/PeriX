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
//+++ Function: PDDO class setup (multi-index list, factorial products,
//+++           operator-name map, workspace allocation) keyed off a
//+++           single Order = max total derivative degree.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDOperators/PDOperators.h"

#include <string>

#include "Utils/MessagePrinter.h"

PDOperators::PDOperators() = default;

int PDOperators::factorial(int n) {
    int r = 1;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

std::string PDOperators::operatorName(int p, int q, int r) {
    const int total = p + q + r;
    std::string name = "d";
    if (total > 1) name += std::to_string(total);
    name += "/";
    if (p > 0) {
        name += "dx";
        if (p > 1) name += std::to_string(p);
    }
    if (q > 0) {
        name += "dy";
        if (q > 1) name += std::to_string(q);
    }
    if (r > 0) {
        name += "dz";
        if (r > 1) name += std::to_string(r);
    }
    return name;
}

void PDOperators::buildMultiIndices() {
    m_QIndices.clear();
    if (m_Dim == 3) {
        // total-order-ascending; within an order p descending, then q
        // descending (r = order-p-q). e.g. order 1: 100,010,001;
        // order 2: 200,110,101,020,011,002.
        for (int order = 1; order <= m_Order; ++order) {
            for (int p = order; p >= 0; --p) {
                for (int q = order - p; q >= 0; --q) {
                    m_QIndices.push_back({p, q, order - p - q});
                }
            }
        }
    }
    else {
        // 2D: r=0, total-order-ascending, within each order decreasing p
        // (matches Matlab subIndex output: [10,01,20,11,02,30,21,12,03,...]).
        for (int order = 1; order <= m_Order; ++order) {
            for (int p = order; p >= 0; --p) {
                m_QIndices.push_back({p, order - p, 0});
            }
        }
    }
    m_OperatorsVecSize = static_cast<int>(m_QIndices.size());
}

void PDOperators::setup() {
    if (m_Order < 1 || m_Order > kMaxOrder) {
        MessagePrinter::printErrorTxt(
            "PDOperators::setup(): Order must be in [1,"
            + std::to_string(kMaxOrder) + "], got "
            + std::to_string(m_Order));
        MessagePrinter::exitPeriX();
    }

    buildMultiIndices();
    if (m_OperatorsVecSize <= 0) {
        MessagePrinter::printErrorTxt(
            "PDOperators::setup(): multi-index list is empty for Order="
            + std::to_string(m_Order));
        MessagePrinter::exitPeriX();
    }

    m_OperatorsVec.resize(m_OperatorsVecSize);
    m_XsiOperatorsVec.resize(m_OperatorsVecSize);
    m_AMATRIX.resize(m_OperatorsVecSize, m_OperatorsVecSize);
    m_Amat.resize(m_OperatorsVecSize, m_OperatorsVecSize);
    m_BDiag.resize(m_OperatorsVecSize, m_OperatorsVecSize);

    m_FactProducts.assign(static_cast<std::size_t>(m_OperatorsVecSize), 1);
    m_OpNameToIndexMap.clear();
    for (int k = 0; k < m_OperatorsVecSize; ++k) {
        const int p = m_QIndices[static_cast<std::size_t>(k)][0];
        const int q = m_QIndices[static_cast<std::size_t>(k)][1];
        const int r = m_QIndices[static_cast<std::size_t>(k)][2];
        m_FactProducts[static_cast<std::size_t>(k)] = factorial(p) * factorial(q) * factorial(r);
        m_OpNameToIndexMap[operatorName(p, q, r)] = k + 1;
        // constant moment-system RHS: A * a_k = b_k e_k  with b_k = prod(q!)
        m_BDiag(k + 1, k + 1) = static_cast<double>(m_FactProducts[static_cast<std::size_t>(k)]);
    }
}
