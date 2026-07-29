//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <cstdint>

#include "ICSystem/ICBase.h"

/**
 * Reproducible uniform random perturbation. Every seed, including zero, is a
 * deterministic mt19937_64 seed; this public implementation intentionally has
 * no unseeded/random-device mode.
 */
class RandomIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "random"; }

    void setMin(const double &value)          { m_Min=value; }
    void setMax(const double &value)          { m_Max=value; }
    void setSeed(const std::uint64_t &seed)   { m_Seed=seed; }

    [[nodiscard]] double getMin() const { return m_Min; }
    [[nodiscard]] double getMax() const { return m_Max; }
    [[nodiscard]] std::uint64_t getSeed() const { return m_Seed; }

    void apply(const PDMesh &Mesh,
               const std::vector<int> &NodeIDs,
               const int &DofsPerNode,
               VectorXd &Values) const override;

private:
    double m_Min=0.0;
    double m_Max=1.0;
    std::uint64_t m_Seed=0;
};
