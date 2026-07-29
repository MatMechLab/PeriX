//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <cmath>

#include "ICSystem/ICBase.h"

/**
 * Separable cosine profile:
 *   offset + amplitude*product_i cos(wavenumber_i*x_i + phase_i).
 */
class CosineIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "cosine"; }

    void setWaveNumbers(const std::array<double,3> &values) { m_WaveNumbers=values; }
    void setPhases(const std::array<double,3> &values)      { m_Phases=values; }
    void setAmplitudes(const std::vector<double> &values)   { m_Amplitude=values; }
    void setOffsets(const std::vector<double> &values)      { m_Offset=values; }

    [[nodiscard]] const std::array<double,3>& getWaveNumbers() const {
        return m_WaveNumbers;
    }
    [[nodiscard]] const std::array<double,3>& getPhases() const { return m_Phases; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        const double mode=std::cos(m_WaveNumbers[0]*x+m_Phases[0])
                         *std::cos(m_WaveNumbers[1]*y+m_Phases[1])
                         *std::cos(m_WaveNumbers[2]*z+m_Phases[2]);
        return componentValue(m_Offset,component)
             + componentValue(m_Amplitude,component)*mode;
    }

private:
    std::array<double,3> m_WaveNumbers{0.0,0.0,0.0};
    std::array<double,3> m_Phases{0.0,0.0,0.0};
    std::vector<double> m_Amplitude{1.0};
    std::vector<double> m_Offset{0.0};
};
