//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include "ICSystem/ICBase.h"

/**
 * Linear profile:
 *   value = offset + slope_x*x + slope_y*y + slope_z*z.
 */
class LinearIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "linear"; }

    void setOffsets(const std::vector<double> &values) { m_Offsets=values; }
    void setSlopeX(const std::vector<double> &values)  { m_SlopeX=values; }
    void setSlopeY(const std::vector<double> &values)  { m_SlopeY=values; }
    void setSlopeZ(const std::vector<double> &values)  { m_SlopeZ=values; }

    [[nodiscard]] const std::vector<double>& getOffsets() const { return m_Offsets; }
    [[nodiscard]] const std::vector<double>& getSlopeX() const  { return m_SlopeX; }
    [[nodiscard]] const std::vector<double>& getSlopeY() const  { return m_SlopeY; }
    [[nodiscard]] const std::vector<double>& getSlopeZ() const  { return m_SlopeZ; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        return componentValue(m_Offsets,component)
             + componentValue(m_SlopeX,component)*x
             + componentValue(m_SlopeY,component)*y
             + componentValue(m_SlopeZ,component)*z;
    }

private:
    std::vector<double> m_Offsets{0.0};
    std::vector<double> m_SlopeX{0.0};
    std::vector<double> m_SlopeY{0.0};
    std::vector<double> m_SlopeZ{0.0};
};
