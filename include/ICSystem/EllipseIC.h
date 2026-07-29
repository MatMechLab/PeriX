//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <cmath>

#include "ICSystem/ICBase.h"

/**
 * Ellipse in 2D or ellipsoid in 3D. A non-positive semi-axis disables that
 * coordinate. smoothness=0 gives a hard step; positive values use a tanh
 * transition in normalized radial distance.
 */
class EllipseIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "ellipse"; }

    void setCenter(const std::array<double,3> &center)       { m_Center=center; }
    void setSemiAxes(const std::array<double,3> &axes)       { m_SemiAxes=axes; }
    void setInsideValues(const std::vector<double> &values)  { m_Inside=values; }
    void setOutsideValues(const std::vector<double> &values) { m_Outside=values; }
    void setSmoothness(const double &smoothness)              { m_Smoothness=smoothness; }

    [[nodiscard]] const std::array<double,3>& getCenter() const { return m_Center; }
    [[nodiscard]] const std::array<double,3>& getSemiAxes() const { return m_SemiAxes; }
    [[nodiscard]] double getSmoothness() const { return m_Smoothness; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        const double ex=m_SemiAxes[0]>0.0 ? (x-m_Center[0])/m_SemiAxes[0] : 0.0;
        const double ey=m_SemiAxes[1]>0.0 ? (y-m_Center[1])/m_SemiAxes[1] : 0.0;
        const double ez=m_SemiAxes[2]>0.0 ? (z-m_Center[2])/m_SemiAxes[2] : 0.0;
        const double level=ex*ex+ey*ey+ez*ez-1.0;
        const double weight=m_Smoothness>0.0
                          ? 0.5*(1.0-std::tanh(m_Smoothness*level))
                          : (level<=0.0 ? 1.0 : 0.0);
        const double inside=componentValue(m_Inside,component);
        const double outside=componentValue(m_Outside,component);
        return outside+weight*(inside-outside);
    }

private:
    std::array<double,3> m_Center{0.0,0.0,0.0};
    std::array<double,3> m_SemiAxes{1.0,1.0,0.0};
    std::vector<double> m_Inside{1.0};
    std::vector<double> m_Outside{0.0};
    double m_Smoothness=0.0;
};
