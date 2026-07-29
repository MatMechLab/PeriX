//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>

#include "ICSystem/ICBase.h"

/**
 * Axis-aligned box profile. A degenerate range (max <= min) disables that
 * coordinate, which lets a two-component corner describe a 2D box.
 */
class BoxIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "box"; }

    void setMinCorner(const std::array<double,3> &corner)    { m_MinCorner=corner; }
    void setMaxCorner(const std::array<double,3> &corner)    { m_MaxCorner=corner; }
    void setInsideValues(const std::vector<double> &values)  { m_Inside=values; }
    void setOutsideValues(const std::vector<double> &values) { m_Outside=values; }

    [[nodiscard]] const std::array<double,3>& getMinCorner() const { return m_MinCorner; }
    [[nodiscard]] const std::array<double,3>& getMaxCorner() const { return m_MaxCorner; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        const auto inRange=[](const double &value,
                              const double &minimum,
                              const double &maximum) {
            return maximum<=minimum
                 || (value>=minimum && value<=maximum);
        };
        const bool inside=inRange(x,m_MinCorner[0],m_MaxCorner[0])
                       && inRange(y,m_MinCorner[1],m_MaxCorner[1])
                       && inRange(z,m_MinCorner[2],m_MaxCorner[2]);
        return inside ? componentValue(m_Inside,component)
                      : componentValue(m_Outside,component);
    }

private:
    std::array<double,3> m_MinCorner{0.0,0.0,0.0};
    std::array<double,3> m_MaxCorner{0.0,0.0,0.0};
    std::vector<double> m_Inside{1.0};
    std::vector<double> m_Outside{0.0};
};
