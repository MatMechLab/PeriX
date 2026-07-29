//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <cmath>

#include "ICSystem/ICBase.h"

/**
 * Circle in 2D or sphere in 3D. dr=0 gives a hard step; dr>0 blends from the
 * inside value at radius to the outside value at radius+dr with a C1 cubic
 * smoothstep.
 */
class CircleIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "circle"; }

    void setCenter(const std::array<double,3> &center)       { m_Center=center; }
    void setRadius(const double &radius)                     { m_Radius=radius; }
    void setTransitionThickness(const double &thickness)     { m_Dr=thickness; }
    void setInsideValues(const std::vector<double> &values)  { m_Inside=values; }
    void setOutsideValues(const std::vector<double> &values) { m_Outside=values; }

    [[nodiscard]] const std::array<double,3>& getCenter() const { return m_Center; }
    [[nodiscard]] double getRadius() const { return m_Radius; }
    [[nodiscard]] double getTransitionThickness() const { return m_Dr; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        const double dx=x-m_Center[0];
        const double dy=y-m_Center[1];
        const double dz=z-m_Center[2];
        const double distance=std::sqrt(dx*dx+dy*dy+dz*dz);

        double weight=0.0;
        if (distance<=m_Radius) {
            weight=1.0;
        }
        else if (m_Dr>0.0 && distance<m_Radius+m_Dr) {
            const double t=(distance-m_Radius)/m_Dr;
            weight=1.0-t*t*(3.0-2.0*t);
        }

        const double inside=componentValue(m_Inside,component);
        const double outside=componentValue(m_Outside,component);
        return outside+weight*(inside-outside);
    }

private:
    std::array<double,3> m_Center{0.0,0.0,0.0};
    double m_Radius=1.0;
    double m_Dr=0.0;
    std::vector<double> m_Inside{1.0};
    std::vector<double> m_Outside{0.0};
};
