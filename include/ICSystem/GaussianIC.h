//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <array>
#include <cmath>

#include "ICSystem/ICBase.h"

/** Anisotropic Gaussian profile with scalar broadcast or per-DoF values. */
class GaussianIC final : public ICBase {
public:
    [[nodiscard]] std::string getICType() const override { return "gaussian"; }

    void setCenter(const std::array<double,3> &center)       { m_Center=center; }
    void setSigma(const std::array<double,3> &sigma)         { m_Sigma=sigma; }
    void setAmplitudes(const std::vector<double> &values)    { m_Amplitude=values; }
    void setOffsets(const std::vector<double> &values)       { m_Offset=values; }

    [[nodiscard]] const std::array<double,3>& getCenter() const { return m_Center; }
    [[nodiscard]] const std::array<double,3>& getSigma() const { return m_Sigma; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        const double dx=m_Sigma[0]>0.0 ? (x-m_Center[0])/m_Sigma[0] : 0.0;
        const double dy=m_Sigma[1]>0.0 ? (y-m_Center[1])/m_Sigma[1] : 0.0;
        const double dz=m_Sigma[2]>0.0 ? (z-m_Center[2])/m_Sigma[2] : 0.0;
        const double weight=std::exp(-0.5*(dx*dx+dy*dy+dz*dz));
        return componentValue(m_Offset,component)
             + componentValue(m_Amplitude,component)*weight;
    }

private:
    std::array<double,3> m_Center{0.0,0.0,0.0};
    std::array<double,3> m_Sigma{1.0,1.0,0.0};
    std::vector<double> m_Amplitude{1.0};
    std::vector<double> m_Offset{0.0};
};
