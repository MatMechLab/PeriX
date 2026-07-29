//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include "ICSystem/ICBase.h"

/** Constant profile, with scalar broadcast or one value per target DoF. */
class ConstantIC final : public ICBase {
public:
    ConstantIC()=default;
    explicit ConstantIC(const double &value) : m_Values{value} {}
    explicit ConstantIC(const std::vector<double> &values) : m_Values(values) {}

    [[nodiscard]] std::string getICType() const override { return "constant"; }

    void setValue(const double &value)               { m_Values={value}; }
    void setValues(const std::vector<double> &values){ m_Values=values; }
    [[nodiscard]] const std::vector<double>& getValues() const { return m_Values; }

    [[nodiscard]] double computeValue(const double &x,
                                      const double &y,
                                      const double &z,
                                      const int &component) const override {
        (void)x;
        (void)y;
        (void)z;
        return componentValue(m_Values,component);
    }

private:
    std::vector<double> m_Values{0.0};
};
