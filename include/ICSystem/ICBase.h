//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>
#include <vector>

#include "MathUtils/VectorXd.h"

class PDMesh;

/**
 * Base class for one closed-form initial-condition profile.
 *
 * DoF indices are 1-based, consistently with VectorXd and the rest of PeriX.
 * Most profiles only implement computeValue(); the default apply() evaluates
 * that function at every selected PD node. Stateful profiles may override
 * apply() directly.
 */
class ICBase {
public:
    ICBase()=default;
    virtual ~ICBase()=default;

    [[nodiscard]] virtual std::string getICType() const=0;

    void setDofs(const std::vector<int> &dofs)            { m_Dofs=dofs; }
    [[nodiscard]] const std::vector<int>& getDofs() const { return m_Dofs; }

    virtual void apply(const PDMesh &Mesh,
                       const std::vector<int> &NodeIDs,
                       const int &DofsPerNode,
                       VectorXd &Values) const;

    [[nodiscard]] virtual double computeValue(const double &x,
                                              const double &y,
                                              const double &z,
                                              const int &component) const {
        (void)x;
        (void)y;
        (void)z;
        (void)component;
        return 0.0;
    }

protected:
    void validateApplication(const PDMesh &Mesh,
                             const std::vector<int> &NodeIDs,
                             const int &DofsPerNode,
                             const VectorXd &Values) const;

    [[nodiscard]] static double componentValue(const std::vector<double> &values,
                                               const int &component,
                                               const double &emptyValue=0.0) {
        if (values.empty()) return emptyValue;
        if (values.size()==1) return values.front();
        return values.at(static_cast<std::size_t>(component));
    }

    std::vector<int> m_Dofs{1};
};
