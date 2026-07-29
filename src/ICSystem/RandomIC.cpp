//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "ICSystem/RandomIC.h"

#include <random>

#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

void RandomIC::apply(const PDMesh &Mesh,
                     const std::vector<int> &NodeIDs,
                     const int &DofsPerNode,
                     VectorXd &Values) const {
    validateApplication(Mesh,NodeIDs,DofsPerNode,Values);
    if (!(m_Min<m_Max)) {
        MessagePrinter::printErrorTxt(
            "RandomIC::apply: min must be smaller than max");
        MessagePrinter::exitPeriX();
    }

    std::mt19937_64 generator(m_Seed);
    constexpr double inverseTwoTo53=1.0/9007199254740992.0;
    const double width=m_Max-m_Min;

    // Mapping the top 53 engine bits ourselves makes a seeded field identical
    // across standard-library implementations; uniform_real_distribution does
    // not promise that cross-platform reproducibility.
    const auto sample=[&]() {
        const std::uint64_t bits=generator()>>11U;
        return m_Min+width*(static_cast<double>(bits)*inverseTwoTo53);
    };

    for (const int nodeID : NodeIDs) {
        const int rowBase=(nodeID-1)*DofsPerNode;
        for (const int dof : m_Dofs) {
            Values(rowBase+dof)=sample();
        }
    }
}
