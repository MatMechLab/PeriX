//****************************************************************
//* This file is part of the PeriX framework
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "BCSystem/BCSystem.h"
#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/ElmtSystem.h"
#include "MathUtils/VectorXd.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "PDSystem/PDSystem.h"
#include "TimeStepping/TimeStepping.h"

namespace {
class ZeroAccelerationElement final : public ElementBase {
public:
    [[nodiscard]] int getDofsPerNode() const override { return 1; }
    [[nodiscard]] std::string getElementType() const override {
        return "zero_acceleration";
    }
    [[nodiscard]] std::vector<std::string> getDofNames() const override {
        return {"u"};
    }
    [[nodiscard]] bool isExplicit() const override { return true; }
    [[nodiscard]] int getTimeOrder() const override { return 2; }
};

PDMesh makeOneNodeMesh() {
    PDMesh mesh;
    auto &data=mesh.getDataRef();
    data.DX=1.0;
    data.DY=0.0;
    data.DZ=0.0;
    data.NodesNum=1;
    data.BulkElmtsNum=1;
    data.NodesNeighNodesID={{}};
    data.NodesElmtID={1};
    data.NodeCoords={0.0,0.0,0.0};
    data.NodeVolumes={1.0};
    return mesh;
}
}

int main() {
    PDMesh mesh=makeOneNodeMesh();
    PDOperators operators;
    ElmtSystem elements;
    elements.addElement(
        "zero",std::make_unique<ZeroAccelerationElement>());
    BCSystem boundaryConditions;
    PDSystem system;

    VectorXd solution(1,0.0);
    VectorXd oldSolution(1,0.0);
    VectorXd initialVelocity(1,2.0);
    VectorXd acceleration(1,0.0);

    TimeStepping stepping;
    stepping.setDt(0.1);
    stepping.setTotalTime(0.1);
    stepping.setOutputInterval(1);
    stepping.setVerbose(false);

    double finalValue=0.0;
    const auto save=[&](const int &step,const double &time) {
        if (step==1 && std::fabs(time-0.1)<1.0e-15) {
            finalValue=solution(1);
        }
    };

    const bool ok=stepping.solveExplicitDynamics(
        mesh,operators,elements,boundaryConditions,system,1,
        solution,oldSolution,initialVelocity,acceleration,save);

    const double expected=0.2;
    if (!ok || std::fabs(finalValue-expected)>1.0e-14
        || std::fabs(solution(1)-expected)>1.0e-14) {
        std::printf(
            "FAIL initial velocity: ok=%d saved=% .16e solution=% .16e expected=% .16e\n",
            ok?1:0,finalValue,solution(1),expected);
        return 1;
    }

    std::printf("Initial-velocity central-difference test passed\n");
    return 0;
}
