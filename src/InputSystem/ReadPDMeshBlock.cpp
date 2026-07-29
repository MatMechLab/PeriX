//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <cmath>

#include "Mesh/MeshData.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "Utils/MessagePrinter.h"

namespace {
double meshSpacing(const MeshData &data) {
    if (data.CharacteristicLength>0.0) return data.CharacteristicLength;
    if (data.Nx>0) {
        return std::fabs(data.Xmax-data.Xmin)/static_cast<double>(data.Nx);
    }
    return 0.0;
}
}

bool InputSystem::readPDMeshBlock(const nlohmann::ordered_json &json,
                                  const MeshData &meshData,
                                  PDMesh &pdMesh,
                                  PDOperators &operators) {
    const double spacing=meshSpacing(meshData);
    if (!(spacing>0.0) || !std::isfinite(spacing)) {
        MessagePrinter::printErrorTxt("PDMesh: cannot determine a positive point spacing from Mesh");
        return false;
    }
    pdMesh.setHorizonRadiusFromFactor(
        json.at("HorizonRadiusFactor").get<double>(),spacing);
    operators.setOrder(json.at("Order").get<int>());
    if (json.contains("VariableHorizon")) {
        pdMesh.setVariableHorizon(json.at("VariableHorizon").get<bool>());
    }
    if (json.contains("ghost_layer")) {
        pdMesh.setGhostLayer(json.at("ghost_layer").get<int>());
    }
    return true;
}
