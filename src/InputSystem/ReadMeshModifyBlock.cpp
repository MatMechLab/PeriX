//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include "MeshModify/MeshModify.h"

bool InputSystem::readMeshModifyBlock(const nlohmann::ordered_json &json,
                                      MeshModify &meshModify) {
    meshModify.clear();
    meshModify.setCrackTreatment(MeshModify::CrackTreatment::ForceOnly);

    for (const auto &crack:json.at("Cracks")) {
        meshModify.addCrackSegment(
            crack.at("x1").get<double>(),
            crack.at("y1").get<double>(),
            crack.at("x2").get<double>(),
            crack.at("y2").get<double>(),
            crack.value("label",std::string{}));
    }
    return true;
}
