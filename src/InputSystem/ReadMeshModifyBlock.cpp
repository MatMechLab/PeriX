//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <array>
#include <string>

#include "MeshModify/MeshModify.h"
#include "MeshModify/MeshModifyGeometry.h"

namespace {

/** Angle of a preset in radians. The schema already guarantees at most one of
 *  'angle_degrees' / 'angle_radians' is present. */
double readPresetAngle(const nlohmann::ordered_json &item,bool &hasAngle) {
    if (item.contains("angle_degrees")) {
        hasAngle=true;
        return item.at("angle_degrees").get<double>()*meshmodify::kPi/180.0;
    }
    if (item.contains("angle_radians")) {
        hasAngle=true;
        return item.at("angle_radians").get<double>();
    }
    hasAngle=false;
    return 0.0;
}

std::array<double,3> readVec3(const nlohmann::ordered_json &item,
                              const std::string &key) {
    const auto &value=item.at(key);
    return {value[0].get<double>(),
            value[1].get<double>(),
            value[2].get<double>()};
}

} // namespace

bool InputSystem::readMeshModifyBlock(const nlohmann::ordered_json &json,
                                      MeshModify &meshModify) {
    meshModify.clear();
    meshModify.setCrackTreatment(MeshModify::CrackTreatment::ForceOnly);

    if (json.contains("Cracks")) {
        for (const auto &crack:json.at("Cracks")) {
            meshModify.addCrackSegment(
                crack.at("x1").get<double>(),
                crack.at("y1").get<double>(),
                crack.at("x2").get<double>(),
                crack.at("y2").get<double>(),
                crack.value("label",std::string{}));
        }
    }

    if (!json.contains("Presets")) return true;

    for (const auto &item:json.at("Presets")) {
        const std::string type=item.at("type").get<std::string>();
        const std::string label=item.value("label",std::string{});
        bool hasAngle=false;
        const double angle=readPresetAngle(item,hasAngle);

        if (type=="center_crack") {
            MeshModify::CenterCrackPreset preset;
            preset.label=label;
            preset.angleRadians=angle;
            if (item.contains("center")) {
                const auto center=readVec3(item,"center");
                preset.centerX=center[0];
                preset.centerY=center[1];
                preset.centerZ=center[2];
                preset.hasCenterZ=true;
            }
            else {
                preset.centerX=item.at("center_x").get<double>();
                preset.centerY=item.at("center_y").get<double>();
                if (item.contains("center_z")) {
                    preset.centerZ=item.at("center_z").get<double>();
                    preset.hasCenterZ=true;
                }
            }
            preset.length=item.at("length").get<double>();
            if (item.contains("width")) {
                preset.width=item.at("width").get<double>();
                preset.hasWidth=true;
            }
            if (item.contains("normal")) {
                preset.normal=readVec3(item,"normal");
                preset.hasNormal=true;
            }
            if (item.contains("axis")) {
                preset.axis=readVec3(item,"axis");
                preset.hasAxis=true;
            }
            meshModify.addCenterCrackPreset(preset);
        }
        else if (type=="edge_crack") {
            MeshModify::EdgeCrackPreset preset;
            preset.label=label;
            preset.side=item.at("side").get<std::string>();
            preset.position=item.at("position").get<double>();
            preset.length=item.at("length").get<double>();
            preset.hasAngle=hasAngle;
            preset.angleRadians=angle;
            if (item.contains("width")) {
                preset.width=item.at("width").get<double>();
                preset.hasWidth=true;
            }
            meshModify.addEdgeCrackPreset(preset);
        }
    }
    return true;
}
