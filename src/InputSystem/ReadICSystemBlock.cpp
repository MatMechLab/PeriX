//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "ICSystem/BoxIC.h"
#include "ICSystem/CircleIC.h"
#include "ICSystem/ConstantIC.h"
#include "ICSystem/CosineIC.h"
#include "ICSystem/EllipseIC.h"
#include "ICSystem/GaussianIC.h"
#include "ICSystem/ICSystem.h"
#include "ICSystem/LinearIC.h"
#include "ICSystem/RandomIC.h"

namespace {
std::array<double,3> vector3(const nlohmann::ordered_json &value) {
    std::array<double,3> result{0.0,0.0,0.0};
    for (std::size_t i=0;i<value.size();++i) result[i]=value[i].get<double>();
    return result;
}

std::vector<double> profileValues(const nlohmann::ordered_json &entry,
                                  const char *key,
                                  const std::vector<double> &fallback={}) {
    if (!entry.contains(key)) return fallback;
    const auto &value=entry.at(key);
    if (value.is_number()) return {value.get<double>()};
    return value.get<std::vector<double>>();
}

std::vector<int> icDofSlots(const nlohmann::ordered_json &entry,
                            const DofNameMap &dofs) {
    if (entry.contains("dof")) {
        return {dofs.indexOf(entry.at("dof").get<std::string>())};
    }
    std::vector<int> slots;
    for (const auto &name:entry.at("dofs")) {
        slots.push_back(dofs.indexOf(name.get<std::string>()));
    }
    return slots;
}
}

bool InputSystem::readICSystemBlock(const nlohmann::ordered_json &json,
                                    ICSystem &initialConditions,
                                    const DofNameMap &dofs) {
    for (auto it=json.begin();it!=json.end();++it) {
        const std::string name=it.key();
        const auto &entry=it.value();
        const std::string type=entry.at("type").get<std::string>();
        const std::vector<int> slots=icDofSlots(entry,dofs);
        std::unique_ptr<ICBase> condition;

        if (type=="constant") {
            condition=std::make_unique<ConstantIC>(
                profileValues(entry,"value"));
        }
        else if (type=="linear") {
            auto built=std::make_unique<LinearIC>();
            built->setOffsets(profileValues(entry,"offset"));
            built->setSlopeX(profileValues(entry,"slope_x",{0.0}));
            built->setSlopeY(profileValues(entry,"slope_y",{0.0}));
            built->setSlopeZ(profileValues(entry,"slope_z",{0.0}));
            condition=std::move(built);
        }
        else if (type=="box") {
            auto built=std::make_unique<BoxIC>();
            built->setMinCorner(vector3(entry.at("min_corner")));
            built->setMaxCorner(vector3(entry.at("max_corner")));
            built->setInsideValues(profileValues(entry,"inside_values"));
            built->setOutsideValues(profileValues(entry,"outside_values"));
            condition=std::move(built);
        }
        else if (type=="circle") {
            auto built=std::make_unique<CircleIC>();
            built->setCenter(vector3(entry.at("center")));
            built->setRadius(entry.at("radius").get<double>());
            built->setTransitionThickness(entry.at("dr").get<double>());
            built->setInsideValues(profileValues(entry,"inside_values"));
            built->setOutsideValues(profileValues(entry,"outside_values"));
            condition=std::move(built);
        }
        else if (type=="ellipse") {
            auto built=std::make_unique<EllipseIC>();
            built->setCenter(vector3(entry.at("center")));
            built->setSemiAxes(vector3(entry.at("semi_axes")));
            built->setInsideValues(profileValues(entry,"inside_values"));
            built->setOutsideValues(profileValues(entry,"outside_values"));
            built->setSmoothness(entry.value("smoothness",0.0));
            condition=std::move(built);
        }
        else if (type=="gaussian") {
            auto built=std::make_unique<GaussianIC>();
            built->setCenter(vector3(entry.at("center")));
            built->setSigma(vector3(entry.at("sigma")));
            built->setAmplitudes(profileValues(entry,"amplitude"));
            built->setOffsets(profileValues(entry,"offset",{0.0}));
            condition=std::move(built);
        }
        else if (type=="cosine") {
            auto built=std::make_unique<CosineIC>();
            built->setWaveNumbers(vector3(entry.at("wavenumber")));
            built->setPhases(entry.contains("phase")
                ? vector3(entry.at("phase"))
                : std::array<double,3>{0.0,0.0,0.0});
            built->setAmplitudes(profileValues(entry,"amplitude"));
            built->setOffsets(profileValues(entry,"offset",{0.0}));
            condition=std::move(built);
        }
        else if (type=="random") {
            auto built=std::make_unique<RandomIC>();
            built->setMin(entry.at("min").get<double>());
            built->setMax(entry.at("max").get<double>());
            built->setSeed(entry.at("seed").get<std::uint64_t>());
            condition=std::move(built);
        }

        if (!condition) return false;
        condition->setDofs(slots);
        const std::string phygroup=entry.value("phygroup",std::string{});
        const ICField field=entry.value("field",std::string("solution"))=="velocity"
            ? ICField::Velocity : ICField::Solution;
        initialConditions.addIC(name,phygroup,std::move(condition),field);
    }
    return true;
}
