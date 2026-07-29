//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <array>
#include <memory>
#include <vector>

#include "BCSystem/BCSystem.h"
#include "BCSystem/DirichletBC.h"
#include "BCSystem/NeumannBC.h"
#include "BCSystem/PDTractionBC.h"
#include "BCSystem/SpeciesFluxBC.h"

namespace {
std::vector<int> dofSlots(const nlohmann::ordered_json &entry,
                          const DofNameMap &dofs) {
    std::vector<int> slots;
    if (!entry.contains("dofs")) return slots;
    for (const auto &name:entry.at("dofs")) {
        slots.push_back(dofs.indexOf(name.get<std::string>()));
    }
    return slots;
}

std::vector<double> values(const nlohmann::ordered_json &entry) {
    const auto &value=entry.at("value");
    if (value.is_number()) return {value.get<double>()};
    return value.get<std::vector<double>>();
}
}

bool InputSystem::readBCSystemBlock(const nlohmann::ordered_json &json,
                                    BCSystem &boundaryConditions,
                                    const DofNameMap &dofs,
                                    const int meshDim) {
    for (auto it=json.begin();it!=json.end();++it) {
        const std::string name=it.key();
        const auto &entry=it.value();
        const std::string type=entry.at("type").get<std::string>();
        const std::string phygroup=entry.at("phygroup").get<std::string>();
        std::unique_ptr<BCBase> condition;

        if (type=="dirichlet") {
            auto built=std::make_unique<DirichletBC>(values(entry));
            built->setComponents(dofSlots(entry,dofs));
            condition=std::move(built);
        }
        else if (type=="neumann") {
            auto built=std::make_unique<NeumannBC>();
            built->setComponents(dofSlots(entry,dofs));
            condition=std::move(built);
        }
        else if (type=="speciesflux") {
            auto built=std::make_unique<SpeciesFluxBC>(values(entry));
            built->setComponents(dofSlots(entry,dofs));
            condition=std::move(built);
        }
        else if (type=="pdtraction") {
            const auto tractionValues=entry.at("value").get<std::vector<double>>();
            std::array<double,3> traction{0.0,0.0,0.0};
            for (std::size_t i=0;i<tractionValues.size();++i) {
                traction[i]=tractionValues[i];
            }
            auto state=PDTractionBC::StressState::PlaneStress;
            if (meshDim==2 && entry.at("state").get<std::string>()=="plane_strain") {
                state=PDTractionBC::StressState::PlaneStrain;
            }
            auto built=std::make_unique<PDTractionBC>(
                entry.at("E").get<double>(),entry.at("nu").get<double>(),
                state,traction);
            built->setDisplacementComponents(dofSlots(entry,dofs));
            if (entry.contains("Omega")) {
                built->setChemicalExpansion(
                    entry.at("Omega").get<double>(),
                    entry.at("cref").get<double>(),
                    dofs.indexOf(entry.at("c_dof").get<std::string>()));
            }
            condition=std::move(built);
        }

        if (!condition) return false;
        boundaryConditions.addBC(name,phygroup,std::move(condition));
    }
    return true;
}
