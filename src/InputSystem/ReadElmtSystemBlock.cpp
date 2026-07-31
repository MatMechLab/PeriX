//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <memory>

#include "ElmtSystem/CahnHilliardElement.h"
#include "ElmtSystem/DiffusionElement.h"
#include "ElmtSystem/ElmtSystem.h"
#include "ElmtSystem/ExplicitPDDOFracElement.h"
#include "ElmtSystem/FracStressCahnHilliardElement.h"
#include "ElmtSystem/PDDODynamicFracElement.h"
#include "ElmtSystem/PoissonElement.h"
#include "ElmtSystem/StressCahnHilliardElement.h"

bool InputSystem::readElmtSystemBlock(const nlohmann::ordered_json &json,
                                      ElmtSystem &elements,
                                      const int meshDim) {
    const auto it=json.begin();
    const std::string name=it.key();
    const auto &entry=it.value();
    const std::string type=entry.at("type").get<std::string>();
    const auto &params=entry.at("params");
    std::unique_ptr<ElementBase> element;

    if (type=="poisson") {
        element=std::make_unique<PoissonElement>(
            params.at("sigma").get<double>(),params.at("f").get<double>());
    }
    else if (type=="diffusion") {
        element=std::make_unique<DiffusionElement>(
            params.at("D").get<double>(),params.at("f").get<double>());
    }
    else if (type=="cahnhilliard") {
        auto built=std::make_unique<CahnHilliardElement>();
        built->setChi(params.at("chi").get<double>());
        built->setKappa(params.at("kappa").get<double>());
        built->setMobility(params.at("M").get<double>());
        element=std::move(built);
    }
    else if (type=="pddo_dynamic_frac") {
        auto built=std::make_unique<PDDODynamicFracElement>();
        built->setE(params.at("E").get<double>());
        built->setNu(params.at("nu").get<double>());
        built->setRho(params.at("rho").get<double>());
        built->setG0(params.at("Gc").get<double>());
        built->setTensionOnly(params.at("tension_only").get<bool>());
        built->setDamageOn(params.at("damage").get<bool>());
        if (meshDim==2) {
            built->setState(
                params.at("state").get<std::string>()=="plane_stress"
                    ? PDDODynamicFracElement::StressState::PlaneStress
                    : PDDODynamicFracElement::StressState::PlaneStrain);
        }
        element=std::move(built);
    }
    else if (type=="explicit_pddo_frac") {
        auto built=std::make_unique<ExplicitPDDOFracElement>();
        built->setE(params.at("E").get<double>());
        built->setNu(params.at("nu").get<double>());
        built->setRho(params.at("rho").get<double>());
        built->setG0(params.at("Gc").get<double>());
        built->setTensionOnly(params.at("tension_only").get<bool>());
        built->setDamageOn(params.at("damage").get<bool>());
        if (meshDim==2) {
            built->setState(
                params.at("state").get<std::string>()=="plane_stress"
                    ? ExplicitPDDOFracElement::StressState::PlaneStress
                    : ExplicitPDDOFracElement::StressState::PlaneStrain);
        }
        element=std::move(built);
    }
    else if (type=="stress_cahnhilliard") {
        auto built=std::make_unique<StressCahnHilliardElement>();
        built->setE(params.at("E").get<double>());
        built->setNu(params.at("nu").get<double>());
        built->setDiffusivity(params.at("D").get<double>());
        built->setOmega(params.at("Omega").get<double>());
        built->setCref(params.at("cref").get<double>());
        built->setChi(params.at("chi").get<double>());
        built->setKappa(params.at("kappa").get<double>());
        if (meshDim==2) {
            built->setState(
                params.at("state").get<std::string>()=="plane_stress"
                    ? StressCahnHilliardElement::StressState::PlaneStress
                    : StressCahnHilliardElement::StressState::PlaneStrain);
        }
        element=std::move(built);
    }
    else if (type=="frac_stress_cahnhilliard") {
        auto built=std::make_unique<FracStressCahnHilliardElement>();
        built->setE(params.at("E").get<double>());
        built->setNu(params.at("nu").get<double>());
        built->setDiffusivity(params.at("D").get<double>());
        built->setOmega(params.at("Omega").get<double>());
        built->setCref(params.at("cref").get<double>());
        built->setChi(params.at("chi").get<double>());
        built->setKappa(params.at("kappa").get<double>());
        built->setRho(params.at("rho").get<double>());
        built->setG0(params.at("Gc").get<double>());
        built->setTensionOnly(params.at("tension_only").get<bool>());
        built->setDamageOn(params.at("damage").get<bool>());
        if (params.contains("residual_stiffness")) {
            built->setResidualStiffness(
                params.at("residual_stiffness").get<double>());
        }
        if (meshDim==2) {
            built->setState(
                params.at("state").get<std::string>()=="plane_stress"
                    ? FracStressCahnHilliardElement::StressState::PlaneStress
                    : FracStressCahnHilliardElement::StressState::PlaneStrain);
        }
        element=std::move(built);
    }

    if (!element) return false;
    element->setDim(meshDim);
    elements.addElement(name,std::move(element));
    return true;
}
