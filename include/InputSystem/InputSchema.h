//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>

#include "nlohmann/json.hpp"

/**
 * Strict schema for the manuscript-facing PeriX JSON interface.
 *
 * Validation is intentionally allow-list based. Unknown blocks, model types,
 * parameters, and aliases are errors, so adding implementation code elsewhere
 * cannot silently enlarge the public input surface.
 */
class InputSchema {
public:
    [[nodiscard]] static bool validate(
        const nlohmann::ordered_json &document,
        std::string &error,
        bool meshOnly=false);
};
