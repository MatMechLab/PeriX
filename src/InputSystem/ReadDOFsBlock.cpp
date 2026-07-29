//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <vector>

#include "ElmtSystem/ElmtSystem.h"
#include "Utils/MessagePrinter.h"

bool InputSystem::readDOFsBlock(const nlohmann::ordered_json &json,
                                const ElmtSystem &elements,
                                DofNameMap &dofs) {
    std::vector<std::string> names;
    names.reserve(json.size());
    for (const auto &entry:json) names.push_back(entry.get<std::string>());

    for (int i=0;i<elements.getElementsNum();++i) {
        const auto expected=elements.getElementByIndex(i).getDofNames();
        if (names!=expected) {
            MessagePrinter::printErrorTxt(
                "DOFs do not match element '"+elements.getElementNameByIndex(i)
                +"' canonical field order");
            return false;
        }
    }
    dofs.setNames(names);
    return true;
}
