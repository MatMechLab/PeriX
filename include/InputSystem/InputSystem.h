//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>

#include "InputSystem/DofNameMap.h"
#include "nlohmann/json.hpp"

class BCSystem;
class ElmtSystem;
class ICSystem;
class JobSystem;
class LinearSolver;
class Mesh;
struct MeshData;
class MeshModify;
class NonlinearSolver;
class OutputSystem;
class PDMesh;
class PDOperators;
class TimeStepping;

/**
 * Reads one validated JSON document and distributes its public blocks to the
 * PeriX subsystems.
 */
class InputSystem {
public:
    InputSystem()=default;
    InputSystem(int argc,char *argv[]);
    ~InputSystem()=default;

    void init(int argc,char *argv[]);

    bool readInputFile(Mesh &mesh,
                       PDMesh &pdMesh,
                       MeshModify &meshModify,
                       PDOperators &operators,
                       ElmtSystem &elements,
                       BCSystem &boundaryConditions,
                       ICSystem &initialConditions,
                       JobSystem &job,
                       TimeStepping &timeStepping,
                       NonlinearSolver &nonlinearSolver,
                       LinearSolver &linearSolver,
                       OutputSystem &output);

    [[nodiscard]] const std::string& getInputFileName() const noexcept {
        return m_InputFileName;
    }
    [[nodiscard]] bool isReadOnly() const noexcept { return m_ReadOnly; }

private:
    void parseCommandLine(int argc,char *argv[]);

    [[nodiscard]] bool readMeshBlock(const nlohmann::ordered_json &json,Mesh &mesh);
    [[nodiscard]] bool readPDMeshBlock(const nlohmann::ordered_json &json,
                                       const MeshData &meshData,
                                       PDMesh &pdMesh,
                                       PDOperators &operators);
    [[nodiscard]] bool readMeshModifyBlock(const nlohmann::ordered_json &json,
                                           MeshModify &meshModify);
    [[nodiscard]] bool readElmtSystemBlock(const nlohmann::ordered_json &json,
                                           ElmtSystem &elements,
                                           int meshDim);
    [[nodiscard]] bool readDOFsBlock(const nlohmann::ordered_json &json,
                                     const ElmtSystem &elements,
                                     DofNameMap &dofs);
    [[nodiscard]] bool readBCSystemBlock(const nlohmann::ordered_json &json,
                                         BCSystem &boundaryConditions,
                                         const DofNameMap &dofs,
                                         int meshDim);
    [[nodiscard]] bool readICSystemBlock(const nlohmann::ordered_json &json,
                                         ICSystem &initialConditions,
                                         const DofNameMap &dofs);
    [[nodiscard]] bool readJobSystemBlock(const nlohmann::ordered_json &json,
                                          JobSystem &job);
    [[nodiscard]] bool readTimeSteppingBlock(const nlohmann::ordered_json &json,
                                             TimeStepping &timeStepping);
    [[nodiscard]] bool readNonlinearSolverBlock(const nlohmann::ordered_json &json,
                                                NonlinearSolver &solver);
    [[nodiscard]] bool readLinearSolverBlock(const nlohmann::ordered_json &json,
                                             LinearSolver &solver);
    [[nodiscard]] bool readOutputSystemBlock(const nlohmann::ordered_json &json,
                                             OutputSystem &output);

    bool m_ReadOnly=false;
    bool m_HasInputFile=false;
    std::string m_InputFileName;
    nlohmann::ordered_json m_Json;
};
