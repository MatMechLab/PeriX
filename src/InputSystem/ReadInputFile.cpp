//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <exception>
#include <fstream>

#include "BCSystem/BCSystem.h"
#include "ElmtSystem/ElmtSystem.h"
#include "ICSystem/ICSystem.h"
#include "InputSystem/InputSchema.h"
#include "JobSystem/JobSystem.h"
#include "LinearSolver/LinearSolver.h"
#include "Mesh/Mesh.h"
#include "MeshModify/MeshModify.h"
#include "NonlinearSolver/NonlinearSolver.h"
#include "OutputSystem/OutputSystem.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "TimeStepping/TimeStepping.h"
#include "Utils/MessagePrinter.h"

bool InputSystem::readInputFile(Mesh &mesh,
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
                                OutputSystem &output) {
    auto stop=[](const std::string &message)->bool {
        MessagePrinter::printErrorTxt(message);
        MessagePrinter::exitPeriX();
        return false;
    };

    std::ifstream input(m_InputFileName);
    if (!input.is_open()) {
        return stop("cannot open input file '"+m_InputFileName+"'");
    }
    try {
        input >> m_Json;
    }
    catch (const std::exception &exception) {
        return stop("cannot parse input file '"+m_InputFileName
                    +"': "+exception.what());
    }

    std::string schemaError;
    if (!InputSchema::validate(m_Json,schemaError,m_ReadOnly)) {
        return stop("input schema error: "+schemaError);
    }

    if (!readMeshBlock(m_Json.at("Mesh"),mesh)) {
        return stop("failed to configure Mesh");
    }
    if (m_ReadOnly) return true;

    if (!readPDMeshBlock(m_Json.at("PDMesh"),mesh.getMeshDataRef(),
                         pdMesh,operators)) {
        return stop("failed to configure PDMesh");
    }
    if (m_Json.contains("MeshModify")
        && !readMeshModifyBlock(m_Json.at("MeshModify"),meshModify)) {
        return stop("failed to configure MeshModify");
    }
    if (!readElmtSystemBlock(m_Json.at("ElmtSystem"),elements,mesh.getMeshDim())) {
        return stop("failed to configure ElmtSystem");
    }

    DofNameMap dofs;
    if (!readDOFsBlock(m_Json.at("DOFs"),elements,dofs)) {
        return stop("failed to configure DOFs");
    }
    if (!readBCSystemBlock(m_Json.at("BCSystem"),boundaryConditions,
                           dofs,mesh.getMeshDim())) {
        return stop("failed to configure BCSystem");
    }
    if (m_Json.contains("ICSystem")
        && !readICSystemBlock(m_Json.at("ICSystem"),initialConditions,dofs)) {
        return stop("failed to configure ICSystem");
    }
    if (m_Json.contains("JobSystem")
        && !readJobSystemBlock(m_Json.at("JobSystem"),job)) {
        return stop("failed to configure JobSystem");
    }
    if (m_Json.contains("TimeStepping")
        && !readTimeSteppingBlock(m_Json.at("TimeStepping"),timeStepping)) {
        return stop("failed to configure TimeStepping");
    }
    if (m_Json.contains("NonlinearSolver")
        && !readNonlinearSolverBlock(m_Json.at("NonlinearSolver"),nonlinearSolver)) {
        return stop("failed to configure NonlinearSolver");
    }
    if (m_Json.contains("LinearSolver")
        && !readLinearSolverBlock(m_Json.at("LinearSolver"),linearSolver)) {
        return stop("failed to configure LinearSolver");
    }
    if (m_Json.contains("OutputSystem")
        && !readOutputSystemBlock(m_Json.at("OutputSystem"),output)) {
        return stop("failed to configure OutputSystem");
    }
    return true;
}
