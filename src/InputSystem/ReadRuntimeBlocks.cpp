//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include "JobSystem/JobSystem.h"
#include "LinearSolver/LinearSolver.h"
#include "NonlinearSolver/NonlinearSolver.h"
#include "OutputSystem/OutputSystem.h"
#include "TimeStepping/TimeStepping.h"
#include "Utils/MessagePrinter.h"

bool InputSystem::readJobSystemBlock(const nlohmann::ordered_json &json,
                                     JobSystem &job) {
    job.setJobType(json.at("type").get<std::string>()=="transient"
        ? JobType::TRANSIENT : JobType::STATIC);
    const std::string assemble=json.value("assemble",std::string("serial"));
    if (assemble=="serial") {
        job.setAssembleType(AssembleType::SERIAL);
        return true;
    }
    if (assemble=="openmp") {
#ifdef PERIX_PARALLEL_ASSEMBLE
        job.setAssembleType(AssembleType::OPENMP);
        return true;
#else
        MessagePrinter::printErrorTxt(
            "JobSystem: assemble='openmp' requires a build with PARALLEL_ASSEMBLE enabled");
        return false;
#endif
    }
    if (assemble=="cuda") {
#ifdef PERIX_CUDA_ASSEMBLE
        job.setAssembleType(AssembleType::CUDA);
        return true;
#else
        MessagePrinter::printErrorTxt(
            "JobSystem: assemble='cuda' requires a build with CUDA_ASSEMBLE enabled");
        return false;
#endif
    }
    return false;
}

bool InputSystem::readTimeSteppingBlock(const nlohmann::ordered_json &json,
                                        TimeStepping &timeStepping) {
    timeStepping.setDt(json.at("dt").get<double>());
    timeStepping.setTotalTime(json.at("total_time").get<double>());
    if (json.contains("verbose")) {
        timeStepping.setVerbose(json.at("verbose").get<bool>());
    }
    if (json.contains("adaptive")) {
        timeStepping.setAdaptive(json.at("adaptive").get<bool>());
    }
    if (json.contains("optimal_iters")) {
        timeStepping.setOptimalIters(json.at("optimal_iters").get<int>());
    }
    if (json.contains("growth_factor")) {
        timeStepping.setGrowthFactor(json.at("growth_factor").get<double>());
    }
    if (json.contains("cutback_factor")) {
        timeStepping.setCutbackFactor(json.at("cutback_factor").get<double>());
    }
    if (json.contains("max_cutbacks")) {
        timeStepping.setMaxCutbacks(json.at("max_cutbacks").get<int>());
    }
    if (json.contains("min_dt")) {
        timeStepping.setMinDt(json.at("min_dt").get<double>());
    }
    if (json.contains("max_dt")) {
        timeStepping.setMaxDt(json.at("max_dt").get<double>());
    }
    return true;
}

bool InputSystem::readNonlinearSolverBlock(const nlohmann::ordered_json &json,
                                           NonlinearSolver &solver) {
    if (json.contains("max_iters")) solver.setMaxIters(json.at("max_iters").get<int>());
    if (json.contains("abs_tol")) solver.setAbsTol(json.at("abs_tol").get<double>());
    if (json.contains("rel_tol")) solver.setRelTol(json.at("rel_tol").get<double>());
    if (json.contains("verbose")) solver.setVerbose(json.at("verbose").get<bool>());
    return true;
}

bool InputSystem::readLinearSolverBlock(const nlohmann::ordered_json &json,
                                        LinearSolver &solver) {
    const std::string type=json.value("type",std::string("default"));
    LinearSolverType selected=LinearSolverType::DEFAULT;
    if (type=="pardiso") {
#ifdef PERIX_HAS_PARDISO
        selected=LinearSolverType::PARDISO;
#else
        MessagePrinter::printErrorTxt(
            "LinearSolver: type='pardiso' requires a build with oneAPI PARDISO enabled");
        return false;
#endif
    }
    else if (type=="cudss") {
#ifdef PERIX_HAS_CUDSS
        selected=LinearSolverType::CUDSS;
#else
        MessagePrinter::printErrorTxt(
            "LinearSolver: type='cudss' requires a build with cuDSS enabled");
        return false;
#endif
    }
    else if (type=="amgcl") {
#ifdef PERIX_HAS_AMGCL
        selected=LinearSolverType::AMGCL;
#else
        MessagePrinter::printErrorTxt(
            "LinearSolver: type='amgcl' requires a build with AMGCL enabled");
        return false;
#endif
    }
    solver.setSolverType(selected);
    nlohmann::ordered_json parameters=json.value(
        "params",nlohmann::ordered_json::object());
    solver.setSolverParameters(parameters);
    return true;
}

bool InputSystem::readOutputSystemBlock(const nlohmann::ordered_json &json,
                                        OutputSystem &output) {
    if (json.contains("interval")) output.setInterval(json.at("interval").get<int>());
    if (json.contains("Fields")) {
        for (const auto &field:json.at("Fields")) {
            output.addField(field.get<std::string>());
        }
    }
    const std::string format=json.value("format",std::string("vtu"));
    if (format=="exodus") output.setFormat(OutputSystem::Format::Exodus);
    else if (format=="both") output.setFormat(OutputSystem::Format::Both);
    else output.setFormat(OutputSystem::Format::VTU);
    return true;
}
