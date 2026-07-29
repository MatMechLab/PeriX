//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "JobSystem/JobSystem.h"

#include "Utils/MessagePrinter.h"

std::string assembleTypeName(const AssembleType type) {
    switch (type) {
    case AssembleType::SERIAL:
        return "serial";
    case AssembleType::OPENMP:
        return "openmp";
    case AssembleType::CUDA:
        return "cuda";
    }
    return "unknown";
}

std::string jobTypeName(const JobType type) {
    switch (type) {
    case JobType::STATIC:
        return "static";
    case JobType::TRANSIENT:
        return "transient";
    }
    return "unknown";
}

void JobSystem::printJobSystemInfo() const {
    const std::string typeLine="  analysis: "+jobTypeName(m_JobType);
    const std::string backendLine="  assembly: "+assembleTypeName(m_Assembly);
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Job system");
    MessagePrinter::printNormalTxt(typeLine);
    MessagePrinter::printNormalTxt(backendLine);
    MessagePrinter::printStars();
}
