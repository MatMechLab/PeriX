//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>

#include "JobSystem/AssembleType.h"

enum class JobType {
    STATIC,
    TRANSIENT
};

[[nodiscard]] std::string jobTypeName(JobType type);

/**
 * Stores the two run-wide choices exposed by the manuscript's JobSystem
 * input block: analysis type and assembly backend.
 */
class JobSystem {
public:
    JobSystem()=default;

    void setJobType(JobType type) noexcept { m_JobType=type; }
    [[nodiscard]] JobType getJobType() const noexcept { return m_JobType; }
    [[nodiscard]] bool isTransient() const noexcept {
        return m_JobType==JobType::TRANSIENT;
    }

    void setAssembleType(AssembleType type) noexcept { m_Assembly=type; }
    [[nodiscard]] AssembleType getAssembleType() const noexcept {
        return m_Assembly;
    }

    void printJobSystemInfo() const;

private:
    JobType m_JobType=JobType::STATIC;
    AssembleType m_Assembly=AssembleType::SERIAL;
};
