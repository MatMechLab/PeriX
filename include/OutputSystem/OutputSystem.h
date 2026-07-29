//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>
#include <vector>

#include "ElmtSystem/ProjectionInfo.h"

class ElmtSystem;

/**
 * Stores the output cadence, container format, and element-derived fields
 * exposed by the manuscript's OutputSystem block.
 */
class OutputSystem {
public:
    enum class Format {
        VTU,
        Exodus,
        Both
    };

    OutputSystem()=default;

    void setFormat(Format format) noexcept { m_Format=format; }
    [[nodiscard]] Format getFormat() const noexcept { return m_Format; }
    [[nodiscard]] bool wantsVTU() const noexcept {
        return m_Format==Format::VTU || m_Format==Format::Both;
    }
    [[nodiscard]] bool wantsExodus() const noexcept {
        return m_Format==Format::Exodus || m_Format==Format::Both;
    }

    void setInterval(int interval) noexcept {
        m_Interval=interval<1 ? 1 : interval;
    }
    [[nodiscard]] int getInterval() const noexcept { return m_Interval; }

    void addField(const std::string &name);
    [[nodiscard]] int getRequestedFieldsNum() const noexcept {
        return static_cast<int>(m_Requested.size());
    }

    void resolve(const ElmtSystem &elements);
    [[nodiscard]] const std::vector<ProjectionInfo>& getResolvedFields() const noexcept {
        return m_Resolved;
    }
    [[nodiscard]] int getFieldsNum() const noexcept {
        return static_cast<int>(m_Resolved.size());
    }

    void printOutputSystemInfo() const;

private:
    std::vector<std::string> m_Requested;
    std::vector<ProjectionInfo> m_Resolved;
    int m_Interval=1;
    Format m_Format=Format::VTU;
};
