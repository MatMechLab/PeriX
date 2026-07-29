//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <memory>
#include <string>
#include <vector>

struct MeshData;

/**
 * Writes the finite-element mesh and its element-centered results to the
 * Exodus II convention carried by a NetCDF CDF-2 file.
 *
 * The writer is deliberately self-contained: the public implementation does
 * not require the NetCDF or Exodus libraries.
 */
class ExodusWriter {
public:
    ExodusWriter();
    ~ExodusWriter();

    ExodusWriter(const ExodusWriter &)=delete;
    ExodusWriter& operator=(const ExodusWriter &)=delete;
    ExodusWriter(ExodusWriter &&) noexcept;
    ExodusWriter& operator=(ExodusWriter &&) noexcept;

    bool begin(const std::string &filename,
               const MeshData &mesh,
               const std::vector<std::string> &variableNames,
               const std::string &title);

    bool appendStep(double time,
                    const std::vector<std::vector<double>> &elementValues);

    [[nodiscard]] bool isOpen() const noexcept;

private:
    class State;
    std::unique_ptr<State> m_State;
};
