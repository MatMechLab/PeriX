//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Author  : Yang Bai
//+++ Date    : 2026.05.25
//+++ Function: declare the PeriX project banner printer. Owns the
//+++           single ProjectBanner class that holds the semantic
//+++           version string and renders a
//+++           fixed-width, decorative ASCII banner on stdout. The
//+++           banner is the very first thing printed by perix
//+++           (and by the bench targets), so users can read the
//+++           release info at a glance, and so the tail of the
//+++           output is clearly bracketed.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>
#include <utility>

/**
 * Header-banner printer for the PeriX command-line tools.
 *
 * The class owns only the semantic version string. A single static print()
 * call renders the full banner; the non-static variant is provided for
 * callers that want to inspect the version first and then print.
 */
class ProjectBanner {
public:
    /**
     * Construct a banner descriptor for a semantic release version.
     * @param version semantic version string
     */
    explicit ProjectBanner(std::string version):m_Version(std::move(version)) {}

    /**
     * One-shot convenience: build a banner and print it.
     * Equivalent to ProjectBanner(version).print().
     * @param version semantic version string
     */
    static void print(const std::string &version);

    /**
     * Render the banner for this instance to stdout. The width is
     * fixed at 105 characters to match the rest of PeriX's terminal
     * output (see MessagePrinter::m_Words).
     */
    void print() const;

    /**
     * @return the project's display name ("PeriX").
     */
    [[nodiscard]] static std::string getName();

    /**
     * @return the project's one-line tagline.
     */
    [[nodiscard]] static std::string getTagline();

    /**
     * @return the public bug-report contact for the project.
     */
    [[nodiscard]] static std::string getBugReportEmail();

    /**
     * @return the canonical project website / source URL.
     */
    [[nodiscard]] static std::string getWebsite();

    /**
     * @return the project author(s) credit line.
     */
    [[nodiscard]] static std::string getAuthor();

    [[nodiscard]] const std::string& getVersion() const { return m_Version; }

private:
    std::string m_Version; /**< semantic version string */
};
