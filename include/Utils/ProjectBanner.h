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
//+++           single ProjectBanner class that holds the (year,
//+++           month, day, version) release tag and renders a
//+++           fixed-width, decorative ASCII banner on stdout. The
//+++           banner is the very first thing printed by perix
//+++           (and by the bench targets), so users can read the
//+++           release info at a glance, and so the tail of the
//+++           output is clearly bracketed.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <string>

/**
 * Header-banner printer for the PeriX command-line tools.
 *
 * The class does not own runtime state beyond the release tag. A
 * single static print() call renders the full banner; the
 * non-static variant takes a constructed instance and is provided
 * for callers that want to capture version data first (e.g. log
 * it elsewhere) and then print.
 */
class ProjectBanner {
public:
    /**
     * Construct a banner descriptor for a specific release tag.
     * @param year    release year (e.g. 2026)
     * @param month   release month, 1..12
     * @param day     release day, 1..31
     * @param version semantic version number (printed with 2 decimals)
     */
    ProjectBanner(const int year,const int month,const int day,const double version)
        : m_Year(year), m_Month(month), m_Day(day), m_Version(version) {}

    /**
     * One-shot convenience: build a banner and print it.
     * Equivalent to ProjectBanner(year,month,day,version).print().
     * @param year    release year
     * @param month   release month
     * @param day     release day
     * @param version semantic version number
     */
    static void print(const int year,const int month,const int day,const double version);

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

    [[nodiscard]] int    getYear()    const { return m_Year; }
    [[nodiscard]] int    getMonth()   const { return m_Month; }
    [[nodiscard]] int    getDay()     const { return m_Day; }
    [[nodiscard]] double getVersion() const { return m_Version; }

private:
    int    m_Year    = 0;   /**< release year */
    int    m_Month   = 0;   /**< release month (1..12) */
    int    m_Day     = 0;   /**< release day (1..31) */
    double m_Version = 0.0; /**< semantic version, printed with 2 decimals */
};
