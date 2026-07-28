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
//+++ Function: implementation of the ProjectBanner class. Composes
//+++           the fixed-width 105-column PeriX banner (release
//+++           date, version, author, contact, website) into a
//+++           stack of printf calls and emits them to stdout.
//+++           Each metadata string (name, tagline, email, website,
//+++           author) is returned by a small accessor so unit
//+++           tests and downstream tools can read the same
//+++           constants without duplicating literals.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Utils/ProjectBanner.h"

#include <cstdio>

std::string ProjectBanner::getName() {
    return "PeriX";
}

std::string ProjectBanner::getTagline() {
    return "Peridynamics framework for multiphysics simulation";
}

std::string ProjectBanner::getBugReportEmail() {
    return "yangbai90@outlook.com";
}

std::string ProjectBanner::getWebsite() {
    return "https://github.com/MatMechLab/PeriX";
}

std::string ProjectBanner::getAuthor() {
    return "Yang Bai/MMLab-members";
}

void ProjectBanner::print(const int year,const int month,const int day,const double version) {
    ProjectBanner(year,month,day,version).print();
}

void ProjectBanner::print() const {
    std::printf("*********************************************************************************************************\n");
    std::printf("*** Welcome to use PeriX :)                                                          PX           XP  ***\n");
    std::printf("*** Peridynamics framework for multiphysic simulation                                  PX       XP    ***\n");
    std::printf("*** Release date: %4d-%02d-%02d                                                              PX   XP     ***\n",
                m_Year,m_Month,m_Day);
    std::printf("*** Version     : %10.2f                                                                PXP       ***\n",
                m_Version);
    std::printf("*** Bug report  : yangbai90@outlook.com                                                   PX   XP     ***\n");
    std::printf("*** Author      : Yang Bai/MMLab-members                                                PX       XP   ***\n");
    std::printf("*** Website     : https://github.com/MatMechLab/PeriX                                 PX           XP ***\n");
    std::printf("*********************************************************************************************************\n");
}
