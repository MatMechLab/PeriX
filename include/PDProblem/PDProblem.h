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
//+++ Date    : 2026.04.15
//+++ Function: define the pdproblem to be solved in PeriX.
//+++           Drives the static or transient analysis: assemble,
//+++           apply BCs, linear solve via Pardiso, time-stepping
//+++           if requested.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include "BCSystem/BCSystem.h"
#include "ElmtSystem/ElmtSystem.h"
#include "ICSystem/ICSystem.h"
#include "InputSystem/InputSystem.h"
#include "JobSystem/JobSystem.h"
#include "Mesh/Mesh.h"
#include "NonlinearSolver/NonlinearSolver.h"
#include "OutputSystem/OutputSystem.h"
#include "OutputSystem/ExodusWriter.h"
#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "PDSystem/PDSystem.h"
#include "LinearSolver/LinearSolver.h"
#include "MathUtils/SparseMatrix.h"
#include "MeshModify/MeshModify.h"
#include "MathUtils/VectorXd.h"
#include "TimeStepping/TimeStepping.h"
#include "Utils/Timer.h"

#include <utility>
#include <vector>

class PDProblem {
public:
    PDProblem();

    void run(int args,char *argv[]);

private:
    void init(int args,char *argv[]);
    void printBasicInfo();

    /**
     * write the PD-point-cloud and FEM-element solution VTUs for
     * the current step. Each call also appends the (time,filename)
     * entry to the corresponding output history and rewrites the
     * matching ".pvd" collection so the time series is browsable
     * even if the run is interrupted.
    */
    void savePDResults(const string &inputfilename,const int &step,const double &time);
    void saveElementResults(const string &inputfilename,const int &step,const double &time);
    /**
     * Append one step to the Exodus II (.e) output: the finite-element mesh
     * (nodes + bulk element block + boundary node sets) is written once on the
     * first call, then one time record of the per-element results (DoFs and
     * projections, mapped from the PD point at each element centroid) is
     * appended. A single .e file thus holds the whole time series.
     */
    void saveExodusResults(const string &inputfilename,const int &step,const double &time);

    /**
     * helper: rewrite the ParaView Collection (.pvd) index for an
     * output history. Filenames recorded inside the .pvd are
     * stored as basenames so the directory can be moved freely.
     */
    void writePVDFile(const string &pvdFilePath,
                      const std::vector<std::pair<double,string>> &entries) const;

private:
    InputSystem m_InputSystem;
    Mesh m_Mesh;
    MeshModify m_MeshModify;
    PDMesh m_PDMesh;
    PDOperators m_Operators;
    ElmtSystem m_ElmtSystem;
    BCSystem m_BCSystem;
    ICSystem m_ICSystem;
    JobSystem m_JobSystem;
    OutputSystem m_OutputSystem;
    ExodusWriter m_ExodusWriter;   ///< persistent .e writer (mesh once, records appended)
    bool         m_ExodusBegun=false;
    PDSystem m_PDSystem;
    LinearSolver m_LinearSolver;
    NonlinearSolver m_NonlinearSolver;
    TimeStepping m_TimeStepping;

    SparseMatrix m_Matrix;
    VectorXd m_RHS,m_dU,m_U,m_Uold,m_InitialVelocity;
    Timer m_Timer;

    // (time, basename) entries for ParaView Collection (.pvd) files
    std::vector<std::pair<double,string>> m_PDOutputs;
    std::vector<std::pair<double,string>> m_ElmtOutputs;
};
