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
//+++ Date    : 2026.04.14
//+++ Function: dispatcher implementation for the LinearSolver
//+++           wrapper. Default backend is the in-house direct
//+++           profile LDU solver; Pardiso is available as an
//+++           opt-in alternative when oneAPI/MKL is desired,
//+++           gated by the PERIX_HAS_PARDISO compile-time macro.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "LinearSolver/LinearSolver.h"

#include "Utils/MessagePrinter.h"


LinearSolver::LinearSolver(){
    m_SolverType=LinearSolverType::DEFAULT;
    m_LinearSolverName="Default";
    m_Parameters.clear();
}
//***************************************************
void LinearSolver::setSolverType(const LinearSolverType &type){
    if(type==LinearSolverType::DEFAULT){
        m_SolverType=LinearSolverType::DEFAULT;
        m_LinearSolverName="Default";
    }
    else if(type==LinearSolverType::PARDISO){
#ifdef PERIX_HAS_PARDISO
        m_SolverType=LinearSolverType::PARDISO;
        m_LinearSolverName="Pardiso";
#else
        MessagePrinter::printErrorTxt("PARDISO backend was requested but PeriX was built without oneAPI support. "
                                      "Re-configure with -DUSE_ONEAPI=on -DONEAPI_DIR=/path/to/oneapi to enable it.");
        MessagePrinter::exitPeriX();
#endif
    }
    else if(type==LinearSolverType::CUDSS){
#ifdef PERIX_HAS_CUDSS
        m_SolverType=LinearSolverType::CUDSS;
        m_LinearSolverName="cuDSS";
#else
        MessagePrinter::printErrorTxt("cuDSS backend was requested but PeriX was built without cuDSS support. "
                                      "Re-configure with -DUSE_CUDSS=on -DCUDA_DIR=... -DCUDSS_DIR=... to enable it.");
        MessagePrinter::exitPeriX();
#endif
    }
    else if(type==LinearSolverType::AMGCL){
#ifdef PERIX_HAS_AMGCL
        m_SolverType=LinearSolverType::AMGCL;
        m_LinearSolverName="AMGCL";
#else
        MessagePrinter::printErrorTxt(
            "AMGCL backend was requested but PeriX was built without AMGCL support. "
            "Re-configure with -DUSE_AMGCL=on -DAMGCL_DIR=/path/to/amgcl.");
        MessagePrinter::exitPeriX();
#endif
    }
    else{
        MessagePrinter::printErrorTxt("unsupported linear solver type");
        MessagePrinter::exitPeriX();
    }
}
//***************************************************
void LinearSolver::init(SparseMatrix &K){
    if(m_SolverType==LinearSolverType::DEFAULT){
        m_DefaultSolver.initSolver(K,m_Parameters);
    }
#ifdef PERIX_HAS_PARDISO
    else if(m_SolverType==LinearSolverType::PARDISO){
        m_PardisoSolver.initSolver(K,m_Parameters);
    }
#endif
#ifdef PERIX_HAS_CUDSS
    else if(m_SolverType==LinearSolverType::CUDSS){
        m_CudssSolver.initSolver(K,m_Parameters);
    }
#endif
#ifdef PERIX_HAS_AMGCL
    else if(m_SolverType==LinearSolverType::AMGCL){
        m_AmgclSolver.initSolver(K,m_Parameters);
    }
#endif
    else{
        MessagePrinter::printErrorTxt("Unsupported solver type for init");
        MessagePrinter::exitPeriX();
    }
}
//***************************************************
bool LinearSolver::solve(SparseMatrix &A,VectorXd &rhs,VectorXd &x){
    if(m_SolverType==LinearSolverType::DEFAULT){
        return m_DefaultSolver.solveLinearSystem(A,rhs,x);
    }
#ifdef PERIX_HAS_PARDISO
    else if(m_SolverType==LinearSolverType::PARDISO){
        return m_PardisoSolver.solveLinearSystem(A,rhs,x);
    }
#endif
#ifdef PERIX_HAS_CUDSS
    else if(m_SolverType==LinearSolverType::CUDSS){
        return m_CudssSolver.solveLinearSystem(A,rhs,x);
    }
#endif
#ifdef PERIX_HAS_AMGCL
    else if(m_SolverType==LinearSolverType::AMGCL){
        return m_AmgclSolver.solveLinearSystem(A,rhs,x);
    }
#endif
    else{
        MessagePrinter::printErrorTxt("Unsupported solver type for solve process");
        MessagePrinter::exitPeriX();
    }
    return true;
}
//***************************************************
void LinearSolver::printLinearSolverInfo() const{
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Linear solver info:");
    MessagePrinter::printNormalTxt("  backend="+m_LinearSolverName);
    if (m_Parameters.is_object() && !m_Parameters.empty()) {
        MessagePrinter::printNormalTxt("  params:");
        for (auto it=m_Parameters.begin();it!=m_Parameters.end();++it) {
            const auto &v=it.value();
            std::string val;
            if      (v.is_boolean())        val=(v.get<bool>()?"true":"false");
            else if (v.is_number_integer()) val=std::to_string(v.get<long long>());
            else if (v.is_number())         val=std::to_string(v.get<double>());
            else if (v.is_string())         val="\""+v.get<std::string>()+"\"";
            else                            val=v.dump();
            MessagePrinter::printNormalTxt("    "+it.key()+" = "+val);
        }
    }
    else {
        MessagePrinter::printNormalTxt("  params: (defaults)");
    }
    MessagePrinter::printStars();
}
