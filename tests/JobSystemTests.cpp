//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cstdio>
#include <string>

#include "JobSystem/JobSystem.h"

namespace {
int failures=0;

void expect(const char *name,const bool condition) {
    if (condition) return;
    std::printf("FAIL %s\n",name);
    ++failures;
}

void expectName(const char *name,const std::string &actual,const char *expected) {
    if (actual==expected) return;
    std::printf("FAIL %s: expected '%s', got '%s'\n",
                name,expected,actual.c_str());
    ++failures;
}
}

int main() {
    JobSystem job;
    expect("default analysis is static",job.getJobType()==JobType::STATIC);
    expect("default analysis is not transient",!job.isTransient());
    expect("default assembly is serial",
           job.getAssembleType()==AssembleType::SERIAL);

    job.setJobType(JobType::TRANSIENT);
    job.setAssembleType(AssembleType::OPENMP);
    expect("transient selection",job.isTransient());
    expect("OpenMP selection",
           job.getAssembleType()==AssembleType::OPENMP);

    job.setAssembleType(AssembleType::CUDA);
    expect("CUDA selection",job.getAssembleType()==AssembleType::CUDA);

    expectName("static name",jobTypeName(JobType::STATIC),"static");
    expectName("transient name",jobTypeName(JobType::TRANSIENT),"transient");
    expectName("serial name",assembleTypeName(AssembleType::SERIAL),"serial");
    expectName("OpenMP name",assembleTypeName(AssembleType::OPENMP),"openmp");
    expectName("CUDA name",assembleTypeName(AssembleType::CUDA),"cuda");

    if (failures==0) {
        std::printf("JobSystem tests passed\n");
        return 0;
    }
    std::printf("JobSystem tests failed: %d\n",failures);
    return 1;
}
