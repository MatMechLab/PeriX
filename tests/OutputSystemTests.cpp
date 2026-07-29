//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cstdio>

#include "OutputSystem/OutputSystem.h"

namespace {
int failures=0;

void expect(const char *name,const bool condition) {
    if (condition) return;
    std::printf("FAIL %s\n",name);
    ++failures;
}
}

int main() {
    OutputSystem output;
    expect("default format is VTU",
           output.getFormat()==OutputSystem::Format::VTU);
    expect("default writes VTU",output.wantsVTU());
    expect("default skips Exodus",!output.wantsExodus());
    expect("default interval",output.getInterval()==1);

    output.setInterval(25);
    expect("configured interval",output.getInterval()==25);
    output.setInterval(0);
    expect("interval is clamped",output.getInterval()==1);

    output.setFormat(OutputSystem::Format::Exodus);
    expect("Exodus selection",!output.wantsVTU() && output.wantsExodus());
    output.setFormat(OutputSystem::Format::Both);
    expect("combined selection",output.wantsVTU() && output.wantsExodus());

    output.addField("");
    output.addField("solution");
    output.addField("damage");
    output.addField("damage");
    output.addField("stress");
    expect("reserved and duplicate fields omitted",
           output.getRequestedFieldsNum()==2);

    if (failures==0) {
        std::printf("OutputSystem tests passed\n");
        return 0;
    }
    std::printf("OutputSystem tests failed: %d\n",failures);
    return 1;
}
