//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include "Utils/MessagePrinter.h"

namespace {
bool hasJsonExtension(const std::string &fileName) {
    return fileName.size()>=5
        && fileName.compare(fileName.size()-5,5,".json")==0;
}
}

InputSystem::InputSystem(int argc,char *argv[]) {
    parseCommandLine(argc,argv);
}

void InputSystem::init(int argc,char *argv[]) {
    parseCommandLine(argc,argv);
}

void InputSystem::parseCommandLine(int argc,char *argv[]) {
    m_ReadOnly=false;
    m_HasInputFile=false;
    m_InputFileName.clear();
    m_Json.clear();

    auto stop=[](const std::string &message) {
        MessagePrinter::printErrorTxt(message);
        MessagePrinter::exitPeriX();
    };
    if (argv==nullptr || argc<1) {
        stop("invalid command line; usage: perix -i input.json [--read-only]");
    }

    for (int i=1;i<argc;++i) {
        const std::string argument=argv[i] ? argv[i] : "";
        if (argument=="--read-only") {
            if (m_ReadOnly) stop("'--read-only' was supplied more than once");
            m_ReadOnly=true;
        }
        else if (argument=="-i") {
            if (m_HasInputFile) stop("'-i' was supplied more than once");
            if (i+1>=argc || argv[i+1]==nullptr) {
                stop("'-i' requires a JSON input-file name");
            }
            m_InputFileName=argv[++i];
            m_HasInputFile=true;
        }
        else {
            stop("unknown command-line option '"+argument
                 +"'; usage: perix -i input.json [--read-only]");
        }
    }

    if (!m_HasInputFile) {
        stop("no input file supplied; usage: perix -i input.json [--read-only]");
    }
    if (!hasJsonExtension(m_InputFileName)) {
        stop("input file must use the '.json' extension");
    }
}
