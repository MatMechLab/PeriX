//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "OutputSystem/OutputSystem.h"

#include <algorithm>

#include "ElmtSystem/ElementBase.h"
#include "ElmtSystem/ElmtSystem.h"
#include "Utils/MessagePrinter.h"

namespace {
std::string projectionTypeLabel(const ProjectionType type) {
    switch (type) {
    case ProjectionType::Scalar:
        return "scalar";
    case ProjectionType::Vector:
        return "vector";
    case ProjectionType::Tensor:
        return "tensor";
    }
    return "unknown";
}

std::string outputFormatLabel(const OutputSystem::Format format) {
    switch (format) {
    case OutputSystem::Format::VTU:
        return "vtu";
    case OutputSystem::Format::Exodus:
        return "exodus";
    case OutputSystem::Format::Both:
        return "both";
    }
    return "unknown";
}
}

void OutputSystem::addField(const std::string &name) {
    if (name.empty() || name=="solution") return;
    const auto existing=std::find(m_Requested.begin(),m_Requested.end(),name);
    if (existing==m_Requested.end()) m_Requested.push_back(name);
}

void OutputSystem::resolve(const ElmtSystem &elements) {
    m_Resolved.clear();
    for (const std::string &name:m_Requested) {
        const ElementBase *provider=elements.findProjectionProvider(name);
        if (provider==nullptr) {
            MessagePrinter::printErrorTxt(
                "OutputSystem: no published element projection named '"+name+"'");
            MessagePrinter::exitPeriX();
            return;
        }

        const std::vector<ProjectionInfo> available=
            provider->getAvailableProjections();
        const auto match=std::find_if(
            available.begin(),available.end(),
            [&name](const ProjectionInfo &field) {
                return field.Name==name;
            });
        if (match==available.end() || match->Components<1) {
            MessagePrinter::printErrorTxt(
                "OutputSystem: invalid projection metadata for '"+name+"'");
            MessagePrinter::exitPeriX();
            return;
        }
        m_Resolved.push_back(*match);
    }
}

void OutputSystem::printOutputSystemInfo() const {
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Output system");
    MessagePrinter::printNormalTxt(
        "  format: "+outputFormatLabel(m_Format));
    MessagePrinter::printNormalTxt(
        "  interval: "+std::to_string(m_Interval)+" step(s)");
    MessagePrinter::printNormalTxt(
        "  derived fields: "+std::to_string(m_Resolved.size()));
    for (const ProjectionInfo &field:m_Resolved) {
        MessagePrinter::printNormalTxt(
            "  - "+field.Name+" ("+projectionTypeLabel(field.Type)+", "
            +std::to_string(field.Components)+" component(s))");
    }
    MessagePrinter::printStars();
}
