//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSystem.h"

#include <array>
#include <filesystem>
#include <string>

#include "Mesh/Circle2DLatticeGenerator.h"
#include "Mesh/Mesh.h"
#include "Mesh/MeshGenerator.h"
#include "Mesh/MeshImport.h"
#include "Utils/MessagePrinter.h"

namespace {
std::string savedMeshName(const std::string &inputName) {
    const std::filesystem::path input(inputName);
    return (input.parent_path()/(input.stem().string()+"-mesh.vtu")).string();
}

std::string inputRelativePath(const std::string &inputName,
                              const std::string &referencedName) {
    const std::filesystem::path referenced(referencedName);
    if (referenced.is_absolute()) return referenced.string();
    return (std::filesystem::path(inputName).parent_path()/referenced).string();
}
}

bool InputSystem::readMeshBlock(const nlohmann::ordered_json &json,Mesh &mesh) {
    const std::string type=json.at("type").get<std::string>();
    const bool save=json.value("savemesh",false);
    bool ok=false;

    if (type=="perix") {
        const int dim=json.at("dim").get<int>();
        const int nx=json.at("nx").get<int>();
        const int ny=json.at("ny").get<int>();
        const double xmin=json.value("xmin",0.0);
        const double xmax=json.value("xmax",1.0);
        const double ymin=json.value("ymin",0.0);
        const double ymax=json.value("ymax",1.0);
        MeshGenerator generator;
        if (dim==2) {
            mesh.setMeshInfo(nx,ny,xmin,xmax,ymin,ymax,MeshType::QUAD4);
            ok=generator.createMesh(MeshType::QUAD4,mesh.getMeshDataRef());
        }
        else {
            const int nz=json.at("nz").get<int>();
            const double zmin=json.value("zmin",0.0);
            const double zmax=json.value("zmax",1.0);
            mesh.setMeshInfo(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,MeshType::HEX8);
            ok=generator.createMesh(MeshType::HEX8,mesh.getMeshDataRef());
        }
    }
    else if (type=="gmsh") {
        const std::string file=inputRelativePath(
            m_InputFileName,json.at("file").get<std::string>());
        MeshImport importer;
        mesh.resetMeshData();
        ok=importer.importMSH(file,mesh.getMeshDataRef());
    }
    else if (type=="circle") {
        PDLatticeGeneratorBase::Params parameters;
        parameters.R=json.at("radius").get<double>();
        parameters.N=json.at("layers").get<int>();
        if (json.contains("center")) {
            const auto &center=json.at("center");
            for (std::size_t i=0;i<center.size();++i) {
                parameters.center[i]=center[i].get<double>();
            }
        }
        if (json.contains("boundary")) {
            const auto &boundary=json.at("boundary");
            if (boundary.is_boolean()) {
                parameters.makeBoundaryGroup=boundary.get<bool>();
            }
            else {
                parameters.makeBoundaryGroup=true;
                parameters.boundaryName=boundary.get<std::string>();
            }
        }
        Circle2DLatticeGenerator generator;
        mesh.resetMeshData();
        ok=generator.generate(parameters,mesh.getMeshDataRef());
    }

    if (!ok) {
        MessagePrinter::printErrorTxt("Mesh: generation/import failed for public type '"+type+"'");
        return false;
    }
    if (save) mesh.saveMesh(savedMeshName(m_InputFileName));
    return true;
}
