//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "InputSystem/InputSchema.h"

namespace {
using Json=nlohmann::ordered_json;
int failures=0;

void expectAccepted(const std::string &name,const Json &document) {
    std::string error;
    if (InputSchema::validate(document,error)) return;
    std::printf("FAIL accept %-30s %s\n",name.c_str(),error.c_str());
    ++failures;
}

void expectRejected(const std::string &name,const Json &document) {
    std::string error;
    if (!InputSchema::validate(document,error) && !error.empty()) return;
    std::printf("FAIL reject %-30s\n",name.c_str());
    ++failures;
}

Json load(const std::string &fileName) {
    std::ifstream stream(fileName);
    Json document;
    stream >> document;
    return document;
}

Json* findWith(std::vector<Json> &documents,const char *block,const char *member) {
    for (auto &document:documents) {
        if (document.contains(block) && document.at(block).contains(member)) {
            return &document;
        }
    }
    return nullptr;
}
}

int main(int argc,char *argv[]) {
    if (argc<2) {
        std::printf("usage: InputSystemSchemaTests examples/*.json\n");
        return 2;
    }

    std::vector<Json> documents;
    for (int i=1;i<argc;++i) {
        documents.push_back(load(argv[i]));
        expectAccepted(argv[i],documents.back());
    }

    Json changed=documents.front();
    changed["NotPublic"]=Json::object();
    expectRejected("unknown top-level block",changed);

    changed=documents.front();
    changed["Mesh"]["not_public_option"]=true;
    expectRejected("unknown mesh option",changed);

    changed=documents.front();
    changed["ElmtSystem"].begin().value()["type"]="not_public";
    expectRejected("unknown element type",changed);

    changed=documents.front();
    changed["ElmtSystem"].begin().value()["params"]["not_public_option"]=1.0;
    expectRejected("unknown element parameter",changed);

    changed=documents.front();
    changed["BCSystem"].begin().value()["type"]="not_public";
    expectRejected("unknown boundary type",changed);

    changed=documents.front();
    changed["BCSystem"].begin().value()["dofs"]=Json::array({1});
    expectRejected("numeric boundary DoF",changed);

    changed=documents.front();
    changed["BCSystem"].begin().value()["not_public_option"]=0.0;
    expectRejected("unknown boundary option",changed);

    if (Json *spinodal=findWith(documents,"ICSystem","noisy_half")) {
        changed=*spinodal;
        changed["ICSystem"]["noisy_half"].erase("seed");
        expectRejected("unseeded random IC",changed);

        changed=*spinodal;
        changed["ICSystem"]["noisy_half"]["seed"]=0;
        expectAccepted("deterministic zero seed",changed);

    }
    else {
        std::printf("FAIL did not find seeded-random example\n");
        ++failures;
    }

    if (Json *diffusion=findWith(documents,"ICSystem","bump")) {
        changed=*diffusion;
        changed["BCSystem"]["left"]["value"]=1.0;
        expectRejected("nonzero generic Neumann",changed);

        changed=*diffusion;
        changed["BCSystem"]["bottom"]["velocity"]=1.0;
        expectRejected("sustained boundary velocity",changed);
    }
    else {
        std::printf("FAIL did not find diffusion example\n");
        ++failures;
    }

    if (Json *impact=findWith(documents,"ICSystem","impact_initial_velocity")) {
        changed=*impact;
        changed["LinearSolver"]={{"type","default"}};
        expectRejected("solver on explicit run",changed);
    }
    else {
        std::printf("FAIL did not find impact example\n");
        ++failures;
    }

    changed=documents.front();
    changed.erase("DOFs");
    expectRejected("missing symbolic DOFs",changed);

    if (Json *withOutput=findWith(documents,"OutputSystem","interval")) {
        changed=*withOutput;
        changed["OutputSystem"]["not_public_option"]="all";
        expectRejected("unknown output option",changed);
    }

    if (Json *withSolver=findWith(documents,"LinearSolver","type")) {
        changed=*withSolver;
        changed["LinearSolver"]["type"]="not_public";
        expectRejected("unknown solver type",changed);
    }

    changed=documents.front();
    changed["JobSystem"]["not_public_option"]=true;
    expectRejected("unknown job option",changed);

    changed=documents.front();
    changed["JobSystem"]["type"]="not_public";
    expectRejected("unknown job type",changed);

    changed=documents.front();
    changed["JobSystem"]["assemble"]="not_public";
    expectRejected("unknown assembly backend",changed);

    changed=documents.front();
    changed.erase("TimeStepping");
    expectRejected("transient job without time stepping",changed);

    if (Json *fracture=findWith(documents,"MeshModify","Cracks")) {
        changed=*fracture;
        changed["MeshModify"]["type"]="not_public";
        expectRejected("unknown mesh modification",changed);

        changed=*fracture;
        changed["MeshModify"]["treatment"]="not_public";
        expectRejected("unknown crack treatment",changed);

        changed=*fracture;
        changed["MeshModify"]["Cracks"][0]["not_public_option"]=true;
        expectRejected("unknown crack option",changed);

        changed=*fracture;
        changed["MeshModify"]["Cracks"][0]["x2"]=
            changed["MeshModify"]["Cracks"][0]["x1"];
        changed["MeshModify"]["Cracks"][0]["y2"]=
            changed["MeshModify"]["Cracks"][0]["y1"];
        expectRejected("zero-length crack",changed);

        changed=*fracture;
        changed["MeshModify"]["Cracks"]=Json::array();
        expectRejected("empty crack list",changed);
    }
    else {
        std::printf("FAIL did not find pre-existing-crack example\n");
        ++failures;
    }

    if (Json *diffusion=findWith(documents,"ICSystem","bump")) {
        changed=*diffusion;
        changed["MeshModify"]={
            {"type","pre_existing_cracks"},
            {"treatment","force_only"},
            {"Cracks",Json::array({{
                {"x1",0.2},{"y1",0.2},{"x2",0.8},{"y2",0.2}
            }})}
        };
        expectRejected("crack on non-fracture element",changed);
    }

    if (Json *diffusion=findWith(documents,"ICSystem","bump")) {
        const std::vector<std::pair<std::string,Json>> profiles={
            {"constant",{{"type","constant"},{"dof","c"},{"value",0.2}}},
            {"linear",{{"type","linear"},{"dof","c"},{"offset",0.2},
                       {"slope_x",1.0},{"slope_y",-0.5}}},
            {"box",{{"type","box"},{"dof","c"},
                    {"min_corner",Json::array({0.1,0.1})},
                    {"max_corner",Json::array({0.4,0.4})},
                    {"inside_values",1.0},{"outside_values",0.0}}},
            {"circle",{{"type","circle"},{"dof","c"},
                       {"center",Json::array({0.5,0.5})},{"radius",0.2},{"dr",0.02},
                       {"inside_values",1.0},{"outside_values",0.0}}},
            {"ellipse",{{"type","ellipse"},{"dof","c"},
                        {"center",Json::array({0.5,0.5})},
                        {"semi_axes",Json::array({0.2,0.1})},
                        {"inside_values",1.0},{"outside_values",0.0},
                        {"smoothness",4.0}}},
            {"gaussian",{{"type","gaussian"},{"dof","c"},
                         {"center",Json::array({0.5,0.5})},
                         {"sigma",Json::array({0.1,0.2})},
                         {"amplitude",1.0},{"offset",0.0}}},
            {"cosine",{{"type","cosine"},{"dof","c"},
                       {"wavenumber",Json::array({6.0,6.0})},
                       {"phase",Json::array({0.0,0.0})},
                       {"amplitude",0.01},{"offset",0.5}}},
            {"random",{{"type","random"},{"dof","c"},
                       {"min",0.49},{"max",0.51},{"seed",17}}}
        };
        for (const auto &[name,profile]:profiles) {
            changed=*diffusion;
            changed["ICSystem"]={{"profile",profile}};
            expectAccepted("published IC "+name,changed);
        }

        changed=*diffusion;
        changed["Mesh"]={{"type","perix"},{"dim",3},{"nx",2},{"ny",2},{"nz",2},
                         {"meshtype","hex8"},{"savemesh",false}};
        expectAccepted("structured hex8 mesh",changed);

        changed=*diffusion;
        changed["PDMesh"]["VariableHorizon"]=true;
        changed["PDMesh"]["ghost_layer"]=2;
        expectAccepted("published PDMesh options",changed);
    }

    if (Json *diffusion=findWith(documents,"ICSystem","bump")) {
        changed=*diffusion;
        changed["Mesh"]={{"type","circle"},{"radius",1.0},{"layers",20},
                         {"center",Json::array({0.0,0.0})},{"boundary","outer"},
                         {"savemesh",false}};
        changed["DOFs"]=Json::array({"c","mu","ux","uy"});
        changed["ElmtSystem"]={
            {"coupled",{
                {"type","frac_stress_cahnhilliard"},
                {"params",{
                    {"E",100.0},{"nu",0.25},{"D",1.0},{"Omega",3.0},
                    {"cref",0.05},{"chi",2.6},{"kappa",0.01},{"rho",0.0},
                    {"Gc",1.0},{"state","plane_stress"},
                    {"tension_only",true},{"damage",true},
                    {"residual_stiffness",0.001}
                }}
            }}
        };
        changed["BCSystem"]={
            {"influx",{{"type","speciesflux"},{"phygroup","outernodes_ghost"},
                       {"dofs",Json::array({"c"})},{"value",0.01}}},
            {"free",{{"type","pdtraction"},{"phygroup","outernodes_ghost"},
                     {"dofs",Json::array({"ux","uy"})},
                     {"E",100.0},{"nu",0.25},{"state","plane_stress"},
                     {"value",Json::array({0.0,0.0})},
                     {"Omega",3.0},{"cref",0.05},{"c_dof","c"}}}
        };
        changed["ICSystem"]={
            {"c0",{{"type","constant"},{"dof","c"},{"value",0.05}}},
            {"mu0",{{"type","constant"},{"dof","mu"},{"value",0.0}}},
            {"u0",{{"type","constant"},{"dofs",Json::array({"ux","uy"})},
                   {"value",Json::array({0.0,0.0})}}}
        };
        changed["JobSystem"]={{"type","transient"},{"assemble","serial"}};
        changed["LinearSolver"]={{"type","default"}};
        changed["OutputSystem"]={{"interval",10},{"format","both"},
                                 {"Fields",Json::array({"damage","stress"})}};
        expectAccepted("coupled manuscript model",changed);
    }

    if (failures==0) {
        std::printf("InputSystem schema tests passed (%d public decks)\n",argc-1);
        return 0;
    }
    std::printf("InputSystem schema tests failed: %d\n",failures);
    return 1;
}
