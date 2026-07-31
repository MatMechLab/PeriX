//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "InputSystem/InputSchema.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json=nlohmann::ordered_json;

struct SchemaContext {
    int meshDim=0; // zero means that an imported mesh determines it at run time
    std::vector<std::string> dofs;
    std::string elementType;
    bool explicitDynamics=false;
    bool fracture=false;
    bool coupledSpecies=false;
    bool velocityIC=false;
    bool hasPDTraction=false;
    bool hasSpeciesFlux=false;
    std::string jobType="static";
};

bool fail(std::string &error,const std::string &message) {
    error=message;
    return false;
}

bool allowedKeys(const Json &object,
                 const std::initializer_list<std::string_view> allowed,
                 const std::string &context,
                 std::string &error) {
    if (!object.is_object()) return fail(error,context+" must be a JSON object");
    for (const auto &item:object.items()) {
        const auto found=std::find(allowed.begin(),allowed.end(),item.key());
        if (found==allowed.end()) {
            return fail(error,context+": unsupported key '"+item.key()+"'");
        }
    }
    return true;
}

bool requireString(const Json &object,const char *key,
                   const std::string &context,std::string &error,
                   std::string *value=nullptr) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        return fail(error,context+": '"+key+"' is required and must be a string");
    }
    const std::string text=object.at(key).get<std::string>();
    if (text.empty()) return fail(error,context+": '"+key+"' cannot be empty");
    if (value) *value=text;
    return true;
}

bool finiteNumber(const Json &value) {
    if (!value.is_number()) return false;
    return std::isfinite(value.get<double>());
}

bool requireNumber(const Json &object,const char *key,
                   const std::string &context,std::string &error,
                   double *value=nullptr) {
    if (!object.contains(key) || !finiteNumber(object.at(key))) {
        return fail(error,context+": '"+key+"' is required and must be a finite number");
    }
    if (value) *value=object.at(key).get<double>();
    return true;
}

bool optionalNumber(const Json &object,const char *key,
                    const std::string &context,std::string &error,
                    double *value=nullptr) {
    if (!object.contains(key)) return true;
    if (!finiteNumber(object.at(key))) {
        return fail(error,context+": '"+key+"' must be a finite number");
    }
    if (value) *value=object.at(key).get<double>();
    return true;
}

bool requireBoolean(const Json &object,const char *key,
                    const std::string &context,std::string &error) {
    if (!object.contains(key) || !object.at(key).is_boolean()) {
        return fail(error,context+": '"+key+"' is required and must be true or false");
    }
    return true;
}

bool optionalBoolean(const Json &object,const char *key,
                     const std::string &context,std::string &error) {
    if (object.contains(key) && !object.at(key).is_boolean()) {
        return fail(error,context+": '"+key+"' must be true or false");
    }
    return true;
}

bool integerInRange(const Json &value,const std::int64_t minimum,
                    const std::uint64_t maximum,std::uint64_t &out) {
    if (value.is_number_unsigned()) {
        const auto n=value.get<std::uint64_t>();
        if (n>maximum) return false;
        out=n;
        return true;
    }
    if (!value.is_number_integer()) return false;
    const auto n=value.get<std::int64_t>();
    if (n<minimum || static_cast<std::uint64_t>(n)>maximum) return false;
    out=static_cast<std::uint64_t>(n);
    return true;
}

bool requireInteger(const Json &object,const char *key,
                    const std::int64_t minimum,const std::uint64_t maximum,
                    const std::string &context,std::string &error,
                    std::uint64_t *value=nullptr) {
    if (!object.contains(key)) {
        return fail(error,context+": '"+key+"' is required and must be an integer");
    }
    std::uint64_t parsed=0;
    if (!integerInRange(object.at(key),minimum,maximum,parsed)) {
        return fail(error,context+": '"+key+"' is outside the allowed integer range");
    }
    if (value) *value=parsed;
    return true;
}

bool numberArray(const Json &value,const std::size_t minimum,
                 const std::size_t maximum,const std::string &context,
                 std::string &error,std::vector<double> *out=nullptr) {
    if (!value.is_array() || value.size()<minimum || value.size()>maximum) {
        return fail(error,context+" must be an array of "
                         +std::to_string(minimum)
                         +(minimum==maximum ? "" : " to "+std::to_string(maximum))
                         +" finite number(s)");
    }
    std::vector<double> parsed;
    parsed.reserve(value.size());
    for (std::size_t i=0;i<value.size();++i) {
        if (!finiteNumber(value[i])) {
            return fail(error,context+"["+std::to_string(i)+"] must be a finite number");
        }
        parsed.push_back(value[i].get<double>());
    }
    if (out) *out=std::move(parsed);
    return true;
}

bool scalarOrArray(const Json &object,const char *key,const std::size_t dofCount,
                   const std::string &context,std::string &error,
                   std::vector<double> *out=nullptr) {
    if (!object.contains(key)) {
        return fail(error,context+": '"+key+"' is required");
    }
    const auto &entry=object.at(key);
    std::vector<double> parsed;
    if (finiteNumber(entry)) {
        parsed.push_back(entry.get<double>());
    }
    else if (entry.is_array() && !entry.empty()) {
        for (std::size_t i=0;i<entry.size();++i) {
            if (!finiteNumber(entry[i])) {
                return fail(error,context+": '"+key+"' entries must be finite numbers");
            }
            parsed.push_back(entry[i].get<double>());
        }
    }
    else {
        return fail(error,context+": '"+key+"' must be a finite number or a non-empty array");
    }
    if (parsed.size()!=1 && parsed.size()!=dofCount) {
        return fail(error,context+": '"+key+"' must contain one value or one per target DoF");
    }
    if (out) *out=std::move(parsed);
    return true;
}

bool validateMesh(const Json &mesh,SchemaContext &context,std::string &error) {
    if (!mesh.is_object()) return fail(error,"Mesh must be a JSON object");
    std::string type;
    if (!requireString(mesh,"type","Mesh",error,&type)) return false;

    if (type=="perix") {
        if (!allowedKeys(mesh,{"type","dim","meshtype","nx","ny","nz",
                               "xmin","xmax","ymin","ymax","zmin","zmax",
                               "savemesh"},"Mesh",error)) return false;
        std::uint64_t dim=0;
        if (!requireInteger(mesh,"dim",2,3,"Mesh",error,&dim)) return false;
        context.meshDim=static_cast<int>(dim);

        std::string meshType;
        if (!requireString(mesh,"meshtype","Mesh",error,&meshType)) return false;
        if ((dim==2 && meshType!="quad4") || (dim==3 && meshType!="hex8")) {
            return fail(error,"Mesh: structured 2D uses 'quad4' and structured 3D uses 'hex8'");
        }

        constexpr auto maxCells=static_cast<std::uint64_t>(
            std::numeric_limits<int>::max());
        if (!requireInteger(mesh,"nx",1,maxCells,"Mesh",error)) return false;
        if (!requireInteger(mesh,"ny",1,maxCells,"Mesh",error)) return false;
        if (dim==3) {
            if (!requireInteger(mesh,"nz",1,maxCells,"Mesh",error)) return false;
        }
        else if (mesh.contains("nz") || mesh.contains("zmin") || mesh.contains("zmax")) {
            return fail(error,"Mesh: z settings are valid only for a 3D structured mesh");
        }

        double xmin=0.0,xmax=1.0,ymin=0.0,ymax=1.0,zmin=0.0,zmax=1.0;
        if (!optionalNumber(mesh,"xmin","Mesh",error,&xmin)
            || !optionalNumber(mesh,"xmax","Mesh",error,&xmax)
            || !optionalNumber(mesh,"ymin","Mesh",error,&ymin)
            || !optionalNumber(mesh,"ymax","Mesh",error,&ymax)) return false;
        if (dim==3 && (!optionalNumber(mesh,"zmin","Mesh",error,&zmin)
                      || !optionalNumber(mesh,"zmax","Mesh",error,&zmax))) return false;
        if (!(xmin<xmax) || !(ymin<ymax) || (dim==3 && !(zmin<zmax))) {
            return fail(error,"Mesh: every coordinate minimum must be smaller than its maximum");
        }
        return optionalBoolean(mesh,"savemesh","Mesh",error);
    }

    if (type=="gmsh") {
        if (!allowedKeys(mesh,{"type","file","savemesh"},"Mesh",error)) return false;
        if (!requireString(mesh,"file","Mesh",error)) return false;
        if (!optionalBoolean(mesh,"savemesh","Mesh",error)) return false;
        context.meshDim=0;
        return true;
    }

    if (type=="circle") {
        if (!allowedKeys(mesh,{"type","radius","layers","center","boundary","savemesh"},
                         "Mesh",error)) return false;
        double radius=0.0;
        if (!requireNumber(mesh,"radius","Mesh",error,&radius) || radius<=0.0) {
            return fail(error,"Mesh: circle 'radius' must be positive");
        }
        constexpr auto maxLayers=static_cast<std::uint64_t>(
            std::numeric_limits<int>::max());
        if (!requireInteger(mesh,"layers",1,maxLayers,"Mesh",error)) return false;
        if (mesh.contains("center")
            && !numberArray(mesh.at("center"),2,3,"Mesh.center",error)) return false;
        if (mesh.contains("boundary")) {
            const auto &boundary=mesh.at("boundary");
            if (!boundary.is_boolean()
                && (!boundary.is_string() || boundary.get<std::string>().empty())) {
                return fail(error,"Mesh: circle 'boundary' must be a boolean or non-empty group name");
            }
        }
        if (!optionalBoolean(mesh,"savemesh","Mesh",error)) return false;
        context.meshDim=2;
        return true;
    }

    return fail(error,"Mesh: public type must be 'perix', 'gmsh', or 'circle'");
}

bool validatePDMesh(const Json &pdMesh,std::string &error) {
    if (!allowedKeys(pdMesh,{"HorizonRadiusFactor","Order","VariableHorizon",
                             "ghost_layer"},"PDMesh",error)) return false;
    double factor=0.0;
    if (!requireNumber(pdMesh,"HorizonRadiusFactor","PDMesh",error,&factor)
        || factor<=0.0) {
        return fail(error,"PDMesh: 'HorizonRadiusFactor' must be positive");
    }
    std::uint64_t order=0;
    if (!requireInteger(pdMesh,"Order",1,6,"PDMesh",error,&order)) return false;
    if (order<2) {
        return fail(error,"PDMesh: the published element kernels require Order >= 2");
    }
    if (!optionalBoolean(pdMesh,"VariableHorizon","PDMesh",error)) return false;
    if (pdMesh.contains("ghost_layer")) {
        constexpr auto maximum=static_cast<std::uint64_t>(
            std::numeric_limits<int>::max());
        if (!requireInteger(pdMesh,"ghost_layer",1,maximum,"PDMesh",error)) return false;
    }
    return true;
}

// One of 'angle_degrees' / 'angle_radians', never both.
bool validatePresetAngle(const Json &item,const std::string &tag,
                         std::string &error) {
    const bool hasDegrees=item.contains("angle_degrees");
    const bool hasRadians=item.contains("angle_radians");
    if (hasDegrees && hasRadians) {
        return fail(error,tag+": use either 'angle_degrees' or 'angle_radians', not both");
    }
    if (hasDegrees && !requireNumber(item,"angle_degrees",tag,error)) return false;
    if (hasRadians && !requireNumber(item,"angle_radians",tag,error)) return false;
    return true;
}

bool validateVec3(const Json &item,const std::string &key,
                  const std::string &tag,std::string &error) {
    if (!item.contains(key)) return true;
    const auto &value=item.at(key);
    if (!value.is_array() || value.size()!=3) {
        return fail(error,tag+": '"+key+"' must be an array of three numbers");
    }
    for (const auto &component:value) {
        if (!component.is_number() || !std::isfinite(component.get<double>())) {
            return fail(error,tag+": '"+key+"' must be an array of three finite numbers");
        }
    }
    return true;
}

bool validatePresetLabel(const Json &item,const std::string &tag,
                         std::string &error) {
    if (item.contains("label")
        && (!item.at("label").is_string()
            || item.at("label").get<std::string>().empty())) {
        return fail(error,tag+": 'label' must be a non-empty string");
    }
    return true;
}

bool validateMeshModifyPresets(const Json &presets,std::string &error) {
    if (!presets.is_array() || presets.empty()) {
        return fail(error,"MeshModify: 'Presets' must be a non-empty array");
    }
    for (std::size_t i=0;i<presets.size();++i) {
        const auto &item=presets[i];
        const std::string tag="MeshModify.Presets["+std::to_string(i)+"]";
        if (!item.is_object()) return fail(error,tag+" must be an object");
        std::string type;
        if (!requireString(item,"type",tag,error,&type)) return false;
        if (!validatePresetLabel(item,tag,error)) return false;
        if (!validatePresetAngle(item,tag,error)) return false;

        double length=0.0;
        if (type=="center_crack") {
            if (!allowedKeys(item,{"type","label","angle_degrees","angle_radians",
                                   "center_x","center_y","center_z","center",
                                   "length","width","normal","axis"},tag,error)) {
                return false;
            }
            if (item.contains("center")) {
                if (item.contains("center_x") || item.contains("center_y")
                    || item.contains("center_z")) {
                    return fail(error,tag+": use either 'center' or "
                                          "'center_x'/'center_y'/'center_z', not both");
                }
                if (!validateVec3(item,"center",tag,error)) return false;
            }
            else {
                if (!requireNumber(item,"center_x",tag,error)
                    || !requireNumber(item,"center_y",tag,error)) return false;
                if (item.contains("center_z")
                    && !requireNumber(item,"center_z",tag,error)) return false;
            }
            if (!validateVec3(item,"normal",tag,error)) return false;
            if (!validateVec3(item,"axis",tag,error)) return false;
        }
        else if (type=="edge_crack") {
            if (!allowedKeys(item,{"type","label","angle_degrees","angle_radians",
                                   "side","position","length","width"},tag,error)) {
                return false;
            }
            std::string side;
            if (!requireString(item,"side",tag,error,&side)) return false;
            if (side!="left" && side!="right" && side!="bottom" && side!="top") {
                return fail(error,tag+": 'side' must be 'left', 'right', 'bottom' or 'top'");
            }
            if (!requireNumber(item,"position",tag,error)) return false;
        }
        else {
            return fail(error,tag+": unsupported preset type '"+type
                              +"' (supported: center_crack, edge_crack)");
        }

        if (!requireNumber(item,"length",tag,error,&length)) return false;
        if (!(length>0.0)) return fail(error,tag+": 'length' must be a positive number");
        if (item.contains("width")) {
            double width=0.0;
            if (!requireNumber(item,"width",tag,error,&width)) return false;
            if (!(width>0.0)) return fail(error,tag+": 'width' must be a positive number");
        }
    }
    return true;
}

bool validateMeshModify(const Json &block,std::string &error) {
    if (!allowedKeys(block,{"type","treatment","Cracks","Presets"},"MeshModify",error)) return false;
    std::string type,treatment;
    if (!requireString(block,"type","MeshModify",error,&type)
        || !requireString(block,"treatment","MeshModify",error,&treatment)) return false;
    if (type!="pre_existing_cracks") {
        return fail(error,"MeshModify: public type must be 'pre_existing_cracks'");
    }
    if (treatment!="force_only") {
        return fail(error,"MeshModify: the published treatment is 'force_only'");
    }
    const bool hasCracks=block.contains("Cracks");
    const bool hasPresets=block.contains("Presets");
    if (!hasCracks && !hasPresets) {
        return fail(error,"MeshModify: give at least one of 'Cracks' or 'Presets'");
    }
    if (hasPresets && !validateMeshModifyPresets(block.at("Presets"),error)) {
        return false;
    }
    if (!hasCracks) return true;
    if (!block.at("Cracks").is_array() || block.at("Cracks").empty()) {
        return fail(error,"MeshModify: 'Cracks' must be a non-empty array");
    }
    const auto &cracks=block.at("Cracks");
    for (std::size_t i=0;i<cracks.size();++i) {
        const auto &crack=cracks[i];
        const std::string tag="MeshModify.Cracks["+std::to_string(i)+"]";
        if (!allowedKeys(crack,{"x1","y1","x2","y2","label"},tag,error)) return false;
        double x1=0.0,y1=0.0,x2=0.0,y2=0.0;
        if (!requireNumber(crack,"x1",tag,error,&x1)
            || !requireNumber(crack,"y1",tag,error,&y1)
            || !requireNumber(crack,"x2",tag,error,&x2)
            || !requireNumber(crack,"y2",tag,error,&y2)) return false;
        if (x1==x2 && y1==y2) return fail(error,tag+": crack endpoints must differ");
        if (crack.contains("label")
            && (!crack.at("label").is_string()
                || crack.at("label").get<std::string>().empty())) {
            return fail(error,tag+": 'label' must be a non-empty string");
        }
    }
    return true;
}

bool validateDofs(const Json &block,SchemaContext &context,std::string &error) {
    if (!block.is_array() || block.empty()) {
        return fail(error,"DOFs must be a non-empty array of field-name strings");
    }
    std::set<std::string> unique;
    context.dofs.clear();
    for (std::size_t i=0;i<block.size();++i) {
        if (!block[i].is_string() || block[i].get<std::string>().empty()) {
            return fail(error,"DOFs["+std::to_string(i)+"] must be a non-empty string");
        }
        const std::string name=block[i].get<std::string>();
        if (!unique.insert(name).second) {
            return fail(error,"DOFs: duplicate field name '"+name+"'");
        }
        context.dofs.push_back(name);
    }
    return true;
}

bool requirePositive(const Json &params,const char *key,const std::string &tag,
                     std::string &error,double *out=nullptr) {
    double value=0.0;
    if (!requireNumber(params,key,tag,error,&value)) return false;
    if (value<=0.0) return fail(error,tag+": '"+key+"' must be positive");
    if (out) *out=value;
    return true;
}

bool validatePlaneState(const Json &params,const int meshDim,
                        const std::string &tag,std::string &error,
                        const bool requiredIn2D=true) {
    if (meshDim>=3) {
        if (params.contains("state")) {
            return fail(error,tag+": 'state' is a 2D plane reduction and is invalid in 3D");
        }
        return true;
    }
    if (!params.contains("state")) {
        if (meshDim==2 && requiredIn2D) {
            return fail(error,tag+": 'state' is required for a 2D mechanical model");
        }
        return true;
    }
    if (!params.at("state").is_string()) {
        return fail(error,tag+": 'state' must be 'plane_stress' or 'plane_strain'");
    }
    const auto state=params.at("state").get<std::string>();
    if (state!="plane_stress" && state!="plane_strain") {
        return fail(error,tag+": 'state' must be 'plane_stress' or 'plane_strain'");
    }
    return true;
}

bool sameDofs(const std::vector<std::string> &actual,
              const std::initializer_list<std::string_view> expected) {
    if (actual.size()!=expected.size()) return false;
    std::size_t i=0;
    for (const auto name:expected) {
        if (actual[i++]!=name) return false;
    }
    return true;
}

bool validateElementDofs(const std::string &type,SchemaContext &context,
                         const std::string &tag,std::string &error) {
    bool match=false;
    if (type=="poisson") match=sameDofs(context.dofs,{"u"});
    else if (type=="diffusion") match=sameDofs(context.dofs,{"c"});
    else if (type=="cahnhilliard") match=sameDofs(context.dofs,{"c","mu"});
    else if (type=="pddo_dynamic_frac" || type=="explicit_pddo_frac") {
        match=context.meshDim==3 ? sameDofs(context.dofs,{"ux","uy","uz"})
             : context.meshDim==2 ? sameDofs(context.dofs,{"ux","uy"})
             : sameDofs(context.dofs,{"ux","uy"})
               || sameDofs(context.dofs,{"ux","uy","uz"});
    }
    else if (type=="stress_cahnhilliard") {
        match=context.meshDim==3
            ? sameDofs(context.dofs,{"c","mu","ux","uy","uz"})
            : context.meshDim==2
                ? sameDofs(context.dofs,{"c","mu","ux","uy"})
                : sameDofs(context.dofs,{"c","mu","ux","uy"})
                  || sameDofs(context.dofs,{"c","mu","ux","uy","uz"});
    }
    else if (type=="frac_stress_cahnhilliard") {
        match=context.meshDim==3
            ? sameDofs(context.dofs,{"c","mu","ux","uy","uz"})
            : context.meshDim==2
                ? sameDofs(context.dofs,{"c","mu","ux","uy"})
                : sameDofs(context.dofs,{"c","mu","ux","uy"})
                  || sameDofs(context.dofs,{"c","mu","ux","uy","uz"});
    }
    if (!match) {
        return fail(error,tag+": top-level DOFs do not match this element's canonical field order");
    }
    return true;
}

bool validateElmtSystem(const Json &block,SchemaContext &context,std::string &error) {
    if (!block.is_object() || block.size()!=1) {
        return fail(error,"ElmtSystem must contain exactly one published physics element");
    }
    const auto it=block.begin();
    const std::string tag="ElmtSystem."+it.key();
    const auto &entry=it.value();
    if (!allowedKeys(entry,{"type","params"},tag,error)) return false;
    std::string type;
    if (!requireString(entry,"type",tag,error,&type)) return false;
    if (!entry.contains("params") || !entry.at("params").is_object()) {
        return fail(error,tag+": 'params' is required and must be an object");
    }
    const auto &params=entry.at("params");

    if (!validateElementDofs(type,context,tag,error)) return false;

    if (type=="poisson") {
        if (!allowedKeys(params,{"sigma","f"},tag+".params",error)) return false;
        if (!requirePositive(params,"sigma",tag+".params",error)
            || !requireNumber(params,"f",tag+".params",error)) return false;
    }
    else if (type=="diffusion") {
        if (!allowedKeys(params,{"D","f"},tag+".params",error)) return false;
        if (!requirePositive(params,"D",tag+".params",error)
            || !requireNumber(params,"f",tag+".params",error)) return false;
    }
    else if (type=="cahnhilliard") {
        if (!allowedKeys(params,{"chi","kappa","M"},tag+".params",error)) return false;
        if (!requireNumber(params,"chi",tag+".params",error)
            || !requirePositive(params,"kappa",tag+".params",error)
            || !requirePositive(params,"M",tag+".params",error)) return false;
        context.coupledSpecies=true;
    }
    else if (type=="pddo_dynamic_frac" || type=="explicit_pddo_frac") {
        if (!allowedKeys(params,{"E","nu","rho","Gc","state","tension_only","damage"},
                         tag+".params",error)) return false;
        double nu=0.0;
        if (!requirePositive(params,"E",tag+".params",error)
            || !requireNumber(params,"nu",tag+".params",error,&nu)
            || !requirePositive(params,"rho",tag+".params",error)
            || !requirePositive(params,"Gc",tag+".params",error)
            || !requireBoolean(params,"tension_only",tag+".params",error)
            || !requireBoolean(params,"damage",tag+".params",error)
            || !validatePlaneState(params,context.meshDim,tag+".params",error)) return false;
        if (!(nu>-1.0 && nu<0.5)) {
            return fail(error,tag+".params: 'nu' must lie in (-1,0.5)");
        }
        context.fracture=true;
        context.explicitDynamics=(type=="explicit_pddo_frac");
    }
    else if (type=="stress_cahnhilliard") {
        if (!allowedKeys(params,{"E","nu","D","Omega","cref","chi","kappa","state"},
                         tag+".params",error)) return false;
        double nu=0.0;
        if (!requirePositive(params,"E",tag+".params",error)
            || !requireNumber(params,"nu",tag+".params",error,&nu)
            || !requirePositive(params,"D",tag+".params",error)
            || !requireNumber(params,"Omega",tag+".params",error)
            || !requireNumber(params,"cref",tag+".params",error)
            || !requireNumber(params,"chi",tag+".params",error)
            || !requirePositive(params,"kappa",tag+".params",error)
            || !validatePlaneState(params,context.meshDim,tag+".params",error)) return false;
        if (!(nu>-1.0 && nu<0.5)) {
            return fail(error,tag+".params: 'nu' must lie in (-1,0.5)");
        }
        context.coupledSpecies=true;
    }
    else if (type=="frac_stress_cahnhilliard") {
        if (!allowedKeys(params,{"E","nu","D","Omega","cref","chi","kappa","rho",
                                 "Gc","state","tension_only","damage",
                                 "residual_stiffness"},tag+".params",error)) return false;
        double nu=0.0,rho=0.0,residual=1.0e-8;
        if (!requirePositive(params,"E",tag+".params",error)
            || !requireNumber(params,"nu",tag+".params",error,&nu)
            || !requirePositive(params,"D",tag+".params",error)
            || !requireNumber(params,"Omega",tag+".params",error)
            || !requireNumber(params,"cref",tag+".params",error)
            || !requireNumber(params,"chi",tag+".params",error)
            || !requirePositive(params,"kappa",tag+".params",error)
            || !requireNumber(params,"rho",tag+".params",error,&rho)
            || !requirePositive(params,"Gc",tag+".params",error)
            || !requireBoolean(params,"tension_only",tag+".params",error)
            || !requireBoolean(params,"damage",tag+".params",error)
            || !optionalNumber(params,"residual_stiffness",tag+".params",error,&residual)
            || !validatePlaneState(params,context.meshDim,tag+".params",error)) return false;
        if (!(nu>-1.0 && nu<0.5)) {
            return fail(error,tag+".params: 'nu' must lie in (-1,0.5)");
        }
        if (rho<0.0) return fail(error,tag+".params: 'rho' must be non-negative");
        if (!(residual>0.0 && residual<=1.0)) {
            return fail(error,tag+".params: 'residual_stiffness' must lie in (0,1]");
        }
        context.fracture=true;
        context.coupledSpecies=true;
    }
    else {
        return fail(error,tag+": public element type must be one of poisson, diffusion, "
                          "cahnhilliard, pddo_dynamic_frac, explicit_pddo_frac, "
                          "stress_cahnhilliard, frac_stress_cahnhilliard");
    }

    context.elementType=type;
    return true;
}

bool readNamedDofs(const Json &object,const char *key,
                   const SchemaContext &context,const std::string &tag,
                   std::string &error,std::vector<std::string> &out) {
    if (!object.contains(key) || !object.at(key).is_array()
        || object.at(key).empty()) {
        return fail(error,tag+": '"+key+"' must be a non-empty array of DoF names");
    }
    std::set<std::string> unique;
    out.clear();
    for (const auto &entry:object.at(key)) {
        if (!entry.is_string() || entry.get<std::string>().empty()) {
            return fail(error,tag+": '"+key+"' entries must be non-empty strings");
        }
        const std::string name=entry.get<std::string>();
        if (std::find(context.dofs.begin(),context.dofs.end(),name)==context.dofs.end()) {
            return fail(error,tag+": DoF '"+name+"' is not declared in the top-level DOFs");
        }
        if (!unique.insert(name).second) {
            return fail(error,tag+": duplicate DoF '"+name+"'");
        }
        out.push_back(name);
    }
    return true;
}

bool allZero(const std::vector<double> &values) {
    return std::all_of(values.begin(),values.end(),
                       [](const double value){ return value==0.0; });
}

bool validateBCSystem(const Json &block,SchemaContext &context,std::string &error) {
    if (!block.is_object() || block.empty()) {
        return fail(error,"BCSystem must contain at least one boundary condition");
    }
    for (auto it=block.begin();it!=block.end();++it) {
        const auto &entry=it.value();
        const std::string tag="BCSystem."+it.key();
        if (!entry.is_object()) return fail(error,tag+" must be an object");
        std::string type,phygroup;
        if (!requireString(entry,"type",tag,error,&type)
            || !requireString(entry,"phygroup",tag,error,&phygroup)) return false;

        if (type=="dirichlet" || type=="neumann" || type=="speciesflux") {
            if (type=="dirichlet") {
                if (!allowedKeys(entry,{"type","phygroup","dofs","value","direct"},
                                 tag,error)
                    || !optionalBoolean(entry,"direct",tag,error)) {
                    return false;
                }
            }
            else if (!allowedKeys(entry,{"type","phygroup","dofs","value"},
                                  tag,error)) {
                return false;
            }
            std::vector<std::string> dofs;
            if (!readNamedDofs(entry,"dofs",context,tag,error,dofs)) return false;
            std::vector<double> values;
            if (!scalarOrArray(entry,"value",dofs.size(),tag,error,&values)) return false;
            if (type=="neumann" && !allZero(values)) {
                return fail(error,tag+": the public generic Neumann condition is homogeneous; "
                                  "use the published traction or species-flux condition for a load");
            }
            if (type=="speciesflux") {
                if (dofs.size()!=1 || dofs.front()!="c") {
                    return fail(error,tag+": speciesflux must target the concentration DoF 'c'");
                }
                context.hasSpeciesFlux=true;
            }
            continue;
        }

        if (type=="pdtraction") {
            if (!allowedKeys(entry,{"type","phygroup","E","nu","state","value",
                                    "dofs","Omega","cref","c_dof"},tag,error)) return false;
            double nu=0.0;
            if (!requirePositive(entry,"E",tag,error)
                || !requireNumber(entry,"nu",tag,error,&nu)
                || !validatePlaneState(entry,context.meshDim,tag,error)) return false;
            if (!(nu>-1.0 && nu<0.5)) {
                return fail(error,tag+": 'nu' must lie in (-1,0.5)");
            }
            if (!entry.contains("value")) {
                return fail(error,tag+": 'value' is required");
            }
            const std::size_t minDim=context.meshDim==3 ? 3 : 2;
            const std::size_t maxDim=context.meshDim==2 ? 2 : 3;
            if (!numberArray(entry.at("value"),minDim,maxDim,tag+".value",error)) return false;

            std::vector<std::string> displacementDofs;
            if (entry.contains("dofs")
                && !readNamedDofs(entry,"dofs",context,tag,error,displacementDofs)) return false;
            const bool coupledLayout=!context.dofs.empty() && context.dofs.front()=="c";
            if (coupledLayout && displacementDofs.empty()) {
                return fail(error,tag+": coupled traction must name its displacement 'dofs'");
            }
            if (!displacementDofs.empty()) {
                const bool valid=context.meshDim==3
                    ? displacementDofs==std::vector<std::string>({"ux","uy","uz"})
                    : context.meshDim==2
                        ? displacementDofs==std::vector<std::string>({"ux","uy"})
                        : displacementDofs==std::vector<std::string>({"ux","uy"})
                          || displacementDofs==std::vector<std::string>({"ux","uy","uz"});
                if (!valid) {
                    return fail(error,tag+": 'dofs' must list ux,uy (and uz in 3D) in canonical order");
                }
            }

            const bool hasOmega=entry.contains("Omega");
            const bool hasCref=entry.contains("cref");
            const bool hasCDof=entry.contains("c_dof");
            if (hasOmega || hasCref || hasCDof) {
                if (!(hasOmega && hasCref && hasCDof)) {
                    return fail(error,tag+": chemical traction requires Omega, cref, and c_dof together");
                }
                if (!finiteNumber(entry.at("Omega")) || !finiteNumber(entry.at("cref"))
                    || !entry.at("c_dof").is_string()) {
                    return fail(error,tag+": Omega and cref must be numbers and c_dof a field name");
                }
                const std::string cName=entry.at("c_dof").get<std::string>();
                if (cName!="c"
                    || std::find(context.dofs.begin(),context.dofs.end(),cName)==context.dofs.end()) {
                    return fail(error,tag+": c_dof must name the declared concentration field 'c'");
                }
            }
            context.hasPDTraction=true;
            continue;
        }

        return fail(error,tag+": public BC type must be dirichlet, neumann, "
                          "pdtraction, or speciesflux");
    }
    return true;
}

bool readICDofs(const Json &entry,const SchemaContext &context,
                const std::string &tag,std::string &error,
                std::vector<std::string> &dofs) {
    const bool one=entry.contains("dof");
    const bool many=entry.contains("dofs");
    if (one==many) {
        return fail(error,tag+": specify exactly one of 'dof' or 'dofs'");
    }
    if (one) {
        if (!entry.at("dof").is_string() || entry.at("dof").get<std::string>().empty()) {
            return fail(error,tag+": 'dof' must be a declared field name");
        }
        const auto name=entry.at("dof").get<std::string>();
        if (std::find(context.dofs.begin(),context.dofs.end(),name)==context.dofs.end()) {
            return fail(error,tag+": DoF '"+name+"' is not declared in the top-level DOFs");
        }
        dofs={name};
        return true;
    }
    return readNamedDofs(entry,"dofs",context,tag,error,dofs);
}

bool validateICSystem(const Json &block,SchemaContext &context,std::string &error) {
    if (!block.is_object() || block.empty()) {
        return fail(error,"ICSystem must contain at least one initial condition");
    }
    for (auto it=block.begin();it!=block.end();++it) {
        const auto &entry=it.value();
        const std::string tag="ICSystem."+it.key();
        if (!entry.is_object()) return fail(error,tag+" must be an object");
        std::string type;
        if (!requireString(entry,"type",tag,error,&type)) return false;

        std::string field="solution";
        if (entry.contains("field")) {
            if (!entry.at("field").is_string()) {
                return fail(error,tag+": 'field' must be 'solution' or 'velocity'");
            }
            field=entry.at("field").get<std::string>();
            if (field!="solution" && field!="velocity") {
                return fail(error,tag+": 'field' must be 'solution' or 'velocity'");
            }
        }
        if (entry.contains("phygroup")
            && (!entry.at("phygroup").is_string()
                || entry.at("phygroup").get<std::string>().empty())) {
            return fail(error,tag+": 'phygroup' must be a non-empty string");
        }

        std::vector<std::string> dofs;
        if (!readICDofs(entry,context,tag,error,dofs)) return false;
        if (type=="constant") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs","value"},
                             tag,error)
                || !scalarOrArray(entry,"value",dofs.size(),tag,error)) return false;
        }
        else if (type=="linear") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "offset","slope_x","slope_y","slope_z"},tag,error)
                || !scalarOrArray(entry,"offset",dofs.size(),tag,error)) return false;
            for (const char *key:{"slope_x","slope_y","slope_z"}) {
                if (entry.contains(key)
                    && !scalarOrArray(entry,key,dofs.size(),tag,error)) return false;
            }
        }
        else if (type=="box") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "min_corner","max_corner","inside_values",
                                    "outside_values"},tag,error)) return false;
            std::vector<double> minimum,maximum;
            if (!entry.contains("min_corner") || !entry.contains("max_corner")
                || !numberArray(entry.at("min_corner"),2,3,tag+".min_corner",error,&minimum)
                || !numberArray(entry.at("max_corner"),2,3,tag+".max_corner",error,&maximum)) return false;
            if (minimum.size()!=maximum.size()) {
                return fail(error,tag+": min_corner and max_corner must have the same dimension");
            }
            for (std::size_t i=0;i<minimum.size();++i) {
                if (minimum[i]>maximum[i]) {
                    return fail(error,tag+": every min_corner coordinate must not exceed max_corner");
                }
            }
            if (!scalarOrArray(entry,"inside_values",dofs.size(),tag,error)
                || !scalarOrArray(entry,"outside_values",dofs.size(),tag,error)) return false;
        }
        else if (type=="circle") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "center","radius","dr","inside_values",
                                    "outside_values"},tag,error)) return false;
            double radius=0.0,dr=0.0;
            if (!entry.contains("center")
                || !numberArray(entry.at("center"),2,3,tag+".center",error)
                || !requireNumber(entry,"radius",tag,error,&radius)
                || !requireNumber(entry,"dr",tag,error,&dr)
                || radius<=0.0 || dr<0.0) {
                if (error.empty()) error=tag+": radius must be positive and dr non-negative";
                return false;
            }
            if (!scalarOrArray(entry,"inside_values",dofs.size(),tag,error)
                || !scalarOrArray(entry,"outside_values",dofs.size(),tag,error)) return false;
        }
        else if (type=="ellipse") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "center","semi_axes","inside_values",
                                    "outside_values","smoothness"},tag,error)) return false;
            std::vector<double> axes;
            if (!entry.contains("center") || !entry.contains("semi_axes")
                || !numberArray(entry.at("center"),2,3,tag+".center",error)
                || !numberArray(entry.at("semi_axes"),2,3,tag+".semi_axes",error,&axes)) return false;
            if (std::any_of(axes.begin(),axes.end(),[](double a){ return a<=0.0; })) {
                return fail(error,tag+": every supplied semi-axis must be positive");
            }
            double smoothness=0.0;
            if (!optionalNumber(entry,"smoothness",tag,error,&smoothness)
                || smoothness<0.0) {
                return fail(error,tag+": 'smoothness' must be non-negative");
            }
            if (!scalarOrArray(entry,"inside_values",dofs.size(),tag,error)
                || !scalarOrArray(entry,"outside_values",dofs.size(),tag,error)) return false;
        }
        else if (type=="gaussian") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "center","sigma","amplitude","offset"},tag,error)) return false;
            std::vector<double> sigma;
            if (!entry.contains("center") || !entry.contains("sigma")
                || !numberArray(entry.at("center"),2,3,tag+".center",error)
                || !numberArray(entry.at("sigma"),2,3,tag+".sigma",error,&sigma)) return false;
            if (std::any_of(sigma.begin(),sigma.end(),[](double s){ return s<=0.0; })) {
                return fail(error,tag+": every supplied sigma must be positive");
            }
            if (!scalarOrArray(entry,"amplitude",dofs.size(),tag,error)) return false;
            if (entry.contains("offset")
                && !scalarOrArray(entry,"offset",dofs.size(),tag,error)) return false;
        }
        else if (type=="cosine") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "wavenumber","phase","amplitude","offset"},tag,error)) return false;
            if (!entry.contains("wavenumber")
                || !numberArray(entry.at("wavenumber"),2,3,tag+".wavenumber",error)) return false;
            if (entry.contains("phase")
                && !numberArray(entry.at("phase"),2,3,tag+".phase",error)) return false;
            if (!scalarOrArray(entry,"amplitude",dofs.size(),tag,error)) return false;
            if (entry.contains("offset")
                && !scalarOrArray(entry,"offset",dofs.size(),tag,error)) return false;
        }
        else if (type=="random") {
            if (!allowedKeys(entry,{"type","field","phygroup","dof","dofs",
                                    "min","max","seed"},tag,error)) return false;
            double minimum=0.0,maximum=0.0;
            if (!requireNumber(entry,"min",tag,error,&minimum)
                || !requireNumber(entry,"max",tag,error,&maximum)
                || minimum>=maximum) {
                if (error.empty()) error=tag+": 'min' must be smaller than 'max'";
                return false;
            }
            if (!entry.contains("seed")) {
                return fail(error,tag+": seeded random perturbations require an explicit 'seed'");
            }
            std::uint64_t seed=0;
            if (!integerInRange(entry.at("seed"),0,
                                std::numeric_limits<std::uint64_t>::max(),seed)) {
                return fail(error,tag+": 'seed' must be a non-negative integer");
            }
        }
        else {
            return fail(error,tag+": public IC type must be constant, linear, box, circle, "
                              "ellipse, gaussian, cosine, or random");
        }

        if (field=="velocity") context.velocityIC=true;
    }
    return true;
}

bool validateJob(const Json &block,SchemaContext &context,std::string &error) {
    if (!allowedKeys(block,{"type","assemble"},"JobSystem",error)) return false;
    if (!requireString(block,"type","JobSystem",error,&context.jobType)) return false;
    if (context.jobType!="static" && context.jobType!="transient") {
        return fail(error,"JobSystem: 'type' must be 'static' or 'transient'");
    }
    if (block.contains("assemble")) {
        if (!block.at("assemble").is_string()) {
            return fail(error,"JobSystem: 'assemble' must be a string");
        }
        const std::string assemble=block.at("assemble").get<std::string>();
        if (assemble!="serial" && assemble!="openmp" && assemble!="cuda") {
            return fail(error,"JobSystem: 'assemble' must be 'serial', 'openmp', or 'cuda'");
        }
    }
    return true;
}

bool validateTimeStepping(const Json &block,std::string &error) {
    if (!allowedKeys(block,{"dt","total_time","verbose","adaptive","optimal_iters",
                            "growth_factor","cutback_factor","max_cutbacks",
                            "min_dt","max_dt"},"TimeStepping",error)) return false;
    double dt=0.0,total=0.0;
    if (!requireNumber(block,"dt","TimeStepping",error,&dt) || dt<=0.0) {
        return fail(error,"TimeStepping: 'dt' must be positive");
    }
    if (!requireNumber(block,"total_time","TimeStepping",error,&total) || total<=0.0) {
        return fail(error,"TimeStepping: 'total_time' must be positive");
    }
    if (!optionalBoolean(block,"verbose","TimeStepping",error)
        || !optionalBoolean(block,"adaptive","TimeStepping",error)) return false;
    constexpr auto maximum=static_cast<std::uint64_t>(
        std::numeric_limits<int>::max());
    if (block.contains("optimal_iters")
        && !requireInteger(block,"optimal_iters",1,maximum,"TimeStepping",error)) return false;
    if (block.contains("max_cutbacks")
        && !requireInteger(block,"max_cutbacks",0,maximum,"TimeStepping",error)) return false;

    double growth=1.25,cutback=0.5,minDt=0.0,maxDt=0.0;
    if (!optionalNumber(block,"growth_factor","TimeStepping",error,&growth)
        || !optionalNumber(block,"cutback_factor","TimeStepping",error,&cutback)
        || !optionalNumber(block,"min_dt","TimeStepping",error,&minDt)
        || !optionalNumber(block,"max_dt","TimeStepping",error,&maxDt)) return false;
    if (growth<1.0) return fail(error,"TimeStepping: 'growth_factor' must be >= 1");
    if (!(cutback>0.0 && cutback<1.0)) {
        return fail(error,"TimeStepping: 'cutback_factor' must lie in (0,1)");
    }
    if (minDt<0.0 || maxDt<0.0 || (maxDt>0.0 && minDt>maxDt)) {
        return fail(error,"TimeStepping: invalid min_dt/max_dt bounds");
    }
    return true;
}

bool validateNonlinearSolver(const Json &block,std::string &error) {
    if (!allowedKeys(block,{"max_iters","abs_tol","rel_tol","verbose"},
                     "NonlinearSolver",error)) return false;
    constexpr auto maximum=static_cast<std::uint64_t>(
        std::numeric_limits<int>::max());
    if (block.contains("max_iters")
        && !requireInteger(block,"max_iters",1,maximum,"NonlinearSolver",error)) return false;
    for (const char *key:{"abs_tol","rel_tol"}) {
        if (block.contains(key)) {
            double value=0.0;
            if (!requireNumber(block,key,"NonlinearSolver",error,&value) || value<=0.0) {
                return fail(error,"NonlinearSolver: '"+std::string(key)+"' must be positive");
            }
        }
    }
    return optionalBoolean(block,"verbose","NonlinearSolver",error);
}

bool validateLinearSolver(const Json &block,std::string &error) {
    if (!allowedKeys(block,{"type","params"},"LinearSolver",error)) return false;
    std::string type="default";
    if (block.contains("type")) {
        if (!requireString(block,"type","LinearSolver",error,&type)) return false;
    }
    if (type!="default" && type!="pardiso" && type!="cudss"
        && type!="amgcl") {
        return fail(error,
            "LinearSolver: 'type' must be 'default', 'pardiso', 'cudss', or 'amgcl'");
    }
    Json params=Json::object();
    if (block.contains("params")) {
        if (!block.at("params").is_object()) {
            return fail(error,"LinearSolver: 'params' must be an object");
        }
        params=block.at("params");
    }
    if (type=="default") {
        if (!allowedKeys(params,{},"LinearSolver.params",error)) return false;
    }
    else if (type=="pardiso") {
        if (!allowedKeys(params,{"msglvl"},"LinearSolver.params",error)) return false;
        if (params.contains("msglvl")) {
            constexpr auto maximum=static_cast<std::uint64_t>(
                std::numeric_limits<int>::max());
            if (!requireInteger(params,"msglvl",0,maximum,"LinearSolver.params",error)) return false;
        }
    }
    else if (type=="amgcl") {
        if (!allowedKeys(params,{"tol","maxiter","verbose","iluk_level","solver_threads"},
                         "LinearSolver.params",error)) return false;
        if (params.contains("tol")) {
            double tolerance=0.0;
            if (!requireNumber(params,"tol","LinearSolver.params",error,&tolerance)
                || tolerance<=0.0) {
                return fail(error,
                    "LinearSolver.params: 'tol' must be a positive finite number");
            }
        }
        if (params.contains("maxiter")) {
            constexpr auto maximum=static_cast<std::uint64_t>(
                std::numeric_limits<int>::max());
            if (!requireInteger(params,"maxiter",1,maximum,
                                "LinearSolver.params",error)) return false;
        }
        if (params.contains("iluk_level")) {
            if (!requireInteger(params,"iluk_level",1,6,
                                "LinearSolver.params",error)) return false;
        }
        if (params.contains("solver_threads")) {
            constexpr auto maximum=static_cast<std::uint64_t>(
                std::numeric_limits<int>::max());
            if (!requireInteger(params,"solver_threads",0,maximum,
                                "LinearSolver.params",error)) return false;
        }
        if (!optionalBoolean(params,"verbose","LinearSolver.params",error)) {
            return false;
        }
    }
    else if (!allowedKeys(params,{},"LinearSolver.params",error)) {
        return false;
    }
    return true;
}

bool validateOutput(const Json &block,std::string &error) {
    if (!allowedKeys(block,{"interval","Fields","format"},"OutputSystem",error)) return false;
    if (block.contains("interval")) {
        constexpr auto maximum=static_cast<std::uint64_t>(
            std::numeric_limits<int>::max());
        if (!requireInteger(block,"interval",1,maximum,"OutputSystem",error)) return false;
    }
    if (block.contains("Fields")) {
        const auto &fields=block.at("Fields");
        if (!fields.is_array()) return fail(error,"OutputSystem.Fields must be an array");
        std::set<std::string> unique;
        for (std::size_t i=0;i<fields.size();++i) {
            if (!fields[i].is_string() || fields[i].get<std::string>().empty()) {
                return fail(error,"OutputSystem.Fields entries must be non-empty strings");
            }
            if (!unique.insert(fields[i].get<std::string>()).second) {
                return fail(error,"OutputSystem.Fields contains a duplicate field");
            }
        }
    }
    if (block.contains("format")) {
        if (!block.at("format").is_string()) {
            return fail(error,"OutputSystem: 'format' must be a string");
        }
        const auto format=block.at("format").get<std::string>();
        if (format!="vtu" && format!="exodus" && format!="both") {
            return fail(error,"OutputSystem: 'format' must be 'vtu', 'exodus', or 'both'");
        }
    }
    return true;
}
} // namespace

bool InputSchema::validate(const nlohmann::ordered_json &document,
                           std::string &error,const bool meshOnly) {
    error.clear();
    if (!document.is_object()) return fail(error,"input root must be a JSON object");
    if (!allowedKeys(document,{"Mesh","PDMesh","MeshModify","DOFs","ElmtSystem",
                               "BCSystem","ICSystem","JobSystem","TimeStepping",
                               "NonlinearSolver","LinearSolver","OutputSystem"},
                     "input root",error)) return false;
    if (!document.contains("Mesh")) return fail(error,"input requires a top-level 'Mesh' block");

    if (!meshOnly) {
        for (const char *required:{"PDMesh","DOFs","ElmtSystem","BCSystem"}) {
            if (!document.contains(required)) {
                return fail(error,"input requires a top-level '"+std::string(required)+"' block");
            }
        }
    }

    SchemaContext context;
    if (!validateMesh(document.at("Mesh"),context,error)) return false;
    if (document.contains("PDMesh")
        && !validatePDMesh(document.at("PDMesh"),error)) return false;
    if (document.contains("MeshModify")
        && !validateMeshModify(document.at("MeshModify"),error)) return false;
    if (document.contains("DOFs")
        && !validateDofs(document.at("DOFs"),context,error)) return false;

    if (document.contains("ElmtSystem")) {
        if (context.dofs.empty()) {
            return fail(error,"ElmtSystem requires the top-level symbolic DOFs block");
        }
        if (!validateElmtSystem(document.at("ElmtSystem"),context,error)) return false;
    }
    if (document.contains("BCSystem")) {
        if (context.dofs.empty()) {
            return fail(error,"BCSystem requires the top-level symbolic DOFs block");
        }
        if (!validateBCSystem(document.at("BCSystem"),context,error)) return false;
    }
    if (document.contains("ICSystem")) {
        if (context.dofs.empty()) {
            return fail(error,"ICSystem requires the top-level symbolic DOFs block");
        }
        if (!validateICSystem(document.at("ICSystem"),context,error)) return false;
    }
    if (document.contains("JobSystem")
        && !validateJob(document.at("JobSystem"),context,error)) return false;
    if (document.contains("TimeStepping")
        && !validateTimeStepping(document.at("TimeStepping"),error)) return false;
    if (document.contains("NonlinearSolver")
        && !validateNonlinearSolver(document.at("NonlinearSolver"),error)) return false;
    if (document.contains("LinearSolver")
        && !validateLinearSolver(document.at("LinearSolver"),error)) return false;
    if (document.contains("OutputSystem")
        && !validateOutput(document.at("OutputSystem"),error)) return false;

    if (context.jobType=="transient" && !document.contains("TimeStepping")) {
        return fail(error,"a transient JobSystem requires TimeStepping");
    }
    if (context.jobType=="static" && document.contains("TimeStepping")) {
        return fail(error,"TimeStepping is valid only for a transient JobSystem");
    }
    if (context.explicitDynamics && context.jobType!="transient") {
        return fail(error,"explicit_pddo_frac requires a transient JobSystem");
    }
    if (context.explicitDynamics && (document.contains("LinearSolver")
                                     || document.contains("NonlinearSolver"))) {
        return fail(error,"explicit_pddo_frac is matrix-free and does not accept solver blocks");
    }
    if (context.velocityIC && !context.explicitDynamics) {
        return fail(error,"velocity initial conditions are supported by the published "
                          "central-difference element only");
    }
    if (context.explicitDynamics && context.hasPDTraction) {
        return fail(error,"pdtraction requires an assembled linear system and is invalid "
                          "for the matrix-free explicit element");
    }
    if (context.hasSpeciesFlux && !context.coupledSpecies) {
        return fail(error,"speciesflux requires a published species element");
    }
    if (document.contains("MeshModify") && !context.fracture) {
        return fail(error,"MeshModify pre-existing cracks require a published fracture element");
    }
    return true;
}
