//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include "OutputSystem/ExodusWriter.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "Mesh/MeshData.h"

namespace {
constexpr std::uint32_t dimensionTag=10;
constexpr std::uint32_t variableTag=11;
constexpr std::uint32_t attributeTag=12;
constexpr std::uint32_t nameLength=33;
constexpr std::uint32_t qaStringLength=33;

[[nodiscard]] std::uint64_t aligned(const std::uint64_t size) {
    return (size+3U)&~std::uint64_t{3};
}

class BinaryBlock {
public:
    void byte(const std::uint8_t value) {
        m_Bytes.push_back(static_cast<char>(value));
    }

    void unsigned32(const std::uint32_t value) {
        m_Bytes.push_back(static_cast<char>((value>>24U)&0xffU));
        m_Bytes.push_back(static_cast<char>((value>>16U)&0xffU));
        m_Bytes.push_back(static_cast<char>((value>>8U)&0xffU));
        m_Bytes.push_back(static_cast<char>(value&0xffU));
    }

    void unsigned64(const std::uint64_t value) {
        unsigned32(static_cast<std::uint32_t>(value>>32U));
        unsigned32(static_cast<std::uint32_t>(value&0xffffffffU));
    }

    void integer(const std::int32_t value) {
        unsigned32(static_cast<std::uint32_t>(value));
    }

    void real32(const float value) {
        unsigned32(std::bit_cast<std::uint32_t>(value));
    }

    void real64(const double value) {
        unsigned64(std::bit_cast<std::uint64_t>(value));
    }

    void text(const std::string_view value) {
        m_Bytes.insert(m_Bytes.end(),value.begin(),value.end());
    }

    void zeros(const std::uint64_t count) {
        m_Bytes.insert(m_Bytes.end(),static_cast<std::size_t>(count),'\0');
    }

    void fixedText(const std::string_view value,const std::uint32_t width) {
        const std::size_t copied=std::min<std::size_t>(value.size(),width);
        m_Bytes.insert(m_Bytes.end(),value.begin(),value.begin()+copied);
        zeros(width-copied);
    }

    void name(const std::string &value) {
        unsigned32(static_cast<std::uint32_t>(value.size()));
        text(value);
        zeros(aligned(value.size())-value.size());
    }

    void pad() {
        zeros(aligned(m_Bytes.size())-m_Bytes.size());
    }

    [[nodiscard]] const std::vector<char>& bytes() const noexcept {
        return m_Bytes;
    }

private:
    std::vector<char> m_Bytes;
};

enum class DataKind : std::uint32_t {
    Character=2,
    Integer=4,
    Real32=5,
    Real64=6
};

using AttributeData=std::variant<std::string,
                                 std::vector<std::int32_t>,
                                 std::vector<float>,
                                 std::vector<double>>;

struct Attribute {
    std::string Name;
    AttributeData Data;
};

struct Dimension {
    std::string Name;
    std::uint32_t Length;
};

struct Variable {
    std::string Name;
    DataKind Kind;
    std::vector<std::uint32_t> Dimensions;
    std::vector<Attribute> Attributes;
    bool IsRecord=false;
    std::uint32_t StoredSize=0;
    std::uint64_t Offset=0;
    BinaryBlock InitialData;
};

[[nodiscard]] std::uint32_t kindSize(const DataKind kind) {
    switch (kind) {
    case DataKind::Character:
        return 1;
    case DataKind::Integer:
    case DataKind::Real32:
        return 4;
    case DataKind::Real64:
        return 8;
    }
    return 0;
}

[[nodiscard]] DataKind attributeKind(const AttributeData &data) {
    if (std::holds_alternative<std::string>(data)) {
        return DataKind::Character;
    }
    if (std::holds_alternative<std::vector<std::int32_t>>(data)) {
        return DataKind::Integer;
    }
    if (std::holds_alternative<std::vector<float>>(data)) {
        return DataKind::Real32;
    }
    return DataKind::Real64;
}

[[nodiscard]] std::uint32_t attributeCount(const AttributeData &data) {
    return std::visit([](const auto &values) {
        return static_cast<std::uint32_t>(values.size());
    },data);
}

void encodeAttributeData(BinaryBlock &out,const AttributeData &data) {
    std::visit([&out](const auto &values) {
        using Values=std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<Values,std::string>) {
            out.text(values);
        }
        else {
            for (const auto value:values) {
                if constexpr (std::is_same_v<Values,std::vector<std::int32_t>>) {
                    out.integer(value);
                }
                else if constexpr (std::is_same_v<Values,std::vector<float>>) {
                    out.real32(value);
                }
                else {
                    out.real64(value);
                }
            }
        }
    },data);
    out.pad();
}

void encodeAttributes(BinaryBlock &out,
                      const std::vector<Attribute> &attributes) {
    if (attributes.empty()) {
        out.unsigned32(0);
        out.unsigned32(0);
        return;
    }
    out.unsigned32(attributeTag);
    out.unsigned32(static_cast<std::uint32_t>(attributes.size()));
    for (const Attribute &attribute:attributes) {
        out.name(attribute.Name);
        out.unsigned32(static_cast<std::uint32_t>(
            attributeKind(attribute.Data)));
        out.unsigned32(attributeCount(attribute.Data));
        encodeAttributeData(out,attribute.Data);
    }
}

[[nodiscard]] std::uint64_t variableRawSize(
    const Variable &variable,
    const std::vector<Dimension> &dimensions) {
    std::uint64_t entries=1;
    for (const std::uint32_t id:variable.Dimensions) {
        const std::uint32_t length=dimensions[id].Length;
        if (length!=0) entries*=length;
    }
    return entries*kindSize(variable.Kind);
}

[[nodiscard]] BinaryBlock makeHeader(
    const std::vector<Dimension> &dimensions,
    const std::vector<Attribute> &globalAttributes,
    const std::vector<Variable> &variables,
    const std::uint32_t records) {
    BinaryBlock header;
    header.text("CDF");
    header.byte(2);
    header.unsigned32(records);

    header.unsigned32(dimensionTag);
    header.unsigned32(static_cast<std::uint32_t>(dimensions.size()));
    for (const Dimension &dimension:dimensions) {
        header.name(dimension.Name);
        header.unsigned32(dimension.Length);
    }

    encodeAttributes(header,globalAttributes);
    header.unsigned32(variableTag);
    header.unsigned32(static_cast<std::uint32_t>(variables.size()));
    for (const Variable &variable:variables) {
        header.name(variable.Name);
        header.unsigned32(
            static_cast<std::uint32_t>(variable.Dimensions.size()));
        for (const std::uint32_t id:variable.Dimensions) {
            header.unsigned32(id);
        }
        encodeAttributes(header,variable.Attributes);
        header.unsigned32(static_cast<std::uint32_t>(variable.Kind));
        header.unsigned32(variable.StoredSize);
        header.unsigned64(variable.Offset);
    }
    return header;
}

[[nodiscard]] std::string elementLabel(const MeshType type) {
    switch (type) {
    case MeshType::EDGE2:
        return "BAR2";
    case MeshType::TRI3:
        return "TRI3";
    case MeshType::QUAD4:
        return "QUAD4";
    case MeshType::TET4:
        return "TETRA4";
    case MeshType::HEX8:
        return "HEX8";
    default:
        return {};
    }
}

[[nodiscard]] int elementDimension(const MeshType type) {
    switch (type) {
    case MeshType::EDGE2:
        return 1;
    case MeshType::TRI3:
    case MeshType::QUAD4:
        return 2;
    case MeshType::TET4:
    case MeshType::HEX8:
        return 3;
    default:
        return 0;
    }
}

[[nodiscard]] int elementNodes(const MeshType type) {
    switch (type) {
    case MeshType::EDGE2:
        return 2;
    case MeshType::TRI3:
        return 3;
    case MeshType::QUAD4:
    case MeshType::TET4:
        return 4;
    case MeshType::HEX8:
        return 8;
    default:
        return 0;
    }
}

[[nodiscard]] bool validMesh(const MeshData &mesh,std::string &reason) {
    if (mesh.MeshDim<1 || mesh.MeshDim>3) {
        reason="mesh dimension must be one, two, or three";
        return false;
    }
    if (mesh.NodesNum<1
        || mesh.NodeCoords.size()!=static_cast<std::size_t>(mesh.NodesNum)*3U) {
        reason="node coordinates do not match NodesNum";
        return false;
    }
    if (mesh.BulkElmtsNum<1
        || mesh.BulkElmtConn.size()!=static_cast<std::size_t>(
            mesh.BulkElmtsNum)) {
        reason="bulk connectivity does not match BulkElmtsNum";
        return false;
    }
    if (elementLabel(mesh.BulkElmtMeshType).empty()
        || elementDimension(mesh.BulkElmtMeshType)!=mesh.MeshDim
        || elementNodes(mesh.BulkElmtMeshType)!=mesh.NodesNumPerBulkElmt) {
        reason="bulk element type is not supported by the public writer";
        return false;
    }
    for (const std::vector<int> &element:mesh.BulkElmtConn) {
        if (element.size()!=static_cast<std::size_t>(
                mesh.NodesNumPerBulkElmt)) {
            reason="bulk element has an inconsistent node count";
            return false;
        }
        if (std::any_of(element.begin(),element.end(),
                        [&mesh](const int node) {
                            return node<1 || node>mesh.NodesNum;
                        })) {
            reason="bulk connectivity contains an invalid node id";
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool validNames(const std::vector<std::string> &names,
                              std::string &reason) {
    if (names.empty()) {
        reason="at least one element variable is required";
        return false;
    }
    std::set<std::string> unique;
    for (const std::string &name:names) {
        if (name.empty() || name.size()>=nameLength) {
            reason="element-variable names must contain 1 to 32 characters";
            return false;
        }
        if (!unique.insert(name).second) {
            reason="element-variable names must be unique";
            return false;
        }
    }
    return true;
}

void report(const std::string &message) {
    std::cerr<<"ExodusWriter: "<<message<<'\n';
}
}

class ExodusWriter::State {
public:
    std::string FileName;
    std::uint64_t RecordStart=0;
    std::uint64_t RecordSize=0;
    std::uint32_t RecordCount=0;
    std::size_t ElementCount=0;
    std::size_t VariableCount=0;
    double LastTime=0.0;
    bool Ready=false;
};

ExodusWriter::ExodusWriter():m_State(std::make_unique<State>()) {}
ExodusWriter::~ExodusWriter()=default;
ExodusWriter::ExodusWriter(ExodusWriter &&) noexcept=default;
ExodusWriter& ExodusWriter::operator=(ExodusWriter &&) noexcept=default;

bool ExodusWriter::isOpen() const noexcept {
    return m_State!=nullptr && m_State->Ready;
}

bool ExodusWriter::begin(const std::string &filename,
                         const MeshData &mesh,
                         const std::vector<std::string> &variableNames,
                         const std::string &title) {
    m_State=std::make_unique<State>();
    std::string reason;
    if (filename.empty()) {
        report("output filename is empty");
        return false;
    }
    if (!validMesh(mesh,reason) || !validNames(variableNames,reason)) {
        report(reason);
        return false;
    }

    std::vector<Dimension> dimensions;
    const auto addDimension=[&dimensions](std::string name,
                                          const std::uint32_t length) {
        dimensions.push_back({std::move(name),length});
        return static_cast<std::uint32_t>(dimensions.size()-1);
    };
    const std::uint32_t lenString=addDimension(
        "len_string",qaStringLength);
    const std::uint32_t lenName=addDimension("len_name",nameLength);
    const std::uint32_t four=addDimension("four",4);
    const std::uint32_t timeStep=addDimension("time_step",0);
    const std::uint32_t numDim=addDimension(
        "num_dim",static_cast<std::uint32_t>(mesh.MeshDim));
    const std::uint32_t numNodes=addDimension(
        "num_nodes",static_cast<std::uint32_t>(mesh.NodesNum));
    const std::uint32_t numElements=addDimension(
        "num_elem",static_cast<std::uint32_t>(mesh.BulkElmtsNum));
    const std::uint32_t numBlocks=addDimension("num_el_blk",1);
    const std::uint32_t elementsInBlock=addDimension(
        "num_el_in_blk1",static_cast<std::uint32_t>(mesh.BulkElmtsNum));
    const std::uint32_t nodesInElement=addDimension(
        "num_nod_per_el1",
        static_cast<std::uint32_t>(mesh.NodesNumPerBulkElmt));
    const std::uint32_t numElementVariables=addDimension(
        "num_elem_var",static_cast<std::uint32_t>(variableNames.size()));
    const std::uint32_t numQaRecords=addDimension("num_qa_rec",1);

    const std::vector<Attribute> globalAttributes={
        {"floating_point_word_size",std::vector<std::int32_t>{8}},
        {"file_size",std::vector<std::int32_t>{1}},
        {"int64_status",std::vector<std::int32_t>{0}},
        {"maximum_name_length",std::vector<std::int32_t>{32}},
        {"version",std::vector<float>{7.02F}},
        {"api_version",std::vector<float>{7.02F}},
        {"title",title}
    };

    std::vector<Variable> variables;
    const auto addVariable=[&variables](
        std::string name,
        const DataKind kind,
        std::vector<std::uint32_t> dims,
        std::vector<Attribute> attributes={},
        const bool record=false) -> Variable& {
        variables.push_back({std::move(name),kind,std::move(dims),
                             std::move(attributes),record,0,0,{}});
        return variables.back();
    };

    for (int axis=0;axis<mesh.MeshDim;++axis) {
        Variable &coordinate=addVariable(
            std::string("coord")+static_cast<char>('x'+axis),
            DataKind::Real64,{numNodes});
        for (int node=0;node<mesh.NodesNum;++node) {
            coordinate.InitialData.real64(
                mesh.NodeCoords[static_cast<std::size_t>(node)*3U
                                +static_cast<std::size_t>(axis)]);
        }
    }

    Variable &coordinateNames=addVariable(
        "coor_names",DataKind::Character,{numDim,lenName});
    for (int axis=0;axis<mesh.MeshDim;++axis) {
        coordinateNames.InitialData.fixedText(
            std::string(1,static_cast<char>('x'+axis)),nameLength);
    }

    Variable &blockName=addVariable(
        "eb_names",DataKind::Character,{numBlocks,lenName});
    blockName.InitialData.fixedText("block_1",nameLength);

    Variable &blockStatus=addVariable(
        "eb_status",DataKind::Integer,{numBlocks});
    blockStatus.InitialData.integer(1);

    Variable &blockId=addVariable(
        "eb_prop1",DataKind::Integer,{numBlocks},
        {{"name",std::string("ID")}});
    blockId.InitialData.integer(1);

    Variable &connectivity=addVariable(
        "connect1",DataKind::Integer,{elementsInBlock,nodesInElement},
        {{"elem_type",elementLabel(mesh.BulkElmtMeshType)}});
    for (const std::vector<int> &element:mesh.BulkElmtConn) {
        for (const int node:element) {
            connectivity.InitialData.integer(node);
        }
    }

    Variable &elementNames=addVariable(
        "name_elem_var",DataKind::Character,
        {numElementVariables,lenName});
    for (const std::string &name:variableNames) {
        elementNames.InitialData.fixedText(name,nameLength);
    }

    Variable &truthTable=addVariable(
        "elem_var_tab",DataKind::Integer,
        {numBlocks,numElementVariables});
    for (std::size_t i=0;i<variableNames.size();++i) {
        truthTable.InitialData.integer(1);
    }

    Variable &quality=addVariable(
        "qa_records",DataKind::Character,
        {numQaRecords,four,lenString});
    quality.InitialData.fixedText("PeriX",qaStringLength);
    quality.InitialData.fixedText("public",qaStringLength);
    quality.InitialData.fixedText("",qaStringLength);
    quality.InitialData.fixedText("",qaStringLength);

    addVariable("time_whole",DataKind::Real64,{timeStep},{},true);
    for (std::size_t index=0;index<variableNames.size();++index) {
        addVariable("vals_elem_var"+std::to_string(index+1)+"eb1",
                    DataKind::Real64,{timeStep,numElements},{},true);
    }

    for (Variable &variable:variables) {
        const std::uint64_t bytes=aligned(
            variableRawSize(variable,dimensions));
        if (bytes>std::numeric_limits<std::uint32_t>::max()) {
            report("a NetCDF variable exceeds the CDF-2 size limit");
            return false;
        }
        variable.StoredSize=static_cast<std::uint32_t>(bytes);
    }

    for (Variable &variable:variables) {
        variable.Offset=0;
    }
    BinaryBlock provisional=makeHeader(
        dimensions,globalAttributes,variables,0);
    std::uint64_t next=provisional.bytes().size();
    for (Variable &variable:variables) {
        if (!variable.IsRecord) {
            variable.Offset=next;
            next+=variable.StoredSize;
        }
    }
    const std::uint64_t recordStart=next;
    std::uint64_t recordSize=0;
    for (Variable &variable:variables) {
        if (variable.IsRecord) {
            variable.Offset=recordStart+recordSize;
            recordSize+=variable.StoredSize;
        }
    }
    const BinaryBlock header=makeHeader(
        dimensions,globalAttributes,variables,0);
    if (header.bytes().size()!=provisional.bytes().size()) {
        report("internal header layout changed unexpectedly");
        return false;
    }

    std::ofstream output(filename,std::ios::binary|std::ios::trunc);
    if (!output) {
        report("cannot create '"+filename+"'");
        return false;
    }
    output.write(header.bytes().data(),
                 static_cast<std::streamsize>(header.bytes().size()));
    for (const Variable &variable:variables) {
        if (variable.IsRecord) continue;
        const std::uint64_t expected=variableRawSize(variable,dimensions);
        if (variable.InitialData.bytes().size()!=expected) {
            report("internal data-size mismatch for '"+variable.Name+"'");
            return false;
        }
        output.seekp(static_cast<std::streamoff>(variable.Offset));
        output.write(variable.InitialData.bytes().data(),
                     static_cast<std::streamsize>(
                         variable.InitialData.bytes().size()));
        const std::uint64_t padding=
            variable.StoredSize-variable.InitialData.bytes().size();
        for (std::uint64_t i=0;i<padding;++i) output.put('\0');
    }
    output.flush();
    if (!output.good()) {
        report("I/O failure while creating '"+filename+"'");
        return false;
    }

    m_State->FileName=filename;
    m_State->RecordStart=recordStart;
    m_State->RecordSize=recordSize;
    m_State->ElementCount=static_cast<std::size_t>(mesh.BulkElmtsNum);
    m_State->VariableCount=variableNames.size();
    m_State->Ready=true;
    return true;
}

bool ExodusWriter::appendStep(
    const double time,
    const std::vector<std::vector<double>> &elementValues) {
    if (!isOpen()) return false;
    if (!std::isfinite(time)
        || (m_State->RecordCount>0 && time<=m_State->LastTime)) {
        report("time values must be finite and strictly increasing");
        return false;
    }
    if (elementValues.size()!=m_State->VariableCount
        || std::any_of(elementValues.begin(),elementValues.end(),
                       [this](const std::vector<double> &values) {
                           return values.size()!=m_State->ElementCount;
                       })) {
        report("element-value dimensions do not match begin()");
        return false;
    }

    BinaryBlock record;
    record.real64(time);
    for (const std::vector<double> &field:elementValues) {
        for (const double value:field) record.real64(value);
        record.pad();
    }
    if (record.bytes().size()!=m_State->RecordSize) {
        report("internal record-size mismatch");
        return false;
    }

    std::fstream output(m_State->FileName,
                        std::ios::binary|std::ios::in|std::ios::out);
    if (!output) {
        report("cannot reopen '"+m_State->FileName+"'");
        return false;
    }
    const std::uint64_t offset=m_State->RecordStart
        +static_cast<std::uint64_t>(m_State->RecordCount)
         *m_State->RecordSize;
    output.seekp(static_cast<std::streamoff>(offset));
    output.write(record.bytes().data(),
                 static_cast<std::streamsize>(record.bytes().size()));
    output.flush();
    if (!output.good()) {
        report("failed while writing a time record");
        return false;
    }

    const std::uint32_t nextRecord=m_State->RecordCount+1;
    BinaryBlock count;
    count.unsigned32(nextRecord);
    output.seekp(4);
    output.write(count.bytes().data(),4);
    output.flush();
    if (!output.good()) {
        report("failed while updating the time-record count");
        return false;
    }
    m_State->RecordCount=nextRecord;
    m_State->LastTime=time;
    return true;
}
