//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "Mesh/MeshData.h"
#include "OutputSystem/ExodusWriter.h"

namespace {
int failures=0;

void expect(const std::string &name,const bool condition) {
    if (condition) return;
    std::printf("FAIL %s\n",name.c_str());
    ++failures;
}

std::array<unsigned char,8> prefix(const std::string &path) {
    std::array<unsigned char,8> bytes{};
    std::ifstream input(path,std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
    return bytes;
}

MeshData oneQuad() {
    MeshData mesh;
    mesh.MeshDim=2;
    mesh.NodesNum=4;
    mesh.NodeCoords={
        0.0,0.0,0.0,
        1.0,0.0,0.0,
        1.0,1.0,0.0,
        0.0,1.0,0.0
    };
    mesh.BulkElmtsNum=1;
    mesh.NodesNumPerBulkElmt=4;
    mesh.BulkElmtMeshType=MeshType::QUAD4;
    mesh.BulkElmtConn={{1,2,3,4}};
    return mesh;
}

MeshData oneElement(const MeshType type,const int dimension,
                    const int nodes) {
    MeshData mesh;
    mesh.MeshDim=dimension;
    mesh.NodesNum=nodes;
    mesh.NodeCoords.resize(static_cast<std::size_t>(nodes)*3U,0.0);
    mesh.BulkElmtsNum=1;
    mesh.NodesNumPerBulkElmt=nodes;
    mesh.BulkElmtMeshType=type;
    mesh.BulkElmtConn.resize(1);
    for (int node=1;node<=nodes;++node) {
        mesh.BulkElmtConn.front().push_back(node);
    }
    return mesh;
}
}

int main() {
    const std::string path="/tmp/perix_exodus_writer_test.e";
    std::remove(path.c_str());

    ExodusWriter writer;
    expect("writer starts closed",!writer.isOpen());
    expect("valid mesh begins",
           writer.begin(path,oneQuad(),{"u","damage"},"writer test"));
    expect("writer opens",writer.isOpen());

    const auto initial=prefix(path);
    expect("CDF-2 signature",
           initial[0]=='C' && initial[1]=='D' && initial[2]=='F'
           && initial[3]==2);
    expect("initial record count",
           initial[4]==0 && initial[5]==0
           && initial[6]==0 && initial[7]==0);

    expect("wrong field count rejected",
           !writer.appendStep(0.0,{{1.0}}));
    expect("first step appended",
           writer.appendStep(0.0,{{1.25},{0.0}}));
    expect("second step appended",
           writer.appendStep(0.5,{{2.5},{0.75}}));
    expect("descending time rejected",
           !writer.appendStep(0.25,{{3.0},{1.0}}));

    const auto completed=prefix(path);
    expect("record count is two",
           completed[4]==0 && completed[5]==0
           && completed[6]==0 && completed[7]==2);

    ExodusWriter invalid;
    MeshData bad=oneQuad();
    bad.BulkElmtConn={{1,2,3,8}};
    expect("invalid connectivity rejected",
           !invalid.begin("/tmp/perix_exodus_invalid.e",bad,{"u"},"bad"));
    expect("invalid writer remains closed",!invalid.isOpen());

    bad=oneQuad();
    bad.NodesNumPerBulkElmt=3;
    expect("element topology mismatch rejected",
           !invalid.begin("/tmp/perix_exodus_invalid.e",bad,{"u"},"bad"));

    struct Topology {
        const char *Name;
        MeshType Type;
        int Dimension;
        int Nodes;
    };
    const std::array<Topology,5> topologies={{
        {"edge2",MeshType::EDGE2,1,2},
        {"tri3",MeshType::TRI3,2,3},
        {"quad4",MeshType::QUAD4,2,4},
        {"tet4",MeshType::TET4,3,4},
        {"hex8",MeshType::HEX8,3,8}
    }};
    for (const Topology &topology:topologies) {
        ExodusWriter candidate;
        const std::string label=std::string("public topology ")+topology.Name;
        expect(label,candidate.begin(
            std::string("/tmp/perix_exodus_")+topology.Name+".e",
            oneElement(topology.Type,topology.Dimension,topology.Nodes),
            {"u"},"topology test"));
        expect(label+" record",candidate.appendStep(0.0,{{1.0}}));
    }

    if (failures==0) {
        std::printf("ExodusWriter tests passed\n");
        return 0;
    }
    std::printf("ExodusWriter tests failed: %d\n",failures);
    return 1;
}
