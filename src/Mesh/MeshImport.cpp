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
//+++ Date    : 2026.06.07
//+++ Function: import ASCII Gmsh MSH 4.1 files into MeshData.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Mesh/MeshImport.h"

#include "Utils/MessagePrinter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

namespace {
struct ElementTypeInfo {
    int Dim=0;
    int Nodes=0;
    int VTKType=0;
    MeshType Type=MeshType::NULLTYPE;
    const char *Name="";
};

struct PhysicalGroup {
    int Dim=0;
    int Tag=0;
    std::string Name;
};

struct EntityInfo {
    int Dim=0;
    int Tag=0;
    std::vector<int> PhysicalTags;
};

struct RawElement {
    int Dim=0;
    int EntityTag=0;
    int ElementType=0;
    std::vector<int> Conn;
};

using EntityKey=std::pair<int,int>;

[[nodiscard]] bool fail(const std::string &msg) {
    MessagePrinter::printErrorTxt(msg);
    return false;
}

[[nodiscard]] bool expectToken(std::istream &in,
                               const std::string &expected,
                               const std::string &where) {
    std::string token;
    if (!(in >> token)) {
        return fail("MeshImport: unexpected end of file while reading "+where);
    }
    if (token!=expected) {
        return fail("MeshImport: expected '"+expected+"' after "+where+", got '"+token+"'");
    }
    return true;
}

[[nodiscard]] ElementTypeInfo elementTypeInfo(const int type) {
    switch (type) {
    case 1:  return {1,2,3, MeshType::EDGE2,"edge2"};
    case 2:  return {2,3,5, MeshType::TRI3, "tri3"};
    case 3:  return {2,4,9, MeshType::QUAD4,"quad4"};
    case 4:  return {3,4,10,MeshType::TET4, "tet4"};
    case 5:  return {3,8,12,MeshType::HEX8, "hex8"};
    case 15: return {0,1,1, MeshType::POINT,"point"};
    default: return {};
    }
}

[[nodiscard]] std::array<double,3> coord(const MeshData &data,const int nodeID) {
    const int i=(nodeID-1)*3;
    return {data.NodeCoords[static_cast<std::size_t>(i+0)],
            data.NodeCoords[static_cast<std::size_t>(i+1)],
            data.NodeCoords[static_cast<std::size_t>(i+2)]};
}

[[nodiscard]] std::array<double,3> sub(const std::array<double,3> &a,
                                       const std::array<double,3> &b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}

[[nodiscard]] std::array<double,3> cross(const std::array<double,3> &a,
                                         const std::array<double,3> &b) {
    return {a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}

[[nodiscard]] double dot(const std::array<double,3> &a,
                         const std::array<double,3> &b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

[[nodiscard]] double norm(const std::array<double,3> &a) {
    return std::sqrt(dot(a,a));
}

[[nodiscard]] double dist(const std::array<double,3> &a,
                          const std::array<double,3> &b) {
    return norm(sub(a,b));
}

[[nodiscard]] std::array<double,3> centroid(const MeshData &data,
                                            const std::vector<int> &conn) {
    std::array<double,3> c{0.0,0.0,0.0};
    for (const int n:conn) {
        const auto x=coord(data,n);
        c[0]+=x[0]; c[1]+=x[1]; c[2]+=x[2];
    }
    const double inv=1.0/static_cast<double>(conn.size());
    c[0]*=inv; c[1]*=inv; c[2]*=inv;
    return c;
}

[[nodiscard]] double triangleArea(const std::array<double,3> &a,
                                  const std::array<double,3> &b,
                                  const std::array<double,3> &c) {
    return 0.5*norm(cross(sub(b,a),sub(c,a)));
}

[[nodiscard]] double tetraVolume(const std::array<double,3> &a,
                                 const std::array<double,3> &b,
                                 const std::array<double,3> &c,
                                 const std::array<double,3> &d) {
    return std::fabs(dot(sub(b,a),cross(sub(c,a),sub(d,a))))/6.0;
}

[[nodiscard]] double elementMeasure(const MeshData &data,
                                    const std::vector<int> &conn,
                                    const MeshType type) {
    if (type==MeshType::EDGE2) {
        return dist(coord(data,conn[0]),coord(data,conn[1]));
    }
    if (type==MeshType::TRI3) {
        return triangleArea(coord(data,conn[0]),coord(data,conn[1]),coord(data,conn[2]));
    }
    if (type==MeshType::QUAD4) {
        const auto a=coord(data,conn[0]);
        const auto b=coord(data,conn[1]);
        const auto c=coord(data,conn[2]);
        const auto d=coord(data,conn[3]);
        return triangleArea(a,b,c)+triangleArea(a,c,d);
    }
    if (type==MeshType::TET4) {
        return tetraVolume(coord(data,conn[0]),coord(data,conn[1]),
                           coord(data,conn[2]),coord(data,conn[3]));
    }
    if (type==MeshType::HEX8) {
        const auto p0=coord(data,conn[0]);
        const auto p1=coord(data,conn[1]);
        const auto p2=coord(data,conn[2]);
        const auto p3=coord(data,conn[3]);
        const auto p4=coord(data,conn[4]);
        const auto p5=coord(data,conn[5]);
        const auto p6=coord(data,conn[6]);
        const auto p7=coord(data,conn[7]);
        return tetraVolume(p0,p1,p3,p4)
             + tetraVolume(p1,p2,p3,p6)
             + tetraVolume(p1,p3,p4,p6)
             + tetraVolume(p1,p4,p5,p6)
             + tetraVolume(p3,p4,p6,p7);
    }
    return 0.0;
}

[[nodiscard]] std::vector<std::pair<int,int>> elementEdges(const MeshType type) {
    if (type==MeshType::EDGE2) return {{0,1}};
    if (type==MeshType::TRI3)  return {{0,1},{1,2},{2,0}};
    if (type==MeshType::QUAD4) return {{0,1},{1,2},{2,3},{3,0}};
    if (type==MeshType::TET4)  return {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    if (type==MeshType::HEX8)  return {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},
                                       {6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    return {};
}

[[nodiscard]] std::vector<std::vector<int>> boundaryFaces(const std::vector<int> &conn,
                                                          const MeshType type) {
    if (type==MeshType::TRI3) {
        return {{conn[0],conn[1]},{conn[1],conn[2]},{conn[2],conn[0]}};
    }
    if (type==MeshType::QUAD4) {
        return {{conn[0],conn[1]},{conn[1],conn[2]},{conn[2],conn[3]},{conn[3],conn[0]}};
    }
    if (type==MeshType::TET4) {
        return {{conn[0],conn[2],conn[1]},{conn[0],conn[1],conn[3]},
                {conn[1],conn[2],conn[3]},{conn[2],conn[0],conn[3]}};
    }
    if (type==MeshType::HEX8) {
        return {{conn[0],conn[3],conn[2],conn[1]},{conn[4],conn[5],conn[6],conn[7]},
                {conn[0],conn[1],conn[5],conn[4]},{conn[1],conn[2],conn[6],conn[5]},
                {conn[2],conn[3],conn[7],conn[6]},{conn[3],conn[0],conn[4],conn[7]}};
    }
    return {};
}

[[nodiscard]] std::vector<int> sortedConn(std::vector<int> conn) {
    std::sort(conn.begin(),conn.end());
    return conn;
}

[[nodiscard]] std::array<double,3> boundaryNormal(const MeshData &data,
                                                  const std::vector<int> &boundaryConn,
                                                  const std::array<double,3> &bulkCenter,
                                                  const int meshDim) {
    std::array<double,3> n{0.0,0.0,0.0};
    if (meshDim==2) {
        const auto a=coord(data,boundaryConn[0]);
        const auto b=coord(data,boundaryConn[1]);
        const double tx=b[0]-a[0];
        const double ty=b[1]-a[1];
        n={ty,-tx,0.0};
    }
    else {
        const auto a=coord(data,boundaryConn[0]);
        const auto b=coord(data,boundaryConn[1]);
        const auto c=coord(data,boundaryConn[2]);
        n=cross(sub(b,a),sub(c,a));
    }
    const double len=norm(n);
    if (len<=0.0) return {0.0,0.0,0.0};
    n[0]/=len; n[1]/=len; n[2]/=len;

    const auto faceCenter=centroid(data,boundaryConn);
    if (dot(sub(bulkCenter,faceCenter),n)>0.0) {
        n[0]*=-1.0; n[1]*=-1.0; n[2]*=-1.0;
    }
    return n;
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t mid=values.size()/2;
    std::nth_element(values.begin(),values.begin()+static_cast<long>(mid),values.end());
    double m=values[mid];
    if (values.size()%2==0) {
        std::nth_element(values.begin(),values.begin()+static_cast<long>(mid-1),values.end());
        m=0.5*(m+values[mid-1]);
    }
    return m;
}

void addPhysicalGroup(MeshData &data,
                      const int dim,
                      const int id,
                      const std::string &name) {
    data.PhyGroupDimVec.push_back(dim);
    data.PhyGroupIDVec.push_back(id);
    data.PhyGroupNameVec.push_back(name);
    data.PhyGroupElmtsNumVec.push_back(0);
    data.PhyGroupID2NameMap[id]=name;
    data.PhyGroupName2IDMap[name]=id;
}

void refreshPhysicalGroupCounts(MeshData &data) {
    data.PhyGroupNum=static_cast<int>(data.PhyGroupNameVec.size());
    for (int i=0;i<data.PhyGroupNum;i++) {
        const std::string &name=data.PhyGroupNameVec[static_cast<std::size_t>(i)];
        const auto it=data.PhyGroupName2ElmtConnMap.find(name);
        data.PhyGroupElmtsNumVec[static_cast<std::size_t>(i)]=
            (it==data.PhyGroupName2ElmtConnMap.end()) ? 0 : static_cast<int>(it->second.size());
    }
}

void setRoleElementInfo(MeshData &data,
                        const int meshDim,
                        const std::map<int,int> &roleTypeByDim) {
    auto setInfo=[&](const int dim,int &nodes,int &vtk,MeshType &type,std::string *name) {
        const auto it=roleTypeByDim.find(dim);
        if (it==roleTypeByDim.end()) {
            nodes=0; vtk=0; type=MeshType::NULLTYPE;
            if (name) name->clear();
            return;
        }
        const auto info=elementTypeInfo(it->second);
        nodes=info.Nodes;
        vtk=info.VTKType;
        type=info.Type;
        if (name) *name=info.Name;
    };
    setInfo(meshDim,data.NodesNumPerBulkElmt,data.BulkElmtVTKCellType,
            data.BulkElmtMeshType,&data.BulkMeshTypeName);
    setInfo(1,data.NodesNumPerLineElmt,data.LineElmtVTKCellType,
            data.LineElmtMeshType,nullptr);
    if (meshDim==3) {
        setInfo(2,data.NodesNumPerSurfElmt,data.SurfElmtVTKCellType,
                data.SurfElmtMeshType,nullptr);
    }
}
}

bool MeshImport::importMSH(const std::string &fileName,MeshData &data) const {
    std::ifstream in(fileName);
    if (!in.is_open()) {
        return fail("MeshImport: cannot open msh file '"+fileName+"'");
    }

    std::vector<PhysicalGroup> physicalGroups;
    std::map<std::pair<int,int>,std::string> physicalNameByDimTag;
    std::map<EntityKey,EntityInfo> entities;
    std::unordered_map<unsigned long long,int> nodeTagToID;
    std::vector<double> nodeCoords(3,0.0); // index 0 unused
    std::vector<RawElement> rawElements;
    std::map<int,int> roleTypeByDim;

    bool sawMeshFormat=false;
    bool sawNodes=false;
    bool sawElements=false;

    std::string token;
    while (in >> token) {
        if (token=="$MeshFormat") {
            double version=0.0;
            int fileType=-1;
            int dataSize=0;
            if (!(in >> version >> fileType >> dataSize)) {
                return fail("MeshImport: failed to read $MeshFormat");
            }
            if (version<4.1 || version>=4.2) {
                return fail("MeshImport: only ASCII MSH 4.1 files are supported");
            }
            if (fileType!=0) {
                return fail("MeshImport: binary MSH files are not supported");
            }
            if (dataSize!=8) {
                return fail("MeshImport: unsupported floating-point byte size in MSH file");
            }
            if (!expectToken(in,"$EndMeshFormat","$MeshFormat")) return false;
            sawMeshFormat=true;
        }
        else if (token=="$PhysicalNames") {
            int n=0;
            if (!(in >> n)) return fail("MeshImport: failed to read $PhysicalNames count");
            physicalGroups.reserve(static_cast<std::size_t>(n));
            for (int i=0;i<n;i++) {
                PhysicalGroup g;
                if (!(in >> g.Dim >> g.Tag >> std::quoted(g.Name))) {
                    return fail("MeshImport: failed to read a physical name entry");
                }
                physicalGroups.push_back(g);
                physicalNameByDimTag[{g.Dim,g.Tag}]=g.Name;
            }
            if (!expectToken(in,"$EndPhysicalNames","$PhysicalNames")) return false;
        }
        else if (token=="$Entities") {
            int numPoints=0,numCurves=0,numSurfaces=0,numVolumes=0;
            if (!(in >> numPoints >> numCurves >> numSurfaces >> numVolumes)) {
                return fail("MeshImport: failed to read $Entities header");
            }
            auto readEntity=[&](const int dim)->bool {
                EntityInfo e;
                e.Dim=dim;
                if (!(in >> e.Tag)) return false;
                double dummy=0.0;
                const int bboxValues=(dim==0) ? 3 : 6;
                for (int k=0;k<bboxValues;k++) {
                    if (!(in >> dummy)) return false;
                }
                int nphys=0;
                if (!(in >> nphys)) return false;
                e.PhysicalTags.resize(static_cast<std::size_t>(nphys));
                for (int k=0;k<nphys;k++) {
                    if (!(in >> e.PhysicalTags[static_cast<std::size_t>(k)])) return false;
                }
                if (dim>0) {
                    int nbound=0;
                    if (!(in >> nbound)) return false;
                    int signedTag=0;
                    for (int k=0;k<nbound;k++) {
                        if (!(in >> signedTag)) return false;
                    }
                }
                entities[{dim,e.Tag}]=std::move(e);
                return true;
            };
            for (int i=0;i<numPoints;i++)   if (!readEntity(0)) return fail("MeshImport: failed to read point entity");
            for (int i=0;i<numCurves;i++)   if (!readEntity(1)) return fail("MeshImport: failed to read curve entity");
            for (int i=0;i<numSurfaces;i++) if (!readEntity(2)) return fail("MeshImport: failed to read surface entity");
            for (int i=0;i<numVolumes;i++)  if (!readEntity(3)) return fail("MeshImport: failed to read volume entity");
            if (!expectToken(in,"$EndEntities","$Entities")) return false;
        }
        else if (token=="$Nodes") {
            unsigned long long numEntityBlocks=0,numNodes=0,minTag=0,maxTag=0;
            if (!(in >> numEntityBlocks >> numNodes >> minTag >> maxTag)) {
                return fail("MeshImport: failed to read $Nodes header");
            }
            (void)minTag;
            (void)maxTag;
            nodeTagToID.reserve(static_cast<std::size_t>(numNodes));
            nodeCoords.assign(3,0.0);
            int nextID=1;
            for (unsigned long long b=0;b<numEntityBlocks;b++) {
                int entityDim=0,entityTag=0,parametric=0;
                unsigned long long numNodesInBlock=0;
                if (!(in >> entityDim >> entityTag >> parametric >> numNodesInBlock)) {
                    return fail("MeshImport: failed to read node block header");
                }
                (void)entityTag;
                std::vector<unsigned long long> tags(static_cast<std::size_t>(numNodesInBlock));
                for (auto &tag:tags) {
                    if (!(in >> tag)) return fail("MeshImport: failed to read node tag");
                }
                for (const auto tag:tags) {
                    double x=0.0,y=0.0,z=0.0;
                    if (!(in >> x >> y >> z)) {
                        return fail("MeshImport: failed to read node coordinates");
                    }
                    if (parametric!=0) {
                        double param=0.0;
                        for (int k=0;k<entityDim;k++) {
                            if (!(in >> param)) return fail("MeshImport: failed to read node parametric coordinate");
                        }
                    }
                    nodeTagToID[tag]=nextID++;
                    nodeCoords.push_back(x);
                    nodeCoords.push_back(y);
                    nodeCoords.push_back(z);
                }
            }
            if (!expectToken(in,"$EndNodes","$Nodes")) return false;
            sawNodes=true;
        }
        else if (token=="$Elements") {
            unsigned long long numEntityBlocks=0,numElements=0,minTag=0,maxTag=0;
            if (!(in >> numEntityBlocks >> numElements >> minTag >> maxTag)) {
                return fail("MeshImport: failed to read $Elements header");
            }
            (void)minTag;
            (void)maxTag;
            rawElements.reserve(static_cast<std::size_t>(numElements));
            for (unsigned long long b=0;b<numEntityBlocks;b++) {
                int entityDim=0,entityTag=0,elementType=0;
                unsigned long long numElementsInBlock=0;
                if (!(in >> entityDim >> entityTag >> elementType >> numElementsInBlock)) {
                    return fail("MeshImport: failed to read element block header");
                }
                const auto info=elementTypeInfo(elementType);
                if (info.Nodes<=0) {
                    return fail("MeshImport: unsupported Gmsh element type "+std::to_string(elementType));
                }
                if (info.Dim!=entityDim) {
                    return fail("MeshImport: element type dimension does not match its entity dimension");
                }
                auto roleIt=roleTypeByDim.find(entityDim);
                if (roleIt==roleTypeByDim.end()) {
                    roleTypeByDim[entityDim]=elementType;
                }
                else if (roleIt->second!=elementType) {
                    return fail("MeshImport: mixed element types in the same topological dimension are not supported");
                }
                for (unsigned long long i=0;i<numElementsInBlock;i++) {
                    unsigned long long elementTag=0;
                    if (!(in >> elementTag)) return fail("MeshImport: failed to read element tag");
                    (void)elementTag;
                    RawElement e;
                    e.Dim=entityDim;
                    e.EntityTag=entityTag;
                    e.ElementType=elementType;
                    e.Conn.resize(static_cast<std::size_t>(info.Nodes));
                    for (int a=0;a<info.Nodes;a++) {
                        unsigned long long nodeTag=0;
                        if (!(in >> nodeTag)) return fail("MeshImport: failed to read element connectivity");
                        const auto mapIt=nodeTagToID.find(nodeTag);
                        if (mapIt==nodeTagToID.end()) {
                            return fail("MeshImport: element references node tag "+std::to_string(nodeTag)
                                        +" before it was defined");
                        }
                        e.Conn[static_cast<std::size_t>(a)]=mapIt->second;
                    }
                    rawElements.push_back(std::move(e));
                }
            }
            if (!expectToken(in,"$EndElements","$Elements")) return false;
            sawElements=true;
        }
        else if (token.rfind("$End",0)==0) {
            return fail("MeshImport: unexpected section terminator '"+token+"'");
        }
        else if (token.rfind("$",0)==0) {
            const std::string endToken="$End"+token.substr(1);
            std::string skip;
            while (in >> skip) {
                if (skip==endToken) break;
            }
            if (skip!=endToken) {
                return fail("MeshImport: failed to skip unsupported section '"+token+"'");
            }
        }
        else {
            return fail("MeshImport: unexpected token '"+token+"'");
        }
    }

    if (!sawMeshFormat || !sawNodes || !sawElements) {
        return fail("MeshImport: MSH file must contain $MeshFormat, $Nodes, and $Elements sections");
    }
    if (physicalGroups.empty()) {
        return fail("MeshImport: MSH file must contain $PhysicalNames for PeriX boundary sets");
    }
    if (rawElements.empty()) {
        return fail("MeshImport: MSH file contains no supported elements");
    }

    data=MeshData{};
    data.IsImported=true;
    data.ImportedFileName=fileName;
    data.NodeCoords=std::move(nodeCoords);
    data.NodesNum=static_cast<int>(data.NodeCoords.size()/3)-1;
    data.NodeCoords.erase(data.NodeCoords.begin(),data.NodeCoords.begin()+3);

    int meshDim=0;
    for (const auto &e:rawElements) {
        meshDim=std::max(meshDim,e.Dim);
    }
    if (meshDim<1 || meshDim>3) {
        return fail("MeshImport: invalid imported mesh dimension");
    }
    data.MeshDim=meshDim;
    data.MeshOrder=1;
    setRoleElementInfo(data,meshDim,roleTypeByDim);
    if (data.NodesNumPerBulkElmt<=0) {
        return fail("MeshImport: no bulk elements were found in the imported mesh");
    }

    data.ElmtConn.clear();
    for (const auto &e:rawElements) {
        data.ElmtConn.push_back(e.Conn);
        if (e.Dim==0) data.PointElmtsNum+=1;
        if (e.Dim==1) data.LineElmtsNum+=1;
        if (e.Dim==2 && meshDim==3) data.SurfElmtsNum+=1;
    }
    data.ElmtsNum=static_cast<int>(rawElements.size());

    for (const auto &g:physicalGroups) {
        addPhysicalGroup(data,g.Dim,g.Tag,g.Name);
    }
    if (!data.PhyGroupName2IDMap.contains("alldomain")) {
        addPhysicalGroup(data,meshDim,0,"alldomain");
    }

    std::map<std::vector<int>,int> faceToBulkID;
    std::vector<double> edgeLengths;

    auto physicalNamesForEntity=[&](const int dim,const int entityTag)->std::vector<std::string> {
        std::vector<std::string> names;
        const auto entIt=entities.find({dim,entityTag});
        if (entIt==entities.end()) {
            return names;
        }
        for (const int physicalTag:entIt->second.PhysicalTags) {
            const auto nameIt=physicalNameByDimTag.find({dim,physicalTag});
            if (nameIt!=physicalNameByDimTag.end()) {
                names.push_back(nameIt->second);
            }
        }
        return names;
    };

    for (const auto &e:rawElements) {
        const auto info=elementTypeInfo(e.ElementType);
        if (e.Dim!=meshDim) continue;

        data.BulkElmtsNum+=1;
        const int bulkID=data.BulkElmtsNum;
        data.BulkElmtConn.push_back(e.Conn);
        data.PhyGroupName2BulkElmtIDVecMap["alldomain"].push_back(bulkID);
        data.PhyGroupID2BulkElmtIDVecMap[data.PhyGroupName2IDMap["alldomain"]].push_back(bulkID);

        for (const auto &edge:elementEdges(info.Type)) {
            edgeLengths.push_back(dist(coord(data,e.Conn[static_cast<std::size_t>(edge.first)]),
                                       coord(data,e.Conn[static_cast<std::size_t>(edge.second)])));
        }

        for (const auto &face:boundaryFaces(e.Conn,info.Type)) {
            faceToBulkID[sortedConn(face)]=bulkID;
        }

        for (const auto &name:physicalNamesForEntity(e.Dim,e.EntityTag)) {
            data.PhyGroupName2ElmtConnMap[name].push_back(e.Conn);
            data.PhyGroupID2ElmtConnMap[data.PhyGroupName2IDMap[name]].push_back(e.Conn);
            data.PhyGroupName2BulkElmtIDVecMap[name].push_back(bulkID);
            data.PhyGroupID2BulkElmtIDVecMap[data.PhyGroupName2IDMap[name]].push_back(bulkID);
        }
    }
    data.PhyGroupName2ElmtConnMap["alldomain"]=data.BulkElmtConn;
    data.PhyGroupID2ElmtConnMap[data.PhyGroupName2IDMap["alldomain"]]=data.BulkElmtConn;

    data.BulkElmtVolumes.resize(static_cast<std::size_t>(data.BulkElmtsNum),0.0);
    data.BulkElmtCenters.resize(static_cast<std::size_t>(data.BulkElmtsNum)*3,0.0);
    for (int e=1;e<=data.BulkElmtsNum;e++) {
        const auto &conn=data.BulkElmtConn[static_cast<std::size_t>(e-1)];
        const auto c=centroid(data,conn);
        data.BulkElmtCenters[static_cast<std::size_t>(3*(e-1)+0)]=c[0];
        data.BulkElmtCenters[static_cast<std::size_t>(3*(e-1)+1)]=c[1];
        data.BulkElmtCenters[static_cast<std::size_t>(3*(e-1)+2)]=c[2];
        data.BulkElmtVolumes[static_cast<std::size_t>(e-1)]=elementMeasure(data,conn,data.BulkElmtMeshType);
        if (data.BulkElmtVolumes[static_cast<std::size_t>(e-1)]<=0.0) {
            return fail("MeshImport: imported bulk element has non-positive measure");
        }
    }

    for (const auto &e:rawElements) {
        if (e.Dim>=meshDim) continue;
        const auto names=physicalNamesForEntity(e.Dim,e.EntityTag);
        for (const auto &name:names) {
            data.PhyGroupName2ElmtConnMap[name].push_back(e.Conn);
            data.PhyGroupID2ElmtConnMap[data.PhyGroupName2IDMap[name]].push_back(e.Conn);
            if (e.Dim==meshDim-1) {
                const auto adjIt=faceToBulkID.find(sortedConn(e.Conn));
                if (adjIt==faceToBulkID.end()) {
                    return fail("MeshImport: boundary group '"+name+"' contains an element without an adjacent bulk cell");
                }
                const int bulkID=adjIt->second;
                const auto bulkCenter=std::array<double,3>{
                    data.BulkElmtCenters[static_cast<std::size_t>(3*(bulkID-1)+0)],
                    data.BulkElmtCenters[static_cast<std::size_t>(3*(bulkID-1)+1)],
                    data.BulkElmtCenters[static_cast<std::size_t>(3*(bulkID-1)+2)]};
                data.PhyGroupName2BoundaryBulkElmtIDVecMap[name].push_back(bulkID);
                data.PhyGroupName2BoundaryNormalVecMap[name].push_back(boundaryNormal(data,e.Conn,bulkCenter,meshDim));
                data.PhyGroupName2BoundaryMeasureVecMap[name].push_back(
                    elementMeasure(data,e.Conn,elementTypeInfo(e.ElementType).Type));
            }
        }
    }

    double xmin=std::numeric_limits<double>::max();
    double ymin=std::numeric_limits<double>::max();
    double zmin=std::numeric_limits<double>::max();
    double xmax=-std::numeric_limits<double>::max();
    double ymax=-std::numeric_limits<double>::max();
    double zmax=-std::numeric_limits<double>::max();
    for (int n=1;n<=data.NodesNum;n++) {
        const auto x=coord(data,n);
        xmin=std::min(xmin,x[0]); xmax=std::max(xmax,x[0]);
        ymin=std::min(ymin,x[1]); ymax=std::max(ymax,x[1]);
        zmin=std::min(zmin,x[2]); zmax=std::max(zmax,x[2]);
    }
    data.Xmin=xmin; data.Xmax=xmax;
    data.Ymin=ymin; data.Ymax=ymax;
    data.Zmin=zmin; data.Zmax=zmax;

    // CharacteristicLength is the base PD point SPACING used to size the horizon
    // (HorizonRadius = HorizonRadiusFactor * CharacteristicLength). The PD points sit
    // at element CENTROIDS, whose spacing is ~V^(1/dim) -- NOT the element edge. For
    // quad/hex V^(1/dim) equals the cell edge, but for a simplex (tri/tet) the edge is
    // ~1.5-2x larger, so using the edge would make a factor-3 horizon reach ~6x the
    // point spacing -> in 3D ~8x too many neighbours and a near-dense matrix. Use the
    // median centroid spacing so HorizonRadiusFactor is a true multiple of it.
    std::vector<double> pointSpacings;
    pointSpacings.reserve(static_cast<std::size_t>(data.BulkElmtsNum));
    for (int e=0;e<data.BulkElmtsNum;e++) {
        const double v=data.BulkElmtVolumes[static_cast<std::size_t>(e)];
        if (v>0.0) pointSpacings.push_back(std::pow(v,1.0/static_cast<double>(meshDim)));
    }
    data.CharacteristicLength = pointSpacings.empty() ? median(edgeLengths) : median(pointSpacings);
    if (data.CharacteristicLength<=0.0) {
        return fail("MeshImport: failed to compute a positive imported mesh characteristic length");
    }
    refreshPhysicalGroupCounts(data);

    return true;
}
