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
//+++ Date    : 2026.04.14
//+++ Function: generate the pd points
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#include "PDMesh/PDMesh.h"
#include "Utils/MessagePrinter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

void PDMesh::createPDMesh(MeshData &t_MeshData) {
    if (t_MeshData.BulkElmtsNum <= 0 || t_MeshData.BulkElmtConn.empty()) {
        MessagePrinter::printErrorTxt("pd mesh generation requires at least one bulk element");
        MessagePrinter::exitPeriX();
    }

    m_Data.BulkElmtsNum=t_MeshData.BulkElmtsNum;
    m_Data.NodesNum=t_MeshData.BulkElmtsNum;

    // ghost_layer (PDMesh option, default 1): how many mirror ghost layers each
    // boundary wall spawns, PER GROUP. Layer 1 is the original single half-cell
    // ghost (the exact mirror of the adjacent bulk cell); layers 2..N additionally
    // mirror the next bulk cells inward along the wall normal, so a near-boundary
    // bulk node's PD family is completed on the OUTSIDE of the wall instead of
    // staying one-sided. nLayerFor() returns a group's
    // count (its per-group override, else the global GhostLayer). NodesNum is
    // sized below to the UPPER bound bulk + sum_g(nLayerFor(g)*faces_g); the deeper
    // layers de-duplicate / run out of inward bulk, so the arrays are shrunk to the
    // actual count once built. All counts 1 reproduces the original mesh exactly.
    auto nLayerFor=[&](const std::string &name)->int{
        const auto it=m_Data.GhostLayerGroups.find(name);
        const int v=(it!=m_Data.GhostLayerGroups.end())?it->second:m_Data.GhostLayer;
        return (v<1)?1:v;
    };
    int maxLayer=(m_Data.GhostLayer<1)?1:m_Data.GhostLayer;   // for the GhostLayerIndex sizing decision
    for (const auto &kv : m_Data.GhostLayerGroups) maxLayer=std::max(maxLayer,(kv.second<1)?1:kv.second);

    vector<pair<string, const vector<vector<int>>*>> boundaryGroups;
    boundaryGroups.reserve(t_MeshData.PhyGroupNum);
    for (int i=0;i<t_MeshData.PhyGroupNum;i++) {
        const string &phyname=t_MeshData.PhyGroupNameVec[i];
        const int dim=t_MeshData.PhyGroupDimVec[i];
        if (dim==t_MeshData.MeshDim-1 && phyname!="alldomain") {
            const auto connIt=t_MeshData.PhyGroupName2ElmtConnMap.find(phyname);
            if (connIt == t_MeshData.PhyGroupName2ElmtConnMap.end()) {
                MessagePrinter::printErrorTxt("failed to find physical group connectivity when creating pd mesh");
                MessagePrinter::exitPeriX();
            }
            boundaryGroups.emplace_back(phyname,&connIt->second);
            m_Data.NodesNum+=nLayerFor(phyname)*static_cast<int>(connIt->second.size());
        }
    }

    if (t_MeshData.IsImported) {
        const int meshDim=t_MeshData.MeshDim;
        const bool is3D=(meshDim==3);
        if (meshDim!=2 && meshDim!=3) {
            MessagePrinter::printErrorTxt("imported pd mesh generation currently supports 2d and 3d meshes");
            MessagePrinter::exitPeriX();
        }
        if (t_MeshData.CharacteristicLength<=0.0) {
            MessagePrinter::printErrorTxt("imported pd mesh generation requires a positive characteristic length");
            MessagePrinter::exitPeriX();
        }
        if (static_cast<int>(t_MeshData.BulkElmtVolumes.size())!=t_MeshData.BulkElmtsNum ||
            static_cast<int>(t_MeshData.BulkElmtCenters.size())!=3*t_MeshData.BulkElmtsNum) {
            MessagePrinter::printErrorTxt("imported pd mesh generation requires per-bulk centers and measures");
            MessagePrinter::exitPeriX();
        }

        // CharacteristicLength is the PD point (centroid) spacing -- see MeshImport,
        // where it is the median V^(1/dim), not the element edge -- so a
        // factor-N horizon is a true N x point-spacing horizon for tri/tet too.
        m_Data.DX=t_MeshData.CharacteristicLength;
        m_Data.DY=t_MeshData.CharacteristicLength;
        m_Data.DZ=is3D ? t_MeshData.CharacteristicLength : 0.0;

        int NodesNum=0;
        m_Data.NodeCoords.resize(m_Data.NodesNum*3,0.0);
        m_Data.NodesElmtID.resize(m_Data.NodesNum,-1);
        m_Data.NodeVolumes.resize(m_Data.NodesNum,0.0);
        m_Data.GhostMirrorBulkID.assign(m_Data.NodesNum,0);
        // per-ghost reflection-layer index, only when multi-layer is requested
        // (kept EMPTY when every wall is single-layer so that path is byte-for-byte).
        if (maxLayer>1) m_Data.GhostLayerIndex.assign(m_Data.NodesNum,0);
        else            m_Data.GhostLayerIndex.clear();
        // per-ghost CONSERVATIVE flux thickness t_g = V_bulk / face_measure (0 =>
        // fall back to the geometric |x_g-x_bulk| in the flux BC). Set on the
        // wall-adjacent (layer-1) ghosts below from the boundary element measures.
        m_Data.GhostFluxThickness.assign(m_Data.NodesNum,0.0);
        t_MeshData.BulkElmtPDNodeID.resize(m_Data.BulkElmtsNum,-1);

        for (int e=1;e<=t_MeshData.BulkElmtsNum;e++) {
            NodesNum+=1;
            const int src=3*(e-1);
            const int dst=3*(NodesNum-1);
            m_Data.NodeCoords[dst+0]=t_MeshData.BulkElmtCenters[src+0];
            m_Data.NodeCoords[dst+1]=t_MeshData.BulkElmtCenters[src+1];
            m_Data.NodeCoords[dst+2]=is3D ? t_MeshData.BulkElmtCenters[src+2] : 0.0;
            m_Data.NodesElmtID[NodesNum-1]=e;
            m_Data.NodeVolumes[NodesNum-1]=t_MeshData.BulkElmtVolumes[e-1];
            t_MeshData.BulkElmtPDNodeID[e-1]=NodesNum;
        }

        auto meshCoord=[&](const int nodeID,const int comp)->double{
            return t_MeshData.NodeCoords[(nodeID-1)*3+comp];
        };
        auto boundaryCenter=[&](const vector<int> &conn)->std::array<double,3>{
            std::array<double,3> c{0.0,0.0,0.0};
            for (const int nodeID:conn) {
                c[0]+=meshCoord(nodeID,0);
                c[1]+=meshCoord(nodeID,1);
                c[2]+=meshCoord(nodeID,2);
            }
            const double inv=1.0/static_cast<double>(conn.size());
            c[0]*=inv; c[1]*=inv; c[2]*=inv;
            return c;
        };
        auto dot=[](const std::array<double,3> &a,const std::array<double,3> &b)->double{
            return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
        };
        auto importedBoundaryIsSinglePlane=[&](const std::vector<std::array<double,3>> &normals)->bool{
            if (normals.size()<2) return true;
            std::array<double,3> ref{0.0,0.0,0.0};
            for (const auto &n : normals) {
                const double nn=std::sqrt(dot(n,n));
                if (nn<=0.0) continue;
                ref={n[0]/nn,n[1]/nn,n[2]/nn};
                break;
            }
            if (ref[0]==0.0 && ref[1]==0.0 && ref[2]==0.0) return true;
            constexpr double minParallelDot=0.999; // about 2.6 degrees
            for (const auto &n : normals) {
                const double nn=std::sqrt(dot(n,n));
                if (nn<=0.0) continue;
                const std::array<double,3> u{n[0]/nn,n[1]/nn,n[2]/nn};
                if (std::fabs(dot(ref,u))<minParallelDot) return false;
            }
            return true;
        };

        // For multi-layer ghosts (nLayer>1) we flood INWARD from each layer-1
        // wall ghost, one bulk-cell deeper per layer, mirroring the bulk cells we
        // reach across the wall plane. A cell-list hash of the bulk centroids
        // (cell size = the point spacing) gives the O(1) neighbour query;
        // built only when needed so the single-layer path allocates nothing.
        const double cellH=(m_Data.DX>0.0)?m_Data.DX:1.0;
        constexpr long ghBias=1L<<20;
        auto cellKey3=[&](const double x,const double y,const double z)->std::uint64_t{
            const long ix=std::lround(x/cellH), iy=std::lround(y/cellH), iz=is3D?std::lround(z/cellH):0L;
            return (static_cast<std::uint64_t>(ix+ghBias)<<42)
                  |(static_cast<std::uint64_t>(iy+ghBias)<<21)
                  | static_cast<std::uint64_t>(iz+ghBias);
        };
        std::unordered_map<std::uint64_t,std::vector<int>> bulkHash;
        if (maxLayer>1) {
            bulkHash.reserve(static_cast<std::size_t>(m_Data.BulkElmtsNum));
            for (int b=1;b<=m_Data.BulkElmtsNum;b++)
                bulkHash[cellKey3(m_Data.NodeCoords[(b-1)*3+0],
                                  m_Data.NodeCoords[(b-1)*3+1],
                                  m_Data.NodeCoords[(b-1)*3+2])].push_back(b);
        }
        // all bulk centroids within `radius` of (px,py,pz).
        auto collectBulks=[&](const double px,const double py,const double pz,
                              const double radius,std::vector<int> &out){
            out.clear();
            const double r2=radius*radius;
            const int span=static_cast<int>(std::floor(radius/cellH))+1;
            const long ix=std::lround(px/cellH), iy=std::lround(py/cellH), iz=is3D?std::lround(pz/cellH):0L;
            for (long ox=-span;ox<=span;ox++) for (long oy=-span;oy<=span;oy++)
                for (long oz=(is3D?-span:0);oz<=(is3D?span:0);oz++) {
                    const auto it=bulkHash.find((static_cast<std::uint64_t>(ix+ox+ghBias)<<42)
                                               |(static_cast<std::uint64_t>(iy+oy+ghBias)<<21)
                                               | static_cast<std::uint64_t>(iz+oz+ghBias));
                    if (it==bulkHash.end()) continue;
                    for (const int b:it->second) {
                        const double ex=m_Data.NodeCoords[(b-1)*3+0]-px;
                        const double ey=m_Data.NodeCoords[(b-1)*3+1]-py;
                        const double ez=is3D?(m_Data.NodeCoords[(b-1)*3+2]-pz):0.0;
                        if (ex*ex+ey*ey+ez*ez<=r2) out.push_back(b);
                    }
                }
        };

        // a layer-1 wall ghost carries the plane (wall point w + outward normal n)
        // that its deeper mirror layers reflect across.
        struct Seed { int b; std::array<double,3> w; std::array<double,3> n; };

        std::unordered_map<std::string,vector<int>> ghostNodeLists;
        bool anyLayerIndexedGhosts=false;
        for (const auto &[phyname, connVecPtr] : boundaryGroups) {
            const auto &connVec=*connVecPtr;
            int nLayer=nLayerFor(phyname);           // this wall's mirror-layer count
            const auto adjIt=t_MeshData.PhyGroupName2BoundaryBulkElmtIDVecMap.find(phyname);
            const auto normalIt=t_MeshData.PhyGroupName2BoundaryNormalVecMap.find(phyname);
            if (adjIt==t_MeshData.PhyGroupName2BoundaryBulkElmtIDVecMap.end() ||
                normalIt==t_MeshData.PhyGroupName2BoundaryNormalVecMap.end()) {
                MessagePrinter::printErrorTxt("imported boundary physical group '"+phyname
                                              +"' has no boundary-to-bulk mirror metadata");
                MessagePrinter::exitPeriX();
            }
            const auto &adjacentBulkIDs=adjIt->second;
            const auto &normals=normalIt->second;
            if (adjacentBulkIDs.size()!=connVec.size() || normals.size()!=connVec.size()) {
                MessagePrinter::printErrorTxt("imported boundary physical group '"+phyname
                                              +"' has inconsistent mirror metadata");
                MessagePrinter::exitPeriX();
            }
            if (nLayer>1 && !importedBoundaryIsSinglePlane(normals)) {
                MessagePrinter::printWarningTxt(
                    "PDMesh.ghost_layer['"+phyname+"']="+std::to_string(nLayer)
                    +" was requested on a non-planar/curved imported boundary; using 1 layer. "
                     "Multi-layer mirror ghosts require one reflection plane per physical group.");
                nLayer=1;
            }
            if (nLayer>1) anyLayerIndexedGhosts=true;
            // per-boundary-face measure (length in 2D, area in 3D), aligned with
            // connVec/adjacentBulkIDs/normals (MeshImport pushes them together).
            // Used to set the wall-adjacent ghost's CONSERVATIVE flux thickness
            // V_bulk/face_measure; absent/empty => leave it 0 (geometric fallback).
            const auto measureIt=t_MeshData.PhyGroupName2BoundaryMeasureVecMap.find(phyname);
            const std::vector<double> *measures=
                (measureIt!=t_MeshData.PhyGroupName2BoundaryMeasureVecMap.end()
                 && measureIt->second.size()==connVec.size()) ? &measureIt->second : nullptr;

            // "<name>nodes_ghost" = the ghost (fictitious) PD points; "<name>nodes"
            // = the real boundary PD nodes (their mirror bulks, i.e. the cell
            // centres adjacent to the wall). BCs reference the ghost set.
            auto &ghostlist=ghostNodeLists[phyname+"nodes_ghost"];
            auto &reallist=ghostNodeLists[phyname+"nodes"];
            ghostlist.reserve(static_cast<std::size_t>(nLayer)*connVec.size());
            reallist.reserve(static_cast<std::size_t>(nLayer)*connVec.size());
            // bulk cells already mirrored on THIS wall (so a bulk shared by two
            // faces' inward columns is mirrored once); seeded by the layer-1 pass.
            std::unordered_set<int> mirrored;
            std::vector<Seed> seeds;          // layer-1 wall ghosts -> flood origin

            for (std::size_t k=0;k<connVec.size();k++) {
                const int bulkID=adjacentBulkIDs[k];
                if (bulkID<1 || bulkID>m_Data.BulkElmtsNum) {
                    MessagePrinter::printErrorTxt("imported boundary physical group '"+phyname
                                                  +"' references an invalid adjacent bulk element");
                    MessagePrinter::exitPeriX();
                }
                std::array<double,3> n=normals[k];
                const auto bc=boundaryCenter(connVec[k]);
                const int bidx=3*(bulkID-1);
                const std::array<double,3> bulkCenter{
                    m_Data.NodeCoords[bidx+0],
                    m_Data.NodeCoords[bidx+1],
                    m_Data.NodeCoords[bidx+2]};
                std::array<double,3> bulkMinusBoundary{
                    bulkCenter[0]-bc[0],
                    bulkCenter[1]-bc[1],
                    bulkCenter[2]-bc[2]};
                double signedDistance=dot(bulkMinusBoundary,n);
                if (signedDistance>0.0) {
                    n[0]*=-1.0; n[1]*=-1.0; n[2]*=-1.0;
                    signedDistance=-signedDistance;
                }
                if (std::fabs(signedDistance)<=0.0) {
                    MessagePrinter::printErrorTxt("imported boundary physical group '"+phyname
                                                  +"' produced a zero ghost reflection distance");
                    MessagePrinter::exitPeriX();
                }

                NodesNum+=1;
                const int gidx=3*(NodesNum-1);
                m_Data.NodeCoords[gidx+0]=bulkCenter[0]-2.0*signedDistance*n[0];
                m_Data.NodeCoords[gidx+1]=bulkCenter[1]-2.0*signedDistance*n[1];
                m_Data.NodeCoords[gidx+2]=is3D ? bulkCenter[2]-2.0*signedDistance*n[2] : 0.0;
                m_Data.NodesElmtID[NodesNum-1]=-1;
                m_Data.NodeVolumes[NodesNum-1]=m_Data.NodeVolumes[bulkID-1];
                m_Data.GhostMirrorBulkID[NodesNum-1]=bulkID;
                // conservative flux thickness t_g = V_bulk / face_measure: a surface
                // flux density j becomes the source S = j/t_g on the mirror bulk so
                // that sum S*V_bulk = j*face_measure per face = j*Area exactly (the
                // geometric |x_g-x_bulk|=2*signedDistance over-/under-smears on a
                // simplex centroid; only V/measure conserves on any cell shape).
                if (measures!=nullptr) {
                    const double meas=(*measures)[k];
                    if (meas>0.0) m_Data.GhostFluxThickness[NodesNum-1]=m_Data.NodeVolumes[bulkID-1]/meas;
                }
                if (nLayer>1) {
                    m_Data.GhostLayerIndex[NodesNum-1]=1;
                    mirrored.insert(bulkID);
                    seeds.push_back(Seed{bulkID,bc,n});   // n is the outward normal, bc on the wall
                }
                ghostlist.push_back(NodesNum);
                reallist.push_back(bulkID);
            }

            // ---- deeper mirror layers (ghost_layer>1) ----
            // One bounded ghost COLUMN per boundary face, grown along that face's
            // own outward normal n (computed from the user-defined boundary FE
            // mesh -- the reliable, well-defined wall direction). Starting from
            // the layer-1 wall bulk, step ~one point-spacing deeper along -n and
            // pick the bulk cell best aligned with that inward ray (nearest to the
            // ray, ~one spacing in); mirror it across the face plane to the ghost
            // at the symmetric outside position. Exactly one bulk is taken per
            // layer per face (no lateral fan-out), so the total stays within the
            // bulk + nLayer*faces budget; a bulk already mirrored on this wall is
            // skipped (de-dup across adjacent faces' columns). Each ghost is the
            // exact planar mirror of its own bulk, so its outward normal
            // (ghost-bulk)/|.| equals n and the reflection/strong-form BCs are
            // exact for every layer.
            if (nLayer>1) {
                std::vector<int> cand;
                for (const Seed &s : seeds) {
                    int cur=s.b;                                  // current column bulk
                    std::array<double,3> curc{m_Data.NodeCoords[(cur-1)*3+0],
                                              m_Data.NodeCoords[(cur-1)*3+1],
                                              m_Data.NodeCoords[(cur-1)*3+2]};
                    for (int layer=2; layer<=nLayer; layer++) {
                        if (NodesNum>=m_Data.NodesNum) break;     // never exceed the pre-sized budget
                        // probe one point-spacing inward along the wall normal and
                        // take the bulk best matching "~one cell deeper, on the
                        // inward ray". Scoring by |inward - spacing| + lateral lets
                        // the column step past a laterally-zig-zagged tri centroid
                        // (rigid nearest-neighbour stalls there), so each face grows
                        // a full ghost column spanning roughly the horizon.
                        collectBulks(curc[0]-cellH*s.n[0], curc[1]-cellH*s.n[1],
                                     is3D?(curc[2]-cellH*s.n[2]):0.0, 1.1*cellH, cand);
                        int best=-1; double bestScore=1.0e300; std::array<double,3> bestc{0,0,0};
                        for (const int C : cand) {
                            if (C==cur) continue;
                            const std::array<double,3> Cc{m_Data.NodeCoords[(C-1)*3+0],
                                                          m_Data.NodeCoords[(C-1)*3+1],
                                                          m_Data.NodeCoords[(C-1)*3+2]};
                            const std::array<double,3> v{Cc[0]-curc[0],Cc[1]-curc[1],Cc[2]-curc[2]};
                            const double inward=-dot(v,s.n);               // >0 => deeper than cur
                            if (inward<0.35*cellH) continue;              // must move inward
                            const double lx=v[0]+inward*s.n[0], ly=v[1]+inward*s.n[1], lz=v[2]+inward*s.n[2];
                            const double lateral=std::sqrt(lx*lx+ly*ly+lz*lz);
                            if (lateral>0.85*cellH) continue;            // keep ~on the inward ray
                            const double score=lateral+std::fabs(inward-cellH);
                            if (score<bestScore) { bestScore=score; best=C; bestc=Cc; }
                        }
                        if (best<0) break;                                // column ended
                        // advance the column even if this bulk was already mirrored
                        // by an adjacent face (just don't add a duplicate ghost).
                        if (!mirrored.insert(best).second) { cur=best; curc=bestc; continue; }
                        const double depthBest=dot(std::array<double,3>{s.w[0]-bestc[0],s.w[1]-bestc[1],s.w[2]-bestc[2]},s.n);
                        if (depthBest<=0.0) { cur=best; curc=bestc; continue; }
                        NodesNum+=1;
                        const int gx2=3*(NodesNum-1);
                        m_Data.NodeCoords[gx2+0]=bestc[0]+2.0*depthBest*s.n[0];
                        m_Data.NodeCoords[gx2+1]=bestc[1]+2.0*depthBest*s.n[1];
                        m_Data.NodeCoords[gx2+2]=is3D?(bestc[2]+2.0*depthBest*s.n[2]):0.0;
                        m_Data.NodesElmtID[NodesNum-1]=-1;
                        m_Data.NodeVolumes[NodesNum-1]=m_Data.NodeVolumes[best-1];
                        m_Data.GhostMirrorBulkID[NodesNum-1]=best;
                        m_Data.GhostLayerIndex[NodesNum-1]=layer;
                        ghostlist.push_back(NodesNum);
                        reallist.push_back(best);
                        cur=best; curc=bestc;
                    }
                }
            }
        }

        if (NodesNum>m_Data.NodesNum) {
            MessagePrinter::printErrorTxt("the number of imported pd nodes exceeds the expected bulk+ghost upper bound");
            MessagePrinter::exitPeriX();
        }
        // Shrink to the ACTUAL node count: deeper ghost layers de-duplicate and
        // may run out of inward bulk, so fewer than the bulk+nLayer*faces upper
        // bound are created. At nLayer==1 this is a no-op (count == upper bound).
        m_Data.NodesNum=NodesNum;
        m_Data.NodeCoords.resize(static_cast<std::size_t>(NodesNum)*3);
        m_Data.NodesElmtID.resize(static_cast<std::size_t>(NodesNum));
        m_Data.NodeVolumes.resize(static_cast<std::size_t>(NodesNum));
        m_Data.GhostMirrorBulkID.resize(static_cast<std::size_t>(NodesNum));
        m_Data.GhostFluxThickness.resize(static_cast<std::size_t>(NodesNum));
        if (!anyLayerIndexedGhosts) {
            m_Data.GhostLayerIndex.clear();
        }
        else if (!m_Data.GhostLayerIndex.empty())
            m_Data.GhostLayerIndex.resize(static_cast<std::size_t>(NodesNum));
        m_Data.PhyNameToNodeIDsMap.clear();
        for (auto &[name,ids] : ghostNodeLists) {
            std::sort(ids.begin(),ids.end());
            ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
            m_Data.PhyNameToNodeIDsMap[name]=std::move(ids);
        }

        // Preserve full-dimensional imported physical groups as PD-node sets.
        // A generated point cloud can use such a group for a minimal rigid-mode
        // constraint without a coordinate filter: its bulk-element ids map
        // directly through BulkElmtPDNodeID to the corresponding material
        // points. Boundary groups remain handled above as reflected ghost sets.
        for (int i=0;i<t_MeshData.PhyGroupNum;++i) {
            if (t_MeshData.PhyGroupDimVec[static_cast<std::size_t>(i)]
                    !=t_MeshData.MeshDim) {
                continue;
            }
            const std::string &name=
                t_MeshData.PhyGroupNameVec[static_cast<std::size_t>(i)];
            if (name=="alldomain") continue;
            const auto groupIt=
                t_MeshData.PhyGroupName2BulkElmtIDVecMap.find(name);
            if (groupIt==t_MeshData.PhyGroupName2BulkElmtIDVecMap.end()) {
                continue;
            }

            std::vector<int> ids;
            ids.reserve(groupIt->second.size());
            for (const int bulkID : groupIt->second) {
                if (bulkID<1 || bulkID>t_MeshData.BulkElmtsNum) {
                    MessagePrinter::printErrorTxt(
                        "imported bulk physical group '"+name
                        +"' references an invalid bulk element");
                    MessagePrinter::exitPeriX();
                }
                const int nodeID=
                    t_MeshData.BulkElmtPDNodeID[
                        static_cast<std::size_t>(bulkID-1)];
                if (nodeID>0) ids.push_back(nodeID);
            }
            std::sort(ids.begin(),ids.end());
            ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
            m_Data.PhyNameToNodeIDsMap[name]=std::move(ids);
        }

        std::vector<int> all(static_cast<std::size_t>(m_Data.NodesNum));
        for (int i=0;i<m_Data.NodesNum;i++) all[static_cast<std::size_t>(i)]=i+1;
        m_Data.PhyNameToNodeIDsMap["alldomain"]=std::move(all);
        computeNodeNormals();
        return;
    }

    // here we assume the grid size is uniform for simplification
    // TODO: add the non-uniform treatment
    //
    // The PD background mesh is an axis-aligned finite element mesh:
    //   2D -> quad4 cells, ghost rings on the 4 edges (left/right/bottom/top)
    //   3D -> hex8  cells, ghost rings on the 6 faces
    //         (left/right/bottom/top/back/front)
    // One bulk PD point sits at every cell centre; every boundary face/edge
    // spawns one ghost PD point half a cell OUTSIDE the wall, at the exact
    // mirror of the adjacent bulk centre (see the ghost block below).
    const int  meshDim=t_MeshData.MeshDim;
    const bool is3D=(meshDim==3);
    const int  connSize=static_cast<int>(t_MeshData.BulkElmtConn[0].size());
    if (is3D) {
        if (connSize<8) {
            MessagePrinter::printErrorTxt("3d pd mesh generation requires hex8 bulk elements");
            MessagePrinter::exitPeriX();
        }
    }
    else {
        if (connSize<4) {
            MessagePrinter::printErrorTxt("structured 2d pd mesh generation requires quad4 bulk elements; use gmsh mesh import for tri3 meshes");
            MessagePrinter::exitPeriX();
        }
    }

    // coordinate accessor: component c (0=x,1=y,2=z) of mesh node id (1-based)
    auto X=[&](const int nodeId,const int c)->double{
        return t_MeshData.NodeCoords[(nodeId-1)*3+c];
    };

    // uniform cell sizes from the first bulk element. quad4 node order is
    // (i1,i2,i3,i4) CCW; hex8 is (i1..i4 back face, i5..i8 front face) with
    // i1->i2 = +x, i1->i4 = +y, i1->i5 = +z.
    {
        const auto &C=t_MeshData.BulkElmtConn[0];
        if (is3D) {
            m_Data.DX=std::abs(X(C[1],0)-X(C[0],0));
            m_Data.DY=std::abs(X(C[3],1)-X(C[0],1));
            m_Data.DZ=std::abs(X(C[4],2)-X(C[0],2));
        }
        else {
            m_Data.DX=std::abs(X(C[1],0)-X(C[0],0));
            m_Data.DY=std::abs(X(C[2],1)-X(C[1],1));
            m_Data.DZ=0.0;
        }
    }
    if (m_Data.DX<=0.0 || m_Data.DY<=0.0 || (is3D && m_Data.DZ<=0.0)) {
        MessagePrinter::printErrorTxt("pd mesh generation requires positive uniform cell size");
        MessagePrinter::exitPeriX();
    }
    const double cellVolume=is3D ? m_Data.DX*m_Data.DY*m_Data.DZ
                                 : m_Data.DX*m_Data.DY;

    int NodesNum=0;
    m_Data.NodeCoords.resize(m_Data.NodesNum*3,0.0);
    m_Data.NodesElmtID.resize(m_Data.NodesNum,-1);
    m_Data.NodeVolumes.resize(m_Data.NodesNum,0.0);
    t_MeshData.BulkElmtPDNodeID.resize(m_Data.BulkElmtsNum,-1);
    for (int e=1;e<=t_MeshData.BulkElmtsNum;e++) {
        const auto &C=t_MeshData.BulkElmtConn[e-1];
        NodesNum+=1;
        // PD point at the cell centre = midpoint of two opposite corners
        // (quad4: i1 & i3; hex8: i1 & i7).
        const int iA=C[0];
        const int iC=is3D ? C[6] : C[2];
        m_Data.NodeCoords[(NodesNum-1)*3+0]=0.5*(X(iA,0)+X(iC,0));
        m_Data.NodeCoords[(NodesNum-1)*3+1]=0.5*(X(iA,1)+X(iC,1));
        m_Data.NodeCoords[(NodesNum-1)*3+2]=is3D ? 0.5*(X(iA,2)+X(iC,2)) : 0.0;
        m_Data.NodesElmtID[NodesNum-1]=e;
        m_Data.NodeVolumes[NodesNum-1]=cellVolume;
        t_MeshData.BulkElmtPDNodeID[e-1]=NodesNum;
    }

    // build a position -> bulk PD ID hash map so that ghost nodes can resolve
    // their mirror bulk in O(1). The key is the integer cell index (ix,iy,iz)
    // of a point on the (DX/2,DY/2,DZ/2)-shifted cell-centre grid, packed with
    // a bias so the small negative indices of ghost mirrors stay positive.
    auto positionKey=[&](const double x,const double y,const double z)->std::uint64_t{
        const long ix=std::lround((x-0.5*m_Data.DX)/m_Data.DX);
        const long iy=std::lround((y-0.5*m_Data.DY)/m_Data.DY);
        const long iz=is3D ? std::lround((z-0.5*m_Data.DZ)/m_Data.DZ) : 0L;
        constexpr long bias=1L<<20;            // ghost indices are >= -1, stay positive
        const std::uint64_t ux=static_cast<std::uint64_t>(ix+bias);
        const std::uint64_t uy=static_cast<std::uint64_t>(iy+bias);
        const std::uint64_t uz=static_cast<std::uint64_t>(iz+bias);
        return (ux<<42)|(uy<<21)|uz;           // 21 bits per axis (grid <= ~2M/axis)
    };
    std::unordered_map<std::uint64_t,int> bulkPositionToID;
    bulkPositionToID.reserve(static_cast<std::size_t>(m_Data.BulkElmtsNum));
    for (int b=1;b<=m_Data.BulkElmtsNum;++b) {
        bulkPositionToID.emplace(positionKey(m_Data.NodeCoords[(b-1)*3+0],
                                             m_Data.NodeCoords[(b-1)*3+1],
                                             m_Data.NodeCoords[(b-1)*3+2]),b);
    }

    m_Data.GhostMirrorBulkID.assign(m_Data.NodesNum,0);
    // per-ghost CONSERVATIVE flux thickness t_g=V_bulk/face_measure (0 => geometric
    // fallback). Axis-aligned walls leave it 0 (their geometric ghost-bulk distance
    // DX already equals V/measure and is exactly conservative); the CURVED-arc branch
    // stamps V_bulk/arc_measure below, so a flux BC on a generator quarter-disk arc
    // injects j*arc instead of the ~25%-biased centroid-reflection smear.
    m_Data.GhostFluxThickness.assign(m_Data.NodesNum,0.0);
    // per-ghost reflection-layer index, only when multi-layer is requested
    // (kept EMPTY when every wall is single-layer so that path is byte-for-byte).
    if (maxLayer>1) m_Data.GhostLayerIndex.assign(m_Data.NodesNum,0);
    else            m_Data.GhostLayerIndex.clear();

    // now we take care of the boundary (ghost) pd points. Each boundary group
    // ("left","right",... in 2D plus "back","front" in 3D) maps to one wall;
    // its ghost ids are stored under "<name>nodes" for the BC system.
    std::unordered_map<std::string,vector<int>> ghostNodeLists;
    for (const auto &[phyname, connVecPtr] : boundaryGroups) {
        const auto &connVec=*connVecPtr;

        // ---- curved boundary (e.g. the quarter-circle "arc") ----
        // A boundary group carrying EXPLICIT per-face outward normals and
        // adjacent-bulk ids is not an axis-aligned wall: reflect each adjacent
        // bulk PD point across its own face plane to a single exterior ghost
        // (a radial mirror), exactly as the imported path does. Built-in axis
        // walls (left/right/bottom/top) never populate these maps, so their
        // handling below is byte-for-byte unchanged. Single layer (curved).
        {
            const auto nmIt=t_MeshData.PhyGroupName2BoundaryNormalVecMap.find(phyname);
            const auto abIt=t_MeshData.PhyGroupName2BoundaryBulkElmtIDVecMap.find(phyname);
            if (nmIt!=t_MeshData.PhyGroupName2BoundaryNormalVecMap.end() &&
                abIt!=t_MeshData.PhyGroupName2BoundaryBulkElmtIDVecMap.end()) {
                const auto &normals=nmIt->second;
                const auto &adjBulk=abIt->second;
                if (normals.size()!=connVec.size() || adjBulk.size()!=connVec.size()) {
                    MessagePrinter::printErrorTxt("curved boundary group '"+phyname
                        +"' has inconsistent per-face normal / adjacent-bulk metadata");
                    MessagePrinter::exitPeriX();
                }
                // per-face measure (arc length in 2D, surface patch in 3D) for the
                // CONSERVATIVE flux thickness t_g=V_bulk/face_measure, exactly as the
                // imported path. On a curved wall the geometric ghost-bulk distance
                // (2*signedDistance, centroid reflection) mis-smears the radial flux
                // (~25% on a simplex); V/measure conserves j*arc on any cell shape.
                // Absent/empty => leave GhostFluxThickness 0 (geometric fallback).
                const auto cMeasIt=t_MeshData.PhyGroupName2BoundaryMeasureVecMap.find(phyname);
                const std::vector<double> *cMeasures=
                    (cMeasIt!=t_MeshData.PhyGroupName2BoundaryMeasureVecMap.end()
                     && cMeasIt->second.size()==connVec.size()) ? &cMeasIt->second : nullptr;
                auto &ghostlist=ghostNodeLists[phyname+"nodes_ghost"];
                auto &reallist =ghostNodeLists[phyname+"nodes"];
                ghostlist.reserve(connVec.size());
                reallist.reserve(connVec.size());
                for (std::size_t k=0;k<connVec.size();k++) {
                    const auto &Conn=connVec[k];
                    if (Conn.empty()) continue;
                    // face centre = average of its node coords (single node for
                    // the arc; it sits ON the boundary, at radius R).
                    double fx=0.0,fy=0.0,fz=0.0;
                    for (const int nId:Conn) { fx+=X(nId,0); fy+=X(nId,1); fz+=X(nId,2); }
                    const double invn=1.0/static_cast<double>(Conn.size());
                    fx*=invn; fy*=invn; fz*=invn;
                    const int bulkID=adjBulk[k];
                    if (bulkID<1 || bulkID>m_Data.BulkElmtsNum) {
                        MessagePrinter::printErrorTxt("curved boundary group '"+phyname
                            +"' references an invalid adjacent bulk element");
                        MessagePrinter::exitPeriX();
                    }
                    const int bidx=3*(bulkID-1);
                    const double bx=m_Data.NodeCoords[bidx+0];
                    const double by=m_Data.NodeCoords[bidx+1];
                    const double bz=m_Data.NodeCoords[bidx+2];
                    double nx=normals[k][0], ny=normals[k][1], nz=is3D?normals[k][2]:0.0;
                    double sd=(bx-fx)*nx+(by-fy)*ny+(bz-fz)*nz; // signed distance
                    if (sd>0.0) { nx=-nx; ny=-ny; nz=-nz; sd=-sd; }
                    if (std::fabs(sd)<=0.0) {
                        MessagePrinter::printErrorTxt("curved boundary group '"+phyname
                            +"' produced a zero ghost reflection distance");
                        MessagePrinter::exitPeriX();
                    }
                    if (NodesNum>=m_Data.NodesNum) break; // stay within the budget
                    NodesNum+=1;
                    const int gidx=3*(NodesNum-1);
                    m_Data.NodeCoords[gidx+0]=bx-2.0*sd*nx;
                    m_Data.NodeCoords[gidx+1]=by-2.0*sd*ny;
                    m_Data.NodeCoords[gidx+2]=is3D?(bz-2.0*sd*nz):0.0;
                    m_Data.NodesElmtID[NodesNum-1]=-1;
                    m_Data.NodeVolumes[NodesNum-1]=cellVolume;
                    m_Data.GhostMirrorBulkID[NodesNum-1]=bulkID;
                    // conservative flux thickness t_g = V_bulk / arc-face measure
                    if (cMeasures!=nullptr) {
                        const double meas=(*cMeasures)[k];
                        if (meas>0.0) m_Data.GhostFluxThickness[NodesNum-1]=m_Data.NodeVolumes[bulkID-1]/meas;
                    }
                    ghostlist.push_back(NodesNum);
                    reallist.push_back(bulkID);
                }
                continue; // curved group done; skip the axis-wall handling
            }
        }

        const int nLayer=nLayerFor(phyname);     // this wall's mirror-layer count

        // map the boundary name to its outward wall normal: which axis it is
        // (0=x,1=y,2=z) and the sign of the ghost offset (ghost is OUTSIDE).
        int axis=-1; double ghostSign=0.0;
        if      (phyname=="left")  { axis=0; ghostSign=-1.0; }
        else if (phyname=="right") { axis=0; ghostSign=+1.0; }
        else if (phyname=="bottom"){ axis=1; ghostSign=-1.0; }
        else if (phyname=="top")   { axis=1; ghostSign=+1.0; }
        else if (phyname=="back")  { axis=2; ghostSign=-1.0; }
        else if (phyname=="front") { axis=2; ghostSign=+1.0; }
        else {
            MessagePrinter::printErrorTxt("unsupported boundary physical group '"+phyname
                                          +"' for pd mesh generation");
            MessagePrinter::exitPeriX();
        }
        if (axis==2 && !is3D) {
            MessagePrinter::printErrorTxt("back/front boundary groups require a 3d (hex8) pd mesh");
            MessagePrinter::exitPeriX();
        }
        const double half=0.5*(axis==0 ? m_Data.DX : (axis==1 ? m_Data.DY : m_Data.DZ));

        // "<name>nodes_ghost" = the ghost (fictitious) PD points outside the wall;
        // "<name>nodes" = the real boundary PD nodes (their mirror bulks, i.e. the
        // cell centres adjacent to the wall). BCs reference the ghost set.
        auto &ghostlist=ghostNodeLists[phyname+"nodes_ghost"];
        auto &reallist=ghostNodeLists[phyname+"nodes"];
        ghostlist.reserve(static_cast<std::size_t>(nLayer)*connVec.size());
        reallist.reserve(static_cast<std::size_t>(nLayer)*connVec.size());
        // bulk cells already mirrored on THIS wall (de-dup deeper-layer columns).
        std::unordered_set<int> mirrored;

        for (const auto &Conn:connVec) {
            const int nface=static_cast<int>(Conn.size());
            if (nface<2) {
                MessagePrinter::printErrorTxt("boundary connectivity must contain at least two nodes for pd mesh generation");
                MessagePrinter::exitPeriX();
            }
            NodesNum+=1;
            const int idx=(NodesNum-1)*3;

            // face/edge centre = average of its corner nodes; it sits ON the
            // wall (quad4 edge = 2 nodes, hex8 face = 4 nodes).
            double cx=0.0,cy=0.0,cz=0.0;
            for (const int n:Conn) { cx+=X(n,0); cy+=X(n,1); cz+=X(n,2); }
            cx/=nface; cy/=nface; cz/=nface;

            // Ghost (fictitious) PD point. The boundary face sits ON the wall
            // and the adjacent bulk PD point is half a cell INSIDE it (cell
            // centre). Place the ghost half a cell OUTSIDE, i.e. at the exact
            // mirror of that bulk centre across the wall, so ghost and bulk are
            // equidistant from the boundary. Mirror-method natural BCs set the
            // ghost value from
            // this mirror bulk; only with an equidistant ghost/bulk does the
            // reflected field become even about the wall, enforcing zero normal
            // flux EXACTLY at the boundary (a full-cell offset would shift the
            // effective wall by a quarter cell and lose first order there).
            double gx=cx,gy=cy,gz=cz;          // ghost coords (outside the wall)
            double mx=cx,my=cy,mz=cz;          // mirror bulk coords (inside)
            if (axis==0)      { gx=cx+ghostSign*half; mx=cx-ghostSign*half; }
            else if (axis==1) { gy=cy+ghostSign*half; my=cy-ghostSign*half; }
            else              { gz=cz+ghostSign*half; mz=cz-ghostSign*half; }

            m_Data.NodeCoords[idx+0]=gx;
            m_Data.NodeCoords[idx+1]=gy;
            m_Data.NodeCoords[idx+2]=is3D ? gz : 0.0;
            m_Data.NodesElmtID[NodesNum-1]=-1;// boundary pd point: parent element id=-1
            m_Data.NodeVolumes[NodesNum-1]=cellVolume;
            ghostlist.push_back(NodesNum);

            // record the mirror bulk PD node id for this ghost and add it to
            // the real boundary-node set.
            const auto mirrorIt=bulkPositionToID.find(positionKey(mx,my,mz));
            if (mirrorIt!=bulkPositionToID.end()) {
                m_Data.GhostMirrorBulkID[NodesNum-1]=mirrorIt->second;
                if (nLayer>1) { m_Data.GhostLayerIndex[NodesNum-1]=1; mirrored.insert(mirrorIt->second); }
                reallist.push_back(mirrorIt->second);
            }
        }

        // ---- deeper mirror layers (ghost_layer>1) ----
        // The bulk column behind a wall face is the exact axis-aligned stack of
        // cell centres; layer L mirrors the cell (2L-1) half-cells inside the
        // wall to a ghost (2L-1) half-cells outside it. Look the cell up in the
        // same position hash the layer-1 mirror uses; stop when the column ends.
        if (nLayer>1) {
            for (const auto &Conn:connVec) {
                const int nf=static_cast<int>(Conn.size());
                if (nf<2) continue;
                double cx=0.0,cy=0.0,cz=0.0;
                for (const int n:Conn) { cx+=X(n,0); cy+=X(n,1); cz+=X(n,2); }
                cx/=nf; cy/=nf; cz/=nf;
                for (int layer=2;layer<=nLayer;layer++) {
                    if (NodesNum>=m_Data.NodesNum) break;        // never exceed the pre-sized budget
                    const double off=half*static_cast<double>(2*layer-1);
                    double mx=cx,my=cy,mz=cz,gx=cx,gy=cy,gz=cz;
                    if      (axis==0) { mx=cx-ghostSign*off; gx=cx+ghostSign*off; }
                    else if (axis==1) { my=cy-ghostSign*off; gy=cy+ghostSign*off; }
                    else              { mz=cz-ghostSign*off; gz=cz+ghostSign*off; }
                    const auto it=bulkPositionToID.find(positionKey(mx,my,mz));
                    if (it==bulkPositionToID.end()) break;        // column ended
                    const int bk=it->second;
                    if (!mirrored.insert(bk).second) continue;    // already mirrored on this wall
                    NodesNum+=1;
                    const int idx=(NodesNum-1)*3;
                    m_Data.NodeCoords[idx+0]=gx;
                    m_Data.NodeCoords[idx+1]=gy;
                    m_Data.NodeCoords[idx+2]=is3D?gz:0.0;
                    m_Data.NodesElmtID[NodesNum-1]=-1;
                    m_Data.NodeVolumes[NodesNum-1]=cellVolume;
                    m_Data.GhostMirrorBulkID[NodesNum-1]=bk;
                    m_Data.GhostLayerIndex[NodesNum-1]=layer;
                    ghostlist.push_back(NodesNum);
                    reallist.push_back(bk);
                }
            }
        }
    }

    if (NodesNum>m_Data.NodesNum) {
        MessagePrinter::printErrorTxt("the number of pd nodes exceeds the expected bulk+ghost upper bound");
        MessagePrinter::exitPeriX();
    }
    // Shrink to the ACTUAL node count (deeper ghost layers de-duplicate / run
    // out of inward bulk). At nLayer==1 this is a no-op (count == upper bound).
    m_Data.NodesNum=NodesNum;
    m_Data.NodeCoords.resize(static_cast<std::size_t>(NodesNum)*3);
    m_Data.NodesElmtID.resize(static_cast<std::size_t>(NodesNum));
    m_Data.NodeVolumes.resize(static_cast<std::size_t>(NodesNum));
    m_Data.GhostMirrorBulkID.resize(static_cast<std::size_t>(NodesNum));
    m_Data.GhostFluxThickness.resize(static_cast<std::size_t>(NodesNum));
    if (!m_Data.GhostLayerIndex.empty())
        m_Data.GhostLayerIndex.resize(static_cast<std::size_t>(NodesNum));

    m_Data.PhyNameToNodeIDsMap.clear();
    for (auto &[name,ids] : ghostNodeLists) {
        std::sort(ids.begin(),ids.end());
        ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        m_Data.PhyNameToNodeIDsMap[name]=std::move(ids);
    }

    // "alldomain": every PD node (bulk cell centres + boundary ghosts), 1-based.
    // The boundary ("<wall>nodes") groups select edges; "alldomain" lets a BC
    // reach an interior region by combining it with a coordinate box -- e.g. a
    // symmetry plane pinned by u_x=0 down the centre column of a tensile plate.
    {
        std::vector<int> all(static_cast<std::size_t>(m_Data.NodesNum));
        for (int i=0;i<m_Data.NodesNum;++i) all[static_cast<std::size_t>(i)]=i+1;
        m_Data.PhyNameToNodeIDsMap["alldomain"]=std::move(all);
    }

    computeNodeNormals();
}

void PDMesh::computeNodeNormals() {
    // Outward unit normal at every PD node. A boundary ghost was placed as the
    // exact planar mirror of its adjacent bulk centroid across the wall, so
    //   n_g = (x_ghost - x_mirrorbulk)/|x_ghost - x_mirrorbulk|
    // recovers the outward normal of the (d-1)-D boundary finite element that
    // spawned the ghost (axis-aligned wall -> +/-e_axis; imported wall -> the
    // sign-fixed MeshImport::boundaryNormal). This is computed with the SAME
    // arithmetic the natural-BC rows used to do inline (subtract, hypot, divide
    // on the stored coordinates), so the cached normal is bit-for-bit identical
    // to the old on-the-fly value. Bulk nodes and ghosts with no mirror bulk
    // (GhostMirrorBulkID < 1) keep the {0,0,0} sentinel.
    const int n=m_Data.NodesNum;
    m_Data.NodeNormal.assign(static_cast<std::size_t>(std::max(n,0)),
                             std::array<double,3>{0.0,0.0,0.0});
    if (n<=0 || static_cast<int>(m_Data.GhostMirrorBulkID.size())<n) return;
    for (int g=1;g<=n;++g) {
        const int B=m_Data.GhostMirrorBulkID[static_cast<std::size_t>(g-1)];
        if (B<1) continue;                               // bulk node or unmapped ghost
        const double nv0=m_Data.NodeCoords[(g-1)*3+0]-m_Data.NodeCoords[(B-1)*3+0];
        const double nv1=m_Data.NodeCoords[(g-1)*3+1]-m_Data.NodeCoords[(B-1)*3+1];
        const double nv2=m_Data.NodeCoords[(g-1)*3+2]-m_Data.NodeCoords[(B-1)*3+2];
        const double nn=std::sqrt(nv0*nv0+nv1*nv1+nv2*nv2);
        if (nn<=0.0) continue;                           // degenerate (never for a real ghost)
        m_Data.NodeNormal[static_cast<std::size_t>(g-1)]={nv0/nn,nv1/nn,nv2/nn};
    }
}

void PDMesh::computeVariableHorizon() {
    // Off by default: leave NodeHorizon/NodeSpacing empty so every consumer
    // takes the global single-horizon path (byte-for-byte unchanged).
    m_Data.NodeHorizon.clear();
    m_Data.NodeSpacing.clear();
    if (!m_Data.VariableHorizon) return;

    const int n = m_Data.NodesNum;
    if (n <= 0) return;
    if (m_Data.HorizonRadius <= 0.0) {
        MessagePrinter::printErrorTxt("PDMesh::computeVariableHorizon: HorizonRadius must be > 0");
        MessagePrinter::exitPeriX();
    }
    if (static_cast<int>(m_Data.NodeVolumes.size()) < n) {
        MessagePrinter::printErrorTxt("PDMesh::computeVariableHorizon must run after createPDMesh "
                                      "(NodeVolumes not populated)");
        MessagePrinter::exitPeriX();
    }

    const bool   is3D    = (m_Data.DZ > 0.0);
    const double invDim  = is3D ? (1.0/3.0) : 0.5;        // s_i = V_i^(1/dim)
    const double dxChar  = std::max(m_Data.DX, std::max(m_Data.DY, m_Data.DZ));

    // Per-node effective cell size s_i = V_i^(1/dim).
    std::vector<double> s(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        const double v = m_Data.NodeVolumes[static_cast<std::size_t>(i)];
        s[static_cast<std::size_t>(i)] = (v > 0.0) ? std::pow(v, invDim) : 0.0;
    }

    // Reference spacing = median of s_i over BULK nodes (a ghost reuses its
    // mirror bulk's volume, so it would bias the median toward the boundary).
    // Anchoring to the median makes delta_i == HorizonRadius for a uniform mesh,
    // so the existing global calibration (HorizonRadius = factor * median edge)
    // is preserved exactly.
    std::vector<double> sb;
    sb.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (m_Data.NodesElmtID[static_cast<std::size_t>(i)] > 0
            && s[static_cast<std::size_t>(i)] > 0.0) {
            sb.push_back(s[static_cast<std::size_t>(i)]);
        }
    }
    if (sb.empty()) {
        for (int i = 0; i < n; ++i) if (s[static_cast<std::size_t>(i)] > 0.0)
            sb.push_back(s[static_cast<std::size_t>(i)]);
    }
    if (sb.empty()) {
        MessagePrinter::printErrorTxt("PDMesh::computeVariableHorizon: no positive node volume found");
        MessagePrinter::exitPeriX();
    }
    std::sort(sb.begin(), sb.end());
    const double sRef = sb[sb.size()/2];
    if (sRef <= 0.0) {
        MessagePrinter::printErrorTxt("PDMesh::computeVariableHorizon: non-positive reference spacing");
        MessagePrinter::exitPeriX();
    }
    const double invSRef = 1.0/sRef;

    // delta_i = HorizonRadius * (s_i/s_ref); dx_i = max(DX,DY,DZ) * (s_i/s_ref).
    // Both reduce to the global value when the mesh is uniform (s_i == s_ref).
    m_Data.NodeHorizon.assign(static_cast<std::size_t>(n), m_Data.HorizonRadius);
    m_Data.NodeSpacing.assign(static_cast<std::size_t>(n), dxChar);
    double dmin = m_Data.HorizonRadius, dmax = m_Data.HorizonRadius;
    for (int i = 0; i < n; ++i) {
        const double ratio = (s[static_cast<std::size_t>(i)] > 0.0)
                           ? s[static_cast<std::size_t>(i)]*invSRef : 1.0;
        const double di = m_Data.HorizonRadius * ratio;
        m_Data.NodeHorizon[static_cast<std::size_t>(i)] = di;
        m_Data.NodeSpacing[static_cast<std::size_t>(i)] = dxChar * ratio;
        dmin = std::min(dmin, di);
        dmax = std::max(dmax, di);
    }

    constexpr int bufSize = 160;
    char buff[bufSize];
    std::snprintf(buff, bufSize,
                  "  variable horizon ON: delta in [%10.3e, %10.3e], reference delta=%10.3e",
                  dmin, dmax, m_Data.HorizonRadius);
    MessagePrinter::printNormalTxt(buff);
}

void PDMesh::createNeighborNodes() {
    const int n = m_Data.NodesNum;
    m_Data.NodesNeighNodesID.assign(n, std::vector<int>{});

    if (n <= 0) {
        m_Data.MaxNeighbors=0;
        return;
    }
    if (m_Data.HorizonRadius <= 0.0) {
        MessagePrinter::printErrorTxt("pd horizon radius must be positive before creating neighbor nodes");
        MessagePrinter::exitPeriX();
    }

    const double *coords = m_Data.NodeCoords.data();
    const double r = m_Data.HorizonRadius;
    const double r2 = r * r;
    const double r2Tol = 64.0*std::numeric_limits<double>::epsilon()*r2;
    // Variable (per-node) horizon: each node i keeps its own delta_i. The cell
    // list must span the LARGEST horizon so no candidate pair within
    // max(delta_i,delta_j) is missed; the per-pair test below then keeps node i's
    // own horizon family. When VariableHorizon is off NodeHorizon is empty and
    // every line here reduces to the uniform single-horizon path byte-for-byte.
    const bool variableHorizon = !m_Data.NodeHorizon.empty();
    double deltaMax = r;
    if (variableHorizon) {
        deltaMax = 0.0;
        for (const double d : m_Data.NodeHorizon) deltaMax = std::max(deltaMax, d);
    }
    const double cellSize = variableHorizon ? deltaMax : r;
    // createPDMesh sets DZ>0 only for a 3d (hex8) pd mesh; DZ==0 in 2d.
    const bool is3D = (m_Data.DZ > 0.0);

    double minX=std::numeric_limits<double>::max();
    double minY=std::numeric_limits<double>::max();
    double minZ=std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        minX=std::min(minX, coords[3 * i + 0]);
        minY=std::min(minY, coords[3 * i + 1]);
        minZ=std::min(minZ, coords[3 * i + 2]);
    }
    if (!is3D) minZ=0.0;

    struct CellIndex {
        int ix;
        int iy;
        int iz;
    };
    auto getCellIndex = [&](const int nodeId) -> CellIndex {
        return {
            static_cast<int>(std::floor((coords[3 * nodeId + 0] - minX) / cellSize)),
            static_cast<int>(std::floor((coords[3 * nodeId + 1] - minY) / cellSize)),
            is3D ? static_cast<int>(std::floor((coords[3 * nodeId + 2] - minZ) / cellSize)) : 0
        };
    };
    // pack the (ix,iy,iz) cell index into a single key. A bias keeps the
    // single-cell-back stencil offsets (down to -1) non-negative; 21 bits per
    // axis covers any realistic cell grid (<= ~2M cells/axis).
    auto encodeCell = [](const int ix, const int iy, const int iz) -> std::uint64_t {
        constexpr long bias=1L<<20;
        const std::uint64_t ux=static_cast<std::uint64_t>(static_cast<long>(ix)+bias);
        const std::uint64_t uy=static_cast<std::uint64_t>(static_cast<long>(iy)+bias);
        const std::uint64_t uz=static_cast<std::uint64_t>(static_cast<long>(iz)+bias);
        return (ux<<42)|(uy<<21)|uz;
    };

    struct CellBucket {
        int ix;
        int iy;
        int iz;
        std::vector<int> nodes;
    };

    std::vector<CellBucket> cellBuckets;
    cellBuckets.reserve(static_cast<std::size_t>(n));
    std::unordered_map<std::uint64_t, int> cellKeyToBucketID;
    cellKeyToBucketID.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const CellIndex cell=getCellIndex(i);
        const std::uint64_t cellKey=encodeCell(cell.ix,cell.iy,cell.iz);
        const auto [it, inserted]=cellKeyToBucketID.emplace(cellKey, static_cast<int>(cellBuckets.size()));
        if (inserted) {
            cellBuckets.push_back(CellBucket{cell.ix, cell.iy, cell.iz, {}});
        }
        cellBuckets[static_cast<std::size_t>(it->second)].nodes.push_back(i);
    }

    // Forward half-stencil of the cell neighbourhood (the same cell is
    // handled separately). Only the lexicographically-positive half of the
    // 8- (2D) / 26- (3D) neighbourhood is visited so each cell PAIR is
    // processed exactly once: 4 offsets in 2D, 13 in 3D.
    static const int stencil2D[4][3] = {
        {1,-1,0},{1,0,0},{1,1,0},{0,1,0}
    };
    static const int stencil3D[13][3] = {
        {1,-1,-1},{1,-1,0},{1,-1,1},{1,0,-1},{1,0,0},{1,0,1},{1,1,-1},{1,1,0},{1,1,1},
        {0,1,-1},{0,1,0},{0,1,1},
        {0,0,1}
    };
    const int (*stencil)[3] = is3D ? stencil3D : stencil2D;
    const int nstencil       = is3D ? 13 : 4;

    // Crack (slit) bond cutting is delegated to initialCrackCutsBond(), the
    // single ghost-aware crossing rule shared with the force-only seeding in
    // the fracture kernels: a ghost endpoint is resolved to its mirror bulk
    // before the strict xy segment-crossing test (see PDMesh.h for why).
    const bool cutCracks = !m_Data.ForceOnlyCracks && m_Data.hasInitialCracks();

    auto tryAddPair = [&](const int i, const int j) {
        const double xi = coords[3 * i + 0];
        const double yi = coords[3 * i + 1];
        const double zi = coords[3 * i + 2];
        const double xj = coords[3 * j + 0];
        const double yj = coords[3 * j + 1];
        const double zj = coords[3 * j + 2];
        const double dxij = xi - xj;
        const double dyij = yi - yj;
        const double dzij = is3D ? (zi - zj) : 0.0;
        const double dist2 = dxij * dxij + dyij * dyij + dzij * dzij;
        if (!variableHorizon) {
            if (dist2 <= r2 + r2Tol) {
                // Geometric (kinematics-respecting) slit: a crack-crossing bond is
                // deleted from the family. In force-only crack mode the bond is
                // kept (the element zeroes its force instead), so the shape tensor
                // and deformation gradient span the discontinuity.
                if (cutCracks && initialCrackCutsBond(i+1, j+1)) {
                    return; // bond is cut by an initial crack: skip
                }
                m_Data.NodesNeighNodesID[i].push_back(j + 1);
                m_Data.NodesNeighNodesID[j].push_back(i + 1);
            }
            return;
        }
        // ---- variable-horizon symmetric family (per-node delta_i) ----
        // A bond is stored for both nodes when |xi_ij| <= max(delta_i,delta_j).
        // Each node can therefore evaluate its local horizon from symmetric
        // neighbour storage; strong-form PDDO operators ignore bonds outside
        // the evaluating node's own delta_i. Initial cracks remove both entries.
        const double di = m_Data.NodeHorizon[i];
        const double dj = m_Data.NodeHorizon[j];
        const double cut = std::max(di, dj); const double cut2 = cut*cut;
        if (dist2 > cut2 + 64.0*std::numeric_limits<double>::epsilon()*cut2) return;
        if (cutCracks && initialCrackCutsBond(i+1, j+1)) {
            return; // bond is cut by an initial crack: skip
        }
        m_Data.NodesNeighNodesID[i].push_back(j + 1);
        m_Data.NodesNeighNodesID[j].push_back(i + 1);
    };

    for (const auto &bucket : cellBuckets) {
        const auto &bucketNodes=bucket.nodes;
        for (std::size_t localI = 0; localI < bucketNodes.size(); ++localI) {
            for (std::size_t localJ = localI + 1; localJ < bucketNodes.size(); ++localJ) {
                tryAddPair(bucketNodes[localI], bucketNodes[localJ]);
            }
        }

        for (int offset = 0; offset < nstencil; ++offset) {
            const auto neighborIt=cellKeyToBucketID.find(encodeCell(bucket.ix + stencil[offset][0],
                                                                    bucket.iy + stencil[offset][1],
                                                                    bucket.iz + stencil[offset][2]));
            if (neighborIt == cellKeyToBucketID.end()) {
                continue;
            }

            const auto &neighborNodes=cellBuckets[static_cast<std::size_t>(neighborIt->second)].nodes;
            for (const int i : bucketNodes) {
                for (const int j : neighborNodes) {
                    tryAddPair(i, j);
                }
            }
        }
    }

    m_Data.MaxNeighbors=0;
    m_Data.MinNeighbors=100000;
    for (const auto &NodeIDs: m_Data.NodesNeighNodesID) {
        if (static_cast<int>(NodeIDs.size())>m_Data.MaxNeighbors) m_Data.MaxNeighbors=static_cast<int>(NodeIDs.size());
        if (static_cast<int>(NodeIDs.size())<m_Data.MinNeighbors) m_Data.MinNeighbors=static_cast<int>(NodeIDs.size());
    }

    // Coarse-mesh guard: a LOCAL PD model needs the horizon << the body size. If
    // the horizon (factor * point spacing) is comparable to the model itself,
    // every node sees much of the body -- the model becomes effectively non-local
    // and the system matrix near-dense (very slow / ill-posed). This happens when
    // an imported mesh is too coarse for the chosen HorizonRadiusFactor (e.g. a
    // sphere whose tet edge ~ radius/factor, so horizon > radius). Warn so the
    // user refines instead of waiting on a near-dense solve.
    if (m_Data.NodesNum>0) {
        double lo[3]={1e300,1e300,1e300}, hi[3]={-1e300,-1e300,-1e300};
        for (int i=0;i<m_Data.NodesNum;i++)
            for (int a=0;a<3;a++) {
                const double x=m_Data.NodeCoords[static_cast<std::size_t>(i)*3+static_cast<std::size_t>(a)];
                if (x<lo[a]) lo[a]=x;
                if (x>hi[a]) hi[a]=x;
            }
        double extent=0.0; for (int a=0;a<3;a++) extent=std::max(extent,hi[a]-lo[a]);
        double maxH=m_Data.HorizonRadius;
        for (const double d:m_Data.NodeHorizon) maxH=std::max(maxH,d);
        // A LOCAL model is actually near-dense only when a node SEES a large
        // fraction of the body. The bbox/horizon ratio alone misfires on a
        // symmetric SUB-domain (e.g. a 1/8-sphere octant exploiting symmetry):
        // its bounding box is a fraction of the true body, so maxH/extent is
        // inflated even though the per-node neighbour count is perfectly normal
        // (the matrix is NOT dense and the solve is fast). Require BOTH the
        // large horizon ratio AND a high neighbour fraction, so the warning
        // flags the genuinely under-resolved mesh (a node sees most of the body)
        // and stays quiet on a small but well-resolved wedge.
        const bool nearDense = m_Data.MaxNeighbors > m_Data.NodesNum/4;
        if (extent>0.0 && maxH>0.2*extent && nearDense) {
            char buf[360];
            snprintf(buf,sizeof(buf),
                "coarse-mesh warning: the PD horizon (%.3e) is %.0f%% of the model size (%.3e); a "
                "node sees up to %d of %d nodes. The mesh is too coarse for the chosen "
                "HorizonRadiusFactor -- the matrix is near-dense and the solve will be very slow. "
                "Refine so the point spacing is << model_size/factor (or lower the factor).",
                maxH,100.0*maxH/extent,extent,m_Data.MaxNeighbors,m_Data.NodesNum);
            MessagePrinter::printWarningTxt(buf);
        }
    }
}
