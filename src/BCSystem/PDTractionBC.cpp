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
//+++ Date    : 2026.07.28
//+++ Function: strong small-strain PDDO traction implementation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "BCSystem/PDTractionBC.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

#include "PDMesh/PDMesh.h"
#include "PDOperators/PDOperators.h"
#include "Utils/MessagePrinter.h"

namespace {
[[nodiscard]] int normalBin(const std::array<double,3> &normal) {
    const double ax=std::fabs(normal[0]);
    const double ay=std::fabs(normal[1]);
    const double az=std::fabs(normal[2]);
    if (ax>=ay && ax>=az && ax>0.0) return normal[0]>=0.0 ? 1 : -1;
    if (ay>=ax && ay>=az && ay>0.0) return normal[1]>=0.0 ? 2 : -2;
    if (az>0.0) return normal[2]>=0.0 ? 3 : -3;
    return 0;
}

[[nodiscard]] std::vector<int> resolveDisplacementComponents(
    const std::vector<int> &requested,const int dim,const int dofsPerNode) {
    if (requested.empty()) {
        if (dofsPerNode!=dim) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: a coupled field must explicitly list its "
                "displacement DoF slots");
            MessagePrinter::exitPeriX();
        }
        std::vector<int> components(static_cast<std::size_t>(dim));
        for (int a=0;a<dim;++a) {
            components[static_cast<std::size_t>(a)]=a+1;
        }
        return components;
    }
    if (static_cast<int>(requested.size())!=dim) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: displacement component count must equal the "
            "spatial dimension");
        MessagePrinter::exitPeriX();
    }
    return requested;
}

void validateComponents(const std::vector<int> &components,
                        const int dofsPerNode) {
    std::unordered_set<int> unique;
    for (const int component : components) {
        if (component<1 || component>dofsPerNode) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: displacement component "
                +std::to_string(component)+" is outside [1,"
                +std::to_string(dofsPerNode)+"]");
            MessagePrinter::exitPeriX();
        }
        if (!unique.insert(component).second) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: duplicate displacement component "
                +std::to_string(component));
            MessagePrinter::exitPeriX();
        }
    }
}

[[nodiscard]] std::vector<int> resolveBoundaryGhosts(
    const PDMesh &Mesh,const std::vector<int> &nodeIDs) {
    const auto &data=Mesh.getDataConstRef();
    const auto &mirror=data.GhostMirrorBulkID;
    const int nodes=Mesh.getNodesNum();

    std::vector<int> ghosts;
    std::vector<int> bulkIDs;
    for (const int id : nodeIDs) {
        if (id<1 || id>nodes) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: node "+std::to_string(id)
                +" is outside the PD mesh");
            MessagePrinter::exitPeriX();
        }
        if (mirror[static_cast<std::size_t>(id-1)]>=1) {
            ghosts.push_back(id);
        }
        else {
            bulkIDs.push_back(id);
        }
    }

    std::vector<std::pair<int,int>> mapped;
    if (!bulkIDs.empty()) {
        std::vector<char> selectedBulk(static_cast<std::size_t>(nodes+1),0);
        for (const int bulk : bulkIDs) {
            selectedBulk[static_cast<std::size_t>(bulk)]=1;
        }
        for (int ghost=1;ghost<=nodes;++ghost) {
            const int bulk=mirror[static_cast<std::size_t>(ghost-1)];
            if (bulk>=1 && selectedBulk[static_cast<std::size_t>(bulk)]) {
                mapped.emplace_back(
                    ghost,normalBin(
                        data.NodeNormal[static_cast<std::size_t>(ghost-1)]));
            }
        }
    }

    if (!mapped.empty()) {
        int counts[7]={0,0,0,0,0,0,0};
        for (const auto &[ghost,bin] : mapped) {
            (void)ghost;
            if (bin>=-3 && bin<=3) ++counts[bin+3];
        }
        int bestBin=0;
        int bestCount=0;
        int secondCount=0;
        for (int bin=-3;bin<=3;++bin) {
            if (bin==0) continue;
            const int count=counts[bin+3];
            if (count>bestCount) {
                secondCount=bestCount;
                bestCount=count;
                bestBin=bin;
            }
            else if (count>secondCount) {
                secondCount=count;
            }
        }
        if (bestCount==0 || bestCount==secondCount) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: a bulk boundary group maps to multiple equally "
                "likely outward faces; bind the condition to the corresponding "
                "ghost physical group to make the surface unambiguous");
            MessagePrinter::exitPeriX();
        }
        for (const auto &[ghost,bin] : mapped) {
            if (bin==bestBin) ghosts.push_back(ghost);
        }
    }

    std::sort(ghosts.begin(),ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(),ghosts.end()),ghosts.end());
    if (ghosts.empty() && !nodeIDs.empty()) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: the physical group contains no boundary ghost and "
            "no bulk node with a mirror ghost");
        MessagePrinter::exitPeriX();
    }
    return ghosts;
}
}

PDTractionBC::PDTractionBC(const double E,const double nu,
                           const StressState state,
                           const std::array<double,3> &traction)
    :m_Traction(traction) {
    setElastic(E,nu,state);
}

void PDTractionBC::setElastic(const double E,const double nu,
                              const StressState state) {
    if (!(E>0.0) || !std::isfinite(E)) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: Young's modulus must be finite and positive");
        MessagePrinter::exitPeriX();
    }
    if (!(nu>-1.0 && nu<0.5) || !std::isfinite(nu)) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: Poisson's ratio must satisfy -1 < nu < 0.5");
        MessagePrinter::exitPeriX();
    }
    m_E=E;
    m_Nu=nu;
    m_State=state;
    m_ElasticConfigured=true;
}

void PDTractionBC::presetControlledRows(const PDMesh &Mesh,
                                        const std::vector<int> &NodeIDs,
                                        const int &DofsPerNode,
                                        std::vector<char> &mask) const {
    // The strong traction rows are written for the displacement slots only.
    // An explicit slot list already fixes them; an empty list is only legal
    // when the field is pure mechanics, i.e. every slot is a displacement.
    std::vector<int> displacement=m_DisplacementComponents;
    if (displacement.empty()) {
        displacement.resize(static_cast<std::size_t>(DofsPerNode));
        for (int a=0;a<DofsPerNode;++a) {
            displacement[static_cast<std::size_t>(a)]=a+1;
        }
    }
    validateComponents(displacement,DofsPerNode);

    const auto &mirror=Mesh.getDataConstRef().GhostMirrorBulkID;
    if (static_cast<int>(mirror.size())<Mesh.getNodesNum()) return;

    for (const int ghost : resolveBoundaryGhosts(Mesh,NodeIDs)) {
        const std::size_t base=static_cast<std::size_t>(ghost-1)
                             *static_cast<std::size_t>(DofsPerNode);
        for (const int component : displacement) {
            mask[base+static_cast<std::size_t>(component-1)]=1;
        }
    }
}

void PDTractionBC::applyWithOperators(
    const PDMesh &Mesh,PDOperators &Operators,
    const std::vector<int> &NodeIDs,const int &DofsPerNode,
    const VectorXd &U,SparseMatrix &K,VectorXd &RHS,
    const double &time) const {
    (void)time;
    if (!m_ElasticConfigured) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: elastic constants were not configured");
        MessagePrinter::exitPeriX();
    }
    const int dim=Operators.getDim();
    if (dim!=2 && dim!=3) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: PDDO dimension must be two or three");
        MessagePrinter::exitPeriX();
    }
    const auto displacement=resolveDisplacementComponents(
        m_DisplacementComponents,dim,DofsPerNode);
    validateComponents(displacement,DofsPerNode);

    if (m_HasChemicalExpansion) {
        if (m_CComponent<1 || m_CComponent>DofsPerNode) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: concentration component is outside the node "
                "DoF range");
            MessagePrinter::exitPeriX();
        }
        if (std::find(displacement.begin(),displacement.end(),m_CComponent)
            !=displacement.end()) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: concentration and displacement components "
                "must be distinct");
            MessagePrinter::exitPeriX();
        }
        if (!std::isfinite(m_Omega) || !std::isfinite(m_CRef)) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: chemical-expansion parameters must be finite");
            MessagePrinter::exitPeriX();
        }
    }
    for (const double value : m_Traction) {
        if (!std::isfinite(value)) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: traction components must be finite");
            MessagePrinter::exitPeriX();
        }
    }

    const auto &data=Mesh.getDataConstRef();
    if (static_cast<int>(data.GhostMirrorBulkID.size())<Mesh.getNodesNum()
        || static_cast<int>(data.NodeNormal.size())<Mesh.getNodesNum()) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: strong traction needs boundary ghost-to-bulk "
            "mapping and outward node normals");
        MessagePrinter::exitPeriX();
    }

    double C11=0.0;
    double C12=0.0;
    double C66=m_E/(2.0*(1.0+m_Nu));
    double chemicalStress=0.0;
    if (dim==2 && m_State==StressState::PlaneStress) {
        const double denom=1.0-m_Nu*m_Nu;
        C11=m_E/denom;
        C12=m_E*m_Nu/denom;
        chemicalStress=m_E*m_Omega/(3.0*(1.0-m_Nu));
    }
    else {
        const double denom=(1.0+m_Nu)*(1.0-2.0*m_Nu);
        C11=m_E*(1.0-m_Nu)/denom;
        C12=m_E*m_Nu/denom;
        chemicalStress=m_E*m_Omega/(3.0*(1.0-2.0*m_Nu));
    }
    if (!m_HasChemicalExpansion) chemicalStress=0.0;

    const int gxIndex=Operators.getOperatorIndex("d/dx");
    const int gyIndex=Operators.getOperatorIndex("d/dy");
    const int gzIndex=dim==3 ? Operators.getOperatorIndex("d/dz") : -1;
    if (gxIndex<1 || gyIndex<1 || (dim==3 && gzIndex<1)) {
        MessagePrinter::printErrorTxt(
            "PDTractionBC: first-order PDDO gradient operators are unavailable");
        MessagePrinter::exitPeriX();
    }

    const auto boundaryGhosts=resolveBoundaryGhosts(Mesh,NodeIDs);
    const auto &mirror=data.GhostMirrorBulkID;
    const bool savedRespect=Operators.getRespectInitialCracks();
    Operators.setRespectInitialCracks(true);

    for (const int ghost : boundaryGhosts) {
        const int bulk=mirror[static_cast<std::size_t>(ghost-1)];
        const auto &normal=data.NodeNormal[static_cast<std::size_t>(ghost-1)];
        const double nx=normal[0];
        const double ny=normal[1];
        const double nz=normal[2];
        if (bulk<1 || (nx==0.0 && ny==0.0 && nz==0.0)) {
            MessagePrinter::printErrorTxt(
                "PDTractionBC: boundary ghost "+std::to_string(ghost)
                +" has no valid mirror bulk or outward normal");
            MessagePrinter::exitPeriX();
        }

        int ghostDof[3]={0,0,0};
        for (int a=0;a<dim;++a) {
            ghostDof[a]=(ghost-1)*DofsPerNode
                       +displacement[static_cast<std::size_t>(a)];
        }

        const std::size_t ghostIndex=static_cast<std::size_t>(ghost-1);
        const bool deeper=ghostIndex<data.GhostLayerIndex.size()
            && data.GhostLayerIndex[ghostIndex]>1;
        if (deeper) {
            for (int a=0;a<dim;++a) {
                const int bulkDof=(bulk-1)*DofsPerNode
                    +displacement[static_cast<std::size_t>(a)];
                const double scale=bcRowScale(K,ghostDof[a]);
                K.zeroRowAndSetDiagonal(ghostDof[a],scale);
                K.insertValue(ghostDof[a],bulkDof,-scale);
                RHS.insertValue(
                    ghostDof[a],scale*(U(bulkDof)-U(ghostDof[a])));
            }
            continue;
        }

        for (int a=0;a<dim;++a) {
            K.zeroRowAndSetDiagonal(ghostDof[a],0.0);
        }
        double sigmaNormal[3]={0.0,0.0,0.0};

        Operators.calcAMatrix(ghost,data);
        for (const int neighbor : Mesh.getIthNodeNeighborNodeIDs(ghost)) {
            Operators.calcOperators(ghost,neighbor,data);
            const double Gx=Operators.getOperatorByIndex(gxIndex);
            const double Gy=Operators.getOperatorByIndex(gyIndex);
            const double Gz=dim==3
                ? Operators.getOperatorByIndex(gzIndex)
                : 0.0;

            double coefficient[3][3]={{0.0,0.0,0.0},
                                      {0.0,0.0,0.0},
                                      {0.0,0.0,0.0}};
            if (dim==2) {
                coefficient[0][0]=C11*Gx*nx+C66*Gy*ny;
                coefficient[0][1]=C12*Gy*nx+C66*Gx*ny;
                coefficient[1][0]=C12*Gx*ny+C66*Gy*nx;
                coefficient[1][1]=C11*Gy*ny+C66*Gx*nx;
            }
            else {
                coefficient[0][0]=C11*Gx*nx+C66*Gy*ny+C66*Gz*nz;
                coefficient[0][1]=C12*Gy*nx+C66*Gx*ny;
                coefficient[0][2]=C12*Gz*nx+C66*Gx*nz;
                coefficient[1][0]=C66*Gy*nx+C12*Gx*ny;
                coefficient[1][1]=C66*Gx*nx+C11*Gy*ny+C66*Gz*nz;
                coefficient[1][2]=C12*Gz*ny+C66*Gy*nz;
                coefficient[2][0]=C66*Gz*nx+C12*Gx*nz;
                coefficient[2][1]=C66*Gz*ny+C12*Gy*nz;
                coefficient[2][2]=C66*Gx*nx+C66*Gy*ny+C11*Gz*nz;
            }

            int neighborDof[3]={0,0,0};
            for (int b=0;b<dim;++b) {
                neighborDof[b]=(neighbor-1)*DofsPerNode
                    +displacement[static_cast<std::size_t>(b)];
            }
            for (int a=0;a<dim;++a) {
                for (int b=0;b<dim;++b) {
                    const double value=coefficient[a][b];
                    if (value==0.0) continue;
                    K.addValue(ghostDof[a],neighborDof[b],value);
                    K.addValue(ghostDof[a],ghostDof[b],-value);
                    sigmaNormal[a]+=value
                        *(U(neighborDof[b])-U(ghostDof[b]));
                }
            }
        }

        const int concentrationDof=(ghost-1)*DofsPerNode+m_CComponent;
        const double concentrationDifference=m_HasChemicalExpansion
            ? U(concentrationDof)-m_CRef
            : 0.0;
        const double n[3]={nx,ny,nz};
        for (int a=0;a<dim;++a) {
            if (m_HasChemicalExpansion) {
                const double derivative=-chemicalStress*n[a];
                if (derivative!=0.0) {
                    K.addValue(ghostDof[a],concentrationDof,derivative);
                }
            }
            RHS.insertValue(
                ghostDof[a],
                m_Traction[static_cast<std::size_t>(a)]
                -sigmaNormal[a]
                +chemicalStress*concentrationDifference*n[a]);
        }
    }

    Operators.setRespectInitialCracks(savedRespect);
}
