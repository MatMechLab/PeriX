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
//+++ Date    : 2026.05.08
//+++ Function: ElmtSystem implementation. Registers element
//+++           kernels and dispatches per-bond / per-node
//+++           residual / jacobian computation.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ElmtSystem/ElmtSystem.h"

#include <algorithm>
#include <utility>

#include "ElmtSystem/CahnHilliardElement.h"
#include "ElmtSystem/DiffusionElement.h"
#include "ElmtSystem/PDDODynamicFracElement.h"
#include "ElmtSystem/PoissonElement.h"
#include "Utils/MessagePrinter.h"

void ElmtSystem::addElement(const std::string &name,std::unique_ptr<ElementBase> elmt) {
    if (!elmt) {
        MessagePrinter::printErrorTxt("ElmtSystem::addElement: null element passed for name='"+name+"'");
        MessagePrinter::exitPeriX();
    }
    if (hasElement(name)) {
        MessagePrinter::printErrorTxt("ElmtSystem::addElement: duplicate element name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    m_Elements.push_back(ElementEntry{name,std::move(elmt)});
}

bool ElmtSystem::hasElement(const std::string &name) const {
    return std::any_of(m_Elements.begin(),m_Elements.end(),
                       [&name](const ElementEntry &e){ return e.Name==name; });
}

const ElementBase& ElmtSystem::getElement(const std::string &name) const {
    const auto it=std::find_if(m_Elements.begin(),m_Elements.end(),
                               [&name](const ElementEntry &e){ return e.Name==name; });
    if (it==m_Elements.end()) {
        MessagePrinter::printErrorTxt("ElmtSystem::getElement: no element registered under name '"+name+"'");
        MessagePrinter::exitPeriX();
    }
    return *it->Element;
}

const ElementBase& ElmtSystem::getElementByIndex(const int &i) const {
    if (i<0 || i>=static_cast<int>(m_Elements.size())) {
        MessagePrinter::printErrorTxt("ElmtSystem::getElementByIndex: index "+std::to_string(i)+" out of range");
        MessagePrinter::exitPeriX();
    }
    return *m_Elements[static_cast<std::size_t>(i)].Element;
}

const std::string& ElmtSystem::getElementNameByIndex(const int &i) const {
    if (i<0 || i>=static_cast<int>(m_Elements.size())) {
        MessagePrinter::printErrorTxt("ElmtSystem::getElementNameByIndex: index "+std::to_string(i)+" out of range");
        MessagePrinter::exitPeriX();
    }
    return m_Elements[static_cast<std::size_t>(i)].Name;
}

int ElmtSystem::getMaxDofsPerNode() const {
    int maxDofs=0;
    for (const auto &entry : m_Elements) {
        maxDofs=std::max(maxDofs,entry.Element->getDofsPerNode());
    }
    return maxDofs;
}

void ElmtSystem::computeBondResidualAndJacobian(const PDOperators &t_PDOperators,
                                                const LocalElmtInfo &Info,
                                                const int &NodeI,
                                                const int &NodeJ,
                                                const VectorXd &U_I,
                                                const VectorXd &U_J,
                                                const VectorXd &Uold_I,
                                                const VectorXd &Uold_J,
                                                const double  &Volume,
                                                VectorXd &LocalR,
                                                MatrixXd &LocalK_II,
                                                MatrixXd &LocalK_IJ) const {
    if (m_Elements.empty()) {
        MessagePrinter::printErrorTxt("ElmtSystem::computeBondResidualAndJacobian: no element registered");
        MessagePrinter::exitPeriX();
    }

    LocalR.setToZeros();
    LocalK_II.setToZeros();
    LocalK_IJ.setToZeros();

    if (m_Elements.size()==1) {
        m_Elements.front().Element->computeBondResidualAndJacobian(
            t_PDOperators,Info,NodeI,NodeJ,U_I,U_J,Uold_I,Uold_J,Volume,
            LocalR,LocalK_II,LocalK_IJ);
        return;
    }

    const int n=LocalR.getSize();
    // Per-thread scratch so this const method is reentrant: under the
    // OpenMP-parallel assembler several threads call it concurrently, and a
    // shared member buffer would be a data race.
    static thread_local VectorXd tmpR;
    static thread_local MatrixXd tmpK_II, tmpK_IJ;
    if (tmpR.getSize()!=n) {
        tmpR.resize(n);
        tmpK_II.resize(n,n);
        tmpK_IJ.resize(n,n);
    }
    for (const auto &entry : m_Elements) {
        tmpR.setToZeros();
        tmpK_II.setToZeros();
        tmpK_IJ.setToZeros();
        entry.Element->computeBondResidualAndJacobian(
            t_PDOperators,Info,NodeI,NodeJ,U_I,U_J,Uold_I,Uold_J,Volume,
            tmpR,tmpK_II,tmpK_IJ);
        for (int a=1;a<=n;a++) {
            LocalR(a)+=tmpR(a);
            for (int b=1;b<=n;b++) {
                LocalK_II(a,b)+=tmpK_II(a,b);
                LocalK_IJ(a,b)+=tmpK_IJ(a,b);
            }
        }
    }
}

void ElmtSystem::computeNodalResidualAndJacobian(const LocalElmtInfo &Info,
                                                 const int &NodeI,
                                                 const VectorXd &U_I,
                                                 const VectorXd &Uold_I,
                                                 const double  &Volume,
                                                 VectorXd &LocalR_node,
                                                 MatrixXd &LocalK_node) const {
    if (m_Elements.empty()) {
        MessagePrinter::printErrorTxt("ElmtSystem::computeNodalResidualAndJacobian: no element registered");
        MessagePrinter::exitPeriX();
    }

    LocalR_node.setToZeros();
    LocalK_node.setToZeros();

    if (m_Elements.size()==1) {
        m_Elements.front().Element->computeNodalResidualAndJacobian(
            Info,NodeI,U_I,Uold_I,Volume,LocalR_node,LocalK_node);
        return;
    }

    const int n=LocalR_node.getSize();
    // Per-thread scratch (reentrant; see computeBondResidualAndJacobian).
    static thread_local VectorXd tmpR;
    static thread_local MatrixXd tmpK_II;
    if (tmpR.getSize()!=n) {
        tmpR.resize(n);
        tmpK_II.resize(n,n);
    }
    for (const auto &entry : m_Elements) {
        tmpR.setToZeros();
        tmpK_II.setToZeros();
        entry.Element->computeNodalResidualAndJacobian(Info,NodeI,U_I,Uold_I,Volume,tmpR,tmpK_II);
        for (int a=1;a<=n;a++) {
            LocalR_node(a)+=tmpR(a);
            for (int b=1;b<=n;b++) {
                LocalK_node(a,b)+=tmpK_II(a,b);
            }
        }
    }
}

bool ElmtSystem::isExplicit() const {
    if (m_Elements.empty()) return false;
    return std::all_of(m_Elements.begin(),m_Elements.end(),
                       [](const ElementEntry &e){ return e.Element->isExplicit(); });
}

int ElmtSystem::getTimeOrder() const {
    if (m_Elements.empty()) return 1;
    const int order=m_Elements.front().Element->getTimeOrder();
    for (const auto &entry : m_Elements) {
        if (entry.Element->getTimeOrder()!=order) {
            MessagePrinter::printErrorTxt("ElmtSystem::getTimeOrder: all registered kernels must share the "
                                          "same time-derivative order (mixing 1st-order and 2nd-order "
                                          "explicit kernels in one run is not supported)");
            MessagePrinter::exitPeriX();
        }
    }
    return order;
}

std::vector<int> ElmtSystem::getDofTimeOrders() const {
    const int ndof=getMaxDofsPerNode();
    if (m_Elements.empty()) return std::vector<int>(static_cast<std::size_t>(ndof),1);
    // velocity-Verlet (mixed-order) is a single-kernel coupled model; take the
    // front kernel's mask and pad to the global DoF count if needed.
    std::vector<int> orders=m_Elements.front().Element->getDofTimeOrders();
    orders.resize(static_cast<std::size_t>(ndof),
                  orders.empty()?1:orders.back());
    return orders;
}

const ElementBase* ElmtSystem::findProjectionProvider(const std::string &name) const {
    // first registered kernel that advertises the projection; the save paths
    // route each output field here so a multi-element run no longer silently
    // writes zeros for a field owned by element[1..].
    for (const auto &entry : m_Elements)
        for (const auto &p : entry.Element->getAvailableProjections())
            if (p.Name==name) return entry.Element.get();
    return nullptr;
}

double ElmtSystem::estimateStableDt(const PDMesh &Mesh) const {
    // most restrictive positive per-kernel estimate; <=0 = nobody reported one
    double est=-1.0;
    for (const auto &entry : m_Elements) {
        const double d=entry.Element->estimateStableDt(Mesh);
        if (d>0.0 && (est<=0.0 || d<est)) est=d;
    }
    return est;
}

bool ElmtSystem::hasMixedTimeOrders() const {
    if (m_Elements.empty()) return false;
    const auto orders=getDofTimeOrders();
    bool has1=false,has2=false;
    for (const int o : orders) { if (o<=1) has1=true; else has2=true; }
    return has1 && has2;
}

bool ElmtSystem::isQuasiStatic() const {
    if (m_Elements.empty()) return false;
    return std::all_of(m_Elements.begin(),m_Elements.end(),
                       [](const ElementEntry &e){ return e.Element->isQuasiStatic(); });
}

void ElmtSystem::computeBondResidual(const PDOperators &t_PDOperators,
                                     const LocalElmtInfo &Info,
                                     const int &NodeI,
                                     const int &NodeJ,
                                     const VectorXd &U_I,
                                     const VectorXd &U_J,
                                     const VectorXd &Uold_I,
                                     const VectorXd &Uold_J,
                                     const double  &Volume,
                                     VectorXd &LocalR) const {
    if (m_Elements.empty()) {
        MessagePrinter::printErrorTxt("ElmtSystem::computeBondResidual: no element registered");
        MessagePrinter::exitPeriX();
    }

    LocalR.setToZeros();

    if (m_Elements.size()==1) {
        m_Elements.front().Element->computeBondResidual(
            t_PDOperators,Info,NodeI,NodeJ,U_I,U_J,Uold_I,Uold_J,Volume,LocalR);
        return;
    }

    const int n=LocalR.getSize();
    static thread_local VectorXd tmpR;
    if (tmpR.getSize()!=n) tmpR.resize(n);
    for (const auto &entry : m_Elements) {
        tmpR.setToZeros();
        entry.Element->computeBondResidual(
            t_PDOperators,Info,NodeI,NodeJ,U_I,U_J,Uold_I,Uold_J,Volume,tmpR);
        for (int a=1;a<=n;a++) LocalR(a)+=tmpR(a);
    }
}

void ElmtSystem::computeNodalResidual(const LocalElmtInfo &Info,
                                      const int &NodeI,
                                      const VectorXd &U_I,
                                      const VectorXd &Uold_I,
                                      const double  &Volume,
                                      VectorXd &LocalR_node) const {
    if (m_Elements.empty()) {
        MessagePrinter::printErrorTxt("ElmtSystem::computeNodalResidual: no element registered");
        MessagePrinter::exitPeriX();
    }

    LocalR_node.setToZeros();

    if (m_Elements.size()==1) {
        m_Elements.front().Element->computeNodalResidual(Info,NodeI,U_I,Uold_I,Volume,LocalR_node);
        return;
    }

    const int n=LocalR_node.getSize();
    static thread_local VectorXd tmpR;
    if (tmpR.getSize()!=n) tmpR.resize(n);
    for (const auto &entry : m_Elements) {
        tmpR.setToZeros();
        entry.Element->computeNodalResidual(Info,NodeI,U_I,Uold_I,Volume,tmpR);
        for (int a=1;a<=n;a++) LocalR_node(a)+=tmpR(a);
    }
}

void ElmtSystem::preprocessIteration(const PDMesh &Mesh,
                                     PDOperators &ops,
                                     const LocalElmtInfo &Info,
                                     const int &DofsPerNode,
                                     const VectorXd &U,
                                     const VectorXd &Uold) const {
    for (const auto &entry : m_Elements) {
        entry.Element->preprocessIteration(Mesh,ops,Info,DofsPerNode,U,Uold);
    }
}

void ElmtSystem::printElmtSystemInfo() const {
    MessagePrinter::printStars();
    MessagePrinter::printNormalTxt("Element system info");
    MessagePrinter::printNormalTxt("  registered elements="+std::to_string(getElementsNum())
                                   +", max dofs/node="+std::to_string(getMaxDofsPerNode()));
    for (const auto &entry : m_Elements) {
        std::string line="  - name='"+entry.Name
                         +"', type='"+entry.Element->getElementType()
                         +"', dofs/node="+std::to_string(entry.Element->getDofsPerNode());
        if (auto *poisson=dynamic_cast<const PoissonElement*>(entry.Element.get())) {
            line+=", sigma="+std::to_string(poisson->getSigma())
                  +", f="+std::to_string(poisson->getF());
        }
        else if (auto *diffusion=dynamic_cast<const DiffusionElement*>(entry.Element.get())) {
            line+=", D="+std::to_string(diffusion->getD())
                  +", f="+std::to_string(diffusion->getF());
        }
        else if (auto *pdf=dynamic_cast<const PDDODynamicFracElement*>(entry.Element.get())) {
            line+=", state="+pdf->getStateName()
                  +", E="+std::to_string(pdf->getE())
                  +", nu="+std::to_string(pdf->getNu())
                  +", rho="+std::to_string(pdf->getRho())
                  +", G0="+std::to_string(pdf->getG0())
                  +", damage="+(pdf->getDamageOn()?"on":"off")
                  +", tension_only="+(pdf->getTensionOnly()?"on":"off")
                  +", scheme=implicit-backward-euler(pardiso)";
        }
        else if (auto *ch=dynamic_cast<const CahnHilliardElement*>(entry.Element.get())) {
            line+=", chi="+std::to_string(ch->getChi())
                  +", kappa="+std::to_string(ch->getKappa())
                  +", M="+std::to_string(ch->getMobility());
        }
        MessagePrinter::printNormalTxt(line);
    }
    MessagePrinter::printStars();
}
