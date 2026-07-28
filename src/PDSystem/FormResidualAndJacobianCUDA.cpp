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
//+++ Date    : 2026.05.30
//+++ Function: host driver for the CUDA GPU residual/Jacobian
//+++           assembler. Detects whether the (single) registered
//+++           element has a device port (Poisson / Diffusion /
//+++           Mechanics so far), marshals the mesh / PDDO tables / CSR
//+++           pointers / element parameters into the POD bridge, and
//+++           launches the device assembler. Any unsupported element or
//+++           device error transparently falls back to the serial CPU
//+++           assembler, so every input stays correct.
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDSystem/PDSystem.h"

#include <algorithm>
#include <vector>

#include "PDSystem/CudaAssemble.h"
#include "ElmtSystem/PoissonElement.h"
#include "ElmtSystem/DiffusionElement.h"
#include "ElmtSystem/CahnHilliardElement.h"
#include "ElmtSystem/FracStressCahnHilliardElement.h"
#include "ElmtSystem/PDDODynamicFracElement.h"

void PDSystem::formResidualAndJacobianCUDA(const PDMesh &Mesh,
                                           PDOperators &t_PDOperators,
                                           const ElmtSystem &t_ElmtSystem,
                                           const LocalElmtInfo &Info,
                                           const VectorXd &U,
                                           const VectorXd &Uold,
                                           SparseMatrix &K,
                                           VectorXd &RHS) {
    const int ndof=t_ElmtSystem.getMaxDofsPerNode();
    const int dim=t_PDOperators.getDim();

    // ---- CONSERVATIVE Cahn-Hilliard (2-DoF): a bespoke row-centred device kernel
    //      (NOT the generic single-pass PDDO assembler below). The flux-conservative
    //      CPU element caches the degenerate mobility M(c)=M0 c(1-c), the lambda-
    //      corrected per-bond Laplacian, a reverse-bond map, node volumes and ghost
    //      flags in preprocessIteration(); the GPU port reuses those exact host
    //      arrays so the assembled system is bit-parity with the CPU. Detected here;
    //      any device failure falls back transparently to the CPU assembler. ----
    if (t_ElmtSystem.getElementsNum()==1) {
        if (auto *ch=dynamic_cast<const CahnHilliardElement*>(&t_ElmtSystem.getElementByIndex(0))) {
            if (Info.Dt<=0.0) {   // transient-only; a static run takes the CPU error path
                formResidualAndJacobian(Mesh,t_PDOperators,t_ElmtSystem,Info,U,Uold,K,RHS);
                return;
            }
            // build the one-off conservative-flux geometry + refresh the frozen mobility
            LocalElmtInfo ci=Info; ci.UseParallel=true;
            t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,ci,ndof,U,Uold);
            perix_cuda::CahnHilliardParams cp;
            cp.kappa=ch->getKappa(); cp.chi=ch->getChi(); cp.dt=Info.Dt;
            perix_cuda::CahnHilliardMesh cm;
            cm.N=Mesh.getNodesNum();
            cm.nbtot=static_cast<int>(ch->getNbrIds().size());
            cm.off=ch->getNbrOffsets().data();
            cm.nbr=ch->getNbrIds().data();
            cm.rev=ch->getRevBond().data();
            cm.lapbond=ch->getLapBond().data();
            cm.vol=ch->getNodeVolumes().data();
            cm.isghost=ch->getGhostFlags().empty()?nullptr:ch->getGhostFlags().data();
            cm.mnode=ch->getNodeMobility().data();
            if (perix_cuda::assembleCahnHilliard(cm,cp,K.getCSRRowsIndexPtr(),K.getCSRColsIndexPtr(),
                                                 K.getNNZNum(),U.getDataPtr(),Uold.getDataPtr(),
                                                 K.getCSRValuesPtr(),RHS.getDataPtr()))
                return;
            // device path failed -> correct CPU assembler
            formResidualAndJacobian(Mesh,t_PDOperators,t_ElmtSystem,Info,U,Uold,K,RHS);
            return;
        }
    }

    // Resolve the per-element parameters for the device path. Only a single
    // registered element with a device port is handled; anything else falls
    // back to the (correct) serial CPU assembler below. The kernels are
    // dimension-aware (2D and 3D) and variable-horizon-aware.
    perix_cuda::ElmtParams pp;
    pp.dim=dim;
    bool supported=false;
    const double *uold_ptr=nullptr;
    // Small-strain fracture stress-CH uses the generic PDDO moment-matrix setup
    // below and the conservative-flux device kernel.
    const FracStressCahnHilliardElement *frch=nullptr;
    // "cached PDDO callback" elements (moment matrix + Picard-frozen per-node cache)
    // route to the shared assembleCached path after am/ss are built.
    const PDDODynamicFracElement *pdf=nullptr;
    if (t_ElmtSystem.getElementsNum()==1) {
        const ElementBase &e0=t_ElmtSystem.getElementByIndex(0);
        if (auto *po=dynamic_cast<const PoissonElement*>(&e0)) {
            pp.type=perix_cuda::ELMT_POISSON; pp.ndof=1;
            pp.sigma=po->getSigma(); pp.f=po->getF();
            supported=true;
        }
        else if (auto *di=dynamic_cast<const DiffusionElement*>(&e0)) {
            pp.type=perix_cuda::ELMT_DIFFUSION; pp.ndof=1;
            pp.D=di->getD();
            pp.f=di->getF();
            pp.dt=Info.Dt; uold_ptr=Uold.getDataPtr();
            supported=(Info.Dt>0.0);   // transient-only; static run -> CPU error path
        }
        else if (auto *fr=dynamic_cast<const FracStressCahnHilliardElement*>(&e0)) {
            // small-strain fracture + stress-CH: stress_cahnhilliard physics gated by
            // frozen per-bond health, optional inertia. ndof=dim+2, transient.
            pp.ndof=dim+2;
            pp.dt=Info.Dt; uold_ptr=Uold.getDataPtr();
            supported=(Info.Dt>0.0);   // transient-only; static run -> CPU error path
            if (supported) frch=fr;
        }
        else if (auto *pd=dynamic_cast<const PDDODynamicFracElement*>(&e0)) {
            // implicit dynamic fracture on the PDDO operators: health-gated
            // Navier-Cauchy + backward-Euler inertia. ndof=dim, transient.
            pp.ndof=dim;
            pp.dt=Info.Dt; uold_ptr=Uold.getDataPtr();
            supported=(Info.Dt>0.0);   // transient-only (dynamics)
            if (supported) pdf=pd;
        }
        // NOTE: CahnHilliardElement is intercepted ABOVE by its bespoke
        // conservative-flux device path and never reaches this generic block.
        // CahnHilliardElement is intercepted above by its conservative-flux
        // device path and never reaches this generic block.
    }
    // PDMesh.VolumeCorrection=false is expressed to the device by a ZERO
    // characteristic spacing: devVolCorrection(xn,delta,dx_char) returns 1.0 when
    // dx_char<=0, exactly the CPU's `VolumeCorrection ? vc : 1.0` -- see the
    // am.dx_char / am.nodeSpacing marshalling below (e.g. the Kalthoff-Winkler
    // reference drivers run with the rim factor off).
    const bool noVc = !Mesh.getDataConstRef().VolumeCorrection;
    if (!supported || pp.ndof!=ndof) {
        formResidualAndJacobian(Mesh,t_PDOperators,t_ElmtSystem,Info,U,Uold,K,RHS);
        return;
    }

    // Parity with the CPU path: refresh any frozen per-iteration caches
    // (a no-op for the kernels handled here, but keeps the contract identical).
    t_ElmtSystem.preprocessIteration(Mesh,t_PDOperators,Info,ndof,U,Uold);

    const auto &md=Mesh.getDataConstRef();
    const int N=Mesh.getNodesNum();

    // PDDO multi-index tables + operator slot indices. factprod is recomputed
    // here so the GPU path needs no extra accessor on PDOperators.
    const auto &qidx=t_PDOperators.getQIndices();
    const int nop=t_PDOperators.getOperatorsVecSize();
    static std::vector<int> qp, qq, qr;
    static std::vector<double> factprod;
    qp.resize(static_cast<std::size_t>(nop));
    qq.resize(static_cast<std::size_t>(nop));
    qr.resize(static_cast<std::size_t>(nop));
    factprod.resize(static_cast<std::size_t>(nop));
    auto fact=[](int n){ double r=1.0; for (int k=2;k<=n;k++) r*=static_cast<double>(k); return r; };
    perix_cuda::OpSlots ss;
    for (int k=0;k<nop;k++) {
        const int pk=qidx[static_cast<std::size_t>(k)][0];
        const int qk=qidx[static_cast<std::size_t>(k)][1];
        const int rk=qidx[static_cast<std::size_t>(k)][2];
        qp[static_cast<std::size_t>(k)]=pk;
        qq[static_cast<std::size_t>(k)]=qk;
        qr[static_cast<std::size_t>(k)]=rk;
        factprod[static_cast<std::size_t>(k)]=fact(pk)*fact(qk)*fact(rk);
        if (pk==1 && qk==0 && rk==0) ss.dx=k;
        if (pk==0 && qk==1 && rk==0) ss.dy=k;
        if (pk==0 && qk==0 && rk==1) ss.dz=k;
        if (pk==2 && qk==0 && rk==0) ss.dxx=k;
        if (pk==1 && qk==1 && rk==0) ss.dxy=k;
        if (pk==0 && qk==2 && rk==0) ss.dyy=k;
        if (pk==1 && qk==0 && rk==1) ss.dxz=k;
        if (pk==0 && qk==1 && rk==1) ss.dyz=k;
        if (pk==0 && qk==0 && rk==2) ss.dzz=k;
    }

    // Flatten the neighbor lists into CSR form. The cache is rebuilt whenever
    // the node count OR the total bond count changes: keying on N alone would
    // hand back a stale list if connectivity changed while N stayed fixed
    // (e.g. bonds removed by an evolving crack), so we also track the summed
    // neighbor count. The sum is an O(N) read of the per-node list sizes and
    // is negligible next to the device assembly + transfers it guards.
    static std::vector<int> neigh_off, neigh_id;
    static int cachedN=-1, cachedNbtot=-1;
    int nbtot_now=0;
    for (int i=0;i<N;i++)
        nbtot_now+=static_cast<int>(md.NodesNeighNodesID[static_cast<std::size_t>(i)].size());
    if (cachedN!=N || cachedNbtot!=nbtot_now) {
        neigh_off.assign(static_cast<std::size_t>(N)+1,0);
        for (int i=0;i<N;i++) {
            neigh_off[static_cast<std::size_t>(i)+1]=neigh_off[static_cast<std::size_t>(i)]
                +static_cast<int>(md.NodesNeighNodesID[static_cast<std::size_t>(i)].size());
        }
        neigh_id.resize(static_cast<std::size_t>(neigh_off[static_cast<std::size_t>(N)]));
        for (int i=0;i<N;i++) {
            const int b=neigh_off[static_cast<std::size_t>(i)];
            const auto &nb=md.NodesNeighNodesID[static_cast<std::size_t>(i)];
            for (std::size_t t=0;t<nb.size();++t) neigh_id[static_cast<std::size_t>(b)+t]=nb[t];
        }
        cachedN=N; cachedNbtot=nbtot_now;
    }

    // variable horizon: per-node delta_i / spacing (empty vectors => uniform)
    const bool varH = md.VariableHorizon
                      && static_cast<int>(md.NodeHorizon.size())>=N
                      && static_cast<int>(md.NodeSpacing.size())>=N;

    perix_cuda::AssembleMesh am;
    am.N=N;
    am.nnz=K.getNNZNum();
    am.csr_off=K.getCSRRowsIndexPtr();
    am.csr_col=K.getCSRColsIndexPtr();
    am.neigh_off=neigh_off.data();
    am.neigh_id=neigh_id.data();
    am.coords=md.NodeCoords.data();
    am.vols=md.NodeVolumes.data();
    am.qp=qp.data();
    am.qq=qq.data();
    am.qr=qr.data();
    am.factprod=factprod.data();
    am.nop=nop;
    am.dim=dim;
    am.delta=md.HorizonRadius;
    // VolumeCorrection off -> zero spacing so devVolCorrection yields 1.0 for
    // every bond (uniform via dx_char; variable horizon via a zeroed spacing
    // array, since the kernels read nodeSpacing[i] whenever nodeHorizon is set).
    am.dx_char=noVc ? 0.0 : std::max({md.DX,md.DY,md.DZ});
    am.nodeHorizon=varH ? md.NodeHorizon.data() : nullptr;
    static std::vector<double> zeroSpacing;
    if (varH && noVc) zeroSpacing.assign(static_cast<std::size_t>(N),0.0);
    am.nodeSpacing=varH ? (noVc ? zeroSpacing.data() : md.NodeSpacing.data()) : nullptr;

    bool ok;
    if (pdf) {
        // implicit PDDO dynamic fracture: nodecache = committed velocity (3 slots,
        // z padded in 2D), bondcache = frozen per-bond health flattened to the mesh
        // family order (== am.neigh_*; the element indexes health by family slot).
        perix_cuda::CachedParams cp;
        cp.ndof=dim; cp.dim=dim; cp.ncache=3; cp.dt=Info.Dt;
        cp.C11=pdf->getC11(); cp.C12=pdf->getC12(); cp.C66=pdf->getC66();
        cp.rho=pdf->getRho(); cp.bx=pdf->getBx(); cp.by=pdf->getBy(); cp.bz=pdf->getBz();
        static std::vector<double> nodecache;
        nodecache.assign(static_cast<std::size_t>(N)*3,0.0);
        const auto &vel=pdf->getVel();
        if (pdf->getInitialized() && static_cast<int>(vel.size())>=N*dim)
            for (int i=0;i<N;i++)
                for (int d=0;d<dim;d++)
                    nodecache[static_cast<std::size_t>(3*i+d)]=vel[static_cast<std::size_t>(i)*dim+d];
        const auto &bh=pdf->getBondHealth();
        const int nbtot=neigh_off[static_cast<std::size_t>(N)];
        static std::vector<double> healthFlat;
        healthFlat.assign(static_cast<std::size_t>(nbtot),1.0);
        if (pdf->getInitialized() && static_cast<int>(bh.size())==N) {
            for (int i=0;i<N;i++) {
                const int b=neigh_off[static_cast<std::size_t>(i)];
                const auto &hi=bh[static_cast<std::size_t>(i)];
                for (std::size_t t=0;t<hi.size();++t)
                    healthFlat[static_cast<std::size_t>(b)+t]=hi[t];
            }
        }
        ok=perix_cuda::assembleCached(am, cp, ss, nodecache.data(), healthFlat.data(),
                                      U.getDataPtr(), uold_ptr,
                                      K.getCSRValuesPtr(), RHS.getDataPtr());
    } else if (frch) {
        // frac_stress_cahnhilliard: stress-CH conservative flux (m_LapC as BOTH the
        // flux operator and the kappa operator) + the frozen per-bond health gate on
        // the mechanical terms + optional inertia. preprocessIteration (called above)
        // committed the bond health and refreshed the mobility; flatten the nested
        // per-node health to the per-bond layout the kernel walks (== am.neigh_*).
        perix_cuda::StressCahnHilliardParams cp;
        cp.dim=dim; cp.ndof=ndof; cp.dt=Info.Dt;
        cp.C11=frch->getC11(); cp.C12=frch->getC12(); cp.C66=frch->getC66();
        cp.A=frch->getCoupling(); cp.Kh=frch->getKh(); cp.Ch=frch->getCh();
        cp.Omega=frch->getOmega(); cp.cref=frch->getCref();
        cp.chi=frch->getChi(); cp.kappa=frch->getKappa();
        cp.kres=frch->getResidualStiffness(); cp.rho=frch->getRho();
        const auto &noff=frch->getNeighOff();
        const auto &bh=frch->getBondHealth();
        const int nbtot=(static_cast<int>(noff.size())>N)?noff[static_cast<std::size_t>(N)]:0;
        static std::vector<double> healthFlat;
        healthFlat.assign(static_cast<std::size_t>(nbtot),1.0);
        if (frch->getInitialized() && static_cast<int>(bh.size())==N) {
            for (int i=0;i<N;i++) {
                const int b=noff[static_cast<std::size_t>(i)];
                const auto &hi=bh[static_cast<std::size_t>(i)];
                for (std::size_t t=0;t<hi.size();++t)
                    healthFlat[static_cast<std::size_t>(b)+t]=hi[t];
            }
        }
        const auto &lapc=frch->getLapC(); const auto &gh=frch->getGhostFlags(); const auto &vel=frch->getVel();
        const double *velp = (cp.rho>0.0 && static_cast<int>(vel.size())>=N*dim) ? vel.data() : nullptr;
        ok=perix_cuda::assembleStressCahnHilliard(am, cp, ss,
                frch->getRevBond().data(), lapc.data(),
                gh.empty()?nullptr:gh.data(), frch->getNodeMobility().data(),
                healthFlat.data(), velp,
                U.getDataPtr(), uold_ptr, K.getCSRValuesPtr(), RHS.getDataPtr());
    } else {
        ok=perix_cuda::assemble(am,pp,ss,U.getDataPtr(),uold_ptr,
                                K.getCSRValuesPtr(),RHS.getDataPtr());
    }
    if (!ok) {
        // device unavailable / unsupported at run time -> correct CPU path
        formResidualAndJacobian(Mesh,t_PDOperators,t_ElmtSystem,Info,U,Uold,K,RHS);
    }
}
