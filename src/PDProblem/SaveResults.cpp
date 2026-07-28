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
//+++ Function: save results into pd/fem mesh using vtu file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "PDProblem/PDProblem.h"

namespace {
    [[nodiscard]] std::string makeStepSuffix(const int &step) {
        if (step<=0) return "";
        char buf[16];
        std::snprintf(buf,sizeof(buf),"-%05d",step);
        return std::string(buf);
    }

    [[nodiscard]] std::string stripDir(const std::string &path) {
        const auto pos=path.find_last_of("/\\");
        return (pos==std::string::npos) ? path : path.substr(pos+1);
    }

    [[nodiscard]] std::string stripJsonExt(const std::string &name) {
        if (name.size()>=5 && name.compare(name.size()-5,5,".json")==0) {
            return name.substr(0,name.size()-5);
        }
        return name;
    }

    // Per-DoF field names for the VTU solution output. Uses the first
    // registered element's canonical DoF names, which ReadDOFsBlock
    // validates against the user's "DOFs" block and the m_U component
    // order, so name a-1 always matches DoF a. Falls back to Solution_k
    // if the names are unavailable or their count disagrees with ndof.
    [[nodiscard]] std::vector<std::string> resolveDofNames(const ElmtSystem &elmtSys,
                                                           const int &ndof) {
        std::vector<std::string> names;
        if (elmtSys.getElementsNum()>0) names=elmtSys.getElementByIndex(0).getDofNames();
        if (static_cast<int>(names.size())!=ndof) {
            names.clear();
            for (int a=1;a<=ndof;a++) names.push_back("Solution_"+std::to_string(a));
        }
        return names;
    }
}

void PDProblem::savePDResults(const string &inputfilename,const int &step,const double &time) {
    const string tag = "-PD-results";
    const string filename = stripJsonExt(inputfilename)
                            +tag+makeStepSuffix(step)+".vtu";
    std::ofstream out(filename);
    if (!out.is_open()){
        std::cerr << "Error: cannot open file " << filename << '\n';
        return;
    }

    const std::size_t N = m_PDMesh.getNodesNum();

    // Use high precision for coordinates / weights
    out << std::setprecision(16);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    // -------------------------------------------------------------------------
    // Point data
    // -------------------------------------------------------------------------
    const int ndof=m_ElmtSystem.getMaxDofsPerNode();
    const std::vector<std::string> dofNames=resolveDofNames(m_ElmtSystem,ndof);
    out << "      <PointData Scalars=\""
        << (dofNames.empty()?std::string("ElmtID"):dofNames[0]) << "\">\n";
    out << "        <DataArray type=\"Int64\" Name=\"ElmtID\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i){
        out << m_PDMesh.getDataConstRef().NodesElmtID[i] << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    // 1) the solution -- one named scalar per DoF, using the DoF names
    //    from the "DOFs" block (e.g. Cc, Ca, psi) instead of a single
    //    bundled "Solution" array that lumped every DoF together.
    for (int a=1;a<=ndof;a++) {
        out << "        <DataArray type=\"Float64\" Name=\""
            << dofNames[static_cast<std::size_t>(a-1)]
            << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
        out << "          ";
        for (std::size_t i = 0; i < N; ++i){
            out << m_U(static_cast<int>(i)*ndof+a) << ' ';
        }
        out << "\n";
        out << "        </DataArray>\n";
    }

    // 2) user-requested derived projections (strain / stress / vonMises / gradient / flux / ...)
    const auto &fields=m_OutputSystem.getResolvedFields();
    if (!fields.empty() && m_ElmtSystem.getElementsNum()>0) {
        // each projection is computed by the kernel that ADVERTISES it (resolve()
        // guarantees one exists); calling element[0] for everything wrote silent
        // zeros for a field owned by element[1..] in a multi-element run.
        std::vector<const ElementBase*> providers(fields.size());
        for (std::size_t f=0;f<fields.size();++f) {
            providers[f]=m_ElmtSystem.findProjectionProvider(fields[f].Name);
            if (!providers[f]) providers[f]=&m_ElmtSystem.getElementByIndex(0);
        }

        // Per-field per-node storage
        std::vector<std::vector<double>> fieldVals(fields.size());
        for (std::size_t f=0;f<fields.size();++f) {
            fieldVals[f].assign(static_cast<std::size_t>(N)*static_cast<std::size_t>(fields[f].Components),0.0);
        }
        std::vector<double> nodeBuf;
        for (int i=1;i<=static_cast<int>(N);++i) {
            m_Operators.calcAMatrix(i,m_PDMesh.getDataConstRef());
            for (std::size_t f=0;f<fields.size();++f) {
                const int comps=fields[f].Components;
                nodeBuf.assign(static_cast<std::size_t>(comps),0.0);
                providers[f]->computeNodalProjection(fields[f].Name,m_PDMesh,m_Operators,m_U,i,ndof,nodeBuf);
                for (int c=0;c<comps;++c) {
                    fieldVals[f][(static_cast<std::size_t>(i)-1)*static_cast<std::size_t>(comps)
                                 +static_cast<std::size_t>(c)]=nodeBuf[static_cast<std::size_t>(c)];
                }
            }
        }
        for (std::size_t f=0;f<fields.size();++f) {
            const int comps=fields[f].Components;
            out << "        <DataArray type=\"Float64\" Name=\""<<fields[f].Name
                <<"\" NumberOfComponents=\""<<comps<<"\" format=\"ascii\">\n";
            for (int i=0;i<static_cast<int>(N);++i) {
                for (int c=0;c<comps;++c) {
                    out << fieldVals[f][static_cast<std::size_t>(i)*static_cast<std::size_t>(comps)
                                        +static_cast<std::size_t>(c)];
                    out << (c+1<comps?' ':'\n');
                }
            }
            out << "        </DataArray>\n";
        }
    }

    out << "      </PointData>\n";

    // Empty CellData block
    out << "      <CellData>\n";
    out << "      </CellData>\n";

    // -------------------------------------------------------------------------
    // Points: x y z for each node
    // -------------------------------------------------------------------------
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << m_PDMesh.getDataConstRef().NodeCoords[i*3+0] << ' '
            << m_PDMesh.getDataConstRef().NodeCoords[i*3+1] << ' '
            << m_PDMesh.getDataConstRef().NodeCoords[i*3+2]<<"\n";
    }
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // -------------------------------------------------------------------------
    // Cells: one VTK_VERTEX cell per point
    //
    // connectivity: point index for each cell
    // offsets: cumulative number of point indices after each cell
    // types: VTK cell type, 1 = VTK_VERTEX
    // -------------------------------------------------------------------------
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << static_cast<long long>(i);
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    // offsets
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << static_cast<long long>(i + 1);
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    // types
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t i = 0; i < N; ++i)
    {
        out << 1; // VTK_VERTEX
        if (i + 1 < N) out << ' ';
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    // append to the time-series and rewrite the .pvd index
    m_PDOutputs.emplace_back(time,stripDir(filename));
    const string pvdPath = stripJsonExt(inputfilename)+tag+".pvd";
    writePVDFile(pvdPath,m_PDOutputs);
}

void PDProblem::saveElementResults(const string &inputfilename,const int &step,const double &time) {
    const string filename = stripJsonExt(inputfilename)
                            +"-Element-results"+makeStepSuffix(step)+".vtu";
    std::ofstream out;
    out.open(filename, std::ios::out);
    if (!out.is_open()) {
        cout<<"*** Error: can\'t create/open file="<<filename<<", please make sure you have the write permission"<<endl;
        abort();
    }
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
    out << "<UnstructuredGrid>\n";
    out << "<Piece NumberOfPoints=\"" << m_Mesh.getNodesNum() << "\" NumberOfCells=\"" << m_Mesh.getBulkElmtsNum() <<"\">\n";
    out << "<Points>\n";
    out << "<DataArray type=\"Float64\" Name=\"nodes\"  NumberOfComponents=\"3\"  format=\"ascii\">\n";

    //*****************************
    // print out node coordinates -- use full double precision so
    // downstream post-processing (analytical comparison, gradient
    // recovery on the FEM side, etc.) is not capped by the file
    // format.
    out << std::scientific << std::setprecision(16);
    for (int i = 0; i < m_Mesh.getNodesNum(); i++) {
        out << m_Mesh.getMeshDataRef().NodeCoords[i * 3 + 1 - 1] << " "
            << m_Mesh.getMeshDataRef().NodeCoords[i * 3 + 2 - 1] << " "
            << m_Mesh.getMeshDataRef().NodeCoords[i * 3 + 3 - 1] << "\n";
    }
    out << "</DataArray>\n";
    out << "</Points>\n";

    //***************************************
    //*** For cell information
    //***************************************
    out << "<Cells>\n";
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (const auto &cell: m_Mesh.getMeshDataRef().BulkElmtConn) {
        for (const auto &id: cell) {
            out << id - 1 << " ";
        }
        out << "\n";
    }
    out << "</DataArray>\n";

    //***************************************
    //*** For offset
    //***************************************
    out << "<DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    int offset = 0;
    for (int e=1;e<=m_Mesh.getMeshDataRef().BulkElmtsNum;e++) {
        offset += m_Mesh.getMeshDataRef().NodesNumPerBulkElmt;
        out << offset << "\n";
    }
    out << "</DataArray>\n";

    //***************************************
    //*** For vtk cell type
    //***************************************
    out << "<DataArray type=\"Int32\" Name=\"types\"  NumberOfComponents=\"1\"  format=\"ascii\">\n";
    for (int e=1;e<=m_Mesh.getMeshDataRef().BulkElmtsNum;e++) {
        out << m_Mesh.getMeshDataRef().BulkElmtVTKCellType << "\n";
    }
    out << "</DataArray>\n";
    out << "</Cells>\n";

    // for cell data
    out << "<CellData>\n";
    const int ndof=m_ElmtSystem.getMaxDofsPerNode();
    const auto &meshD=m_Mesh.getMeshDataRef();
    const std::vector<std::string> dofNames=resolveDofNames(m_ElmtSystem,ndof);

    // 1) the solution -- one named scalar per DoF (Cc, Ca, psi, ...)
    //    rather than a single bundled "Solution" array.
    for (int a=1;a<=ndof;a++) {
        out << "<DataArray type=\"Float64\" Name=\""
            << dofNames[static_cast<std::size_t>(a-1)]
            << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
        for (int e=1;e<=meshD.BulkElmtsNum;e++) {
            const int NodeID=meshD.BulkElmtPDNodeID[e-1];
            out << m_U((NodeID-1)*ndof+a) << "\n";
        }
        out<<"</DataArray>\n";
    }

    // 2) user-requested derived projections (computed on PD nodes, copied to FEM cells)
    const auto &fields=m_OutputSystem.getResolvedFields();
    if (!fields.empty() && m_ElmtSystem.getElementsNum()>0) {
        // per-field provider (see savePDResults): the advertising kernel computes it
        std::vector<const ElementBase*> providers(fields.size());
        for (std::size_t f=0;f<fields.size();++f) {
            providers[f]=m_ElmtSystem.findProjectionProvider(fields[f].Name);
            if (!providers[f]) providers[f]=&m_ElmtSystem.getElementByIndex(0);
        }
        // Cache projection at each PD bulk node referenced by a FEM element.
        // (Bulk node ids form a contiguous range [1, BulkElmtsNum].)
        std::vector<std::vector<double>> fieldVals(fields.size());
        for (std::size_t f=0;f<fields.size();++f) {
            fieldVals[f].assign(static_cast<std::size_t>(meshD.BulkElmtsNum)
                                *static_cast<std::size_t>(fields[f].Components),0.0);
        }
        std::vector<double> nodeBuf;
        for (int e=1;e<=meshD.BulkElmtsNum;e++) {
            const int NodeID=meshD.BulkElmtPDNodeID[e-1];
            m_Operators.calcAMatrix(NodeID,m_PDMesh.getDataConstRef());
            for (std::size_t f=0;f<fields.size();++f) {
                const int comps=fields[f].Components;
                nodeBuf.assign(static_cast<std::size_t>(comps),0.0);
                providers[f]->computeNodalProjection(fields[f].Name,m_PDMesh,m_Operators,m_U,NodeID,ndof,nodeBuf);
                for (int c=0;c<comps;++c) {
                    fieldVals[f][(static_cast<std::size_t>(e)-1)*static_cast<std::size_t>(comps)
                                 +static_cast<std::size_t>(c)]=nodeBuf[static_cast<std::size_t>(c)];
                }
            }
        }
        for (std::size_t f=0;f<fields.size();++f) {
            const int comps=fields[f].Components;
            out << "<DataArray type=\"Float64\" Name=\""<<fields[f].Name
                <<"\" NumberOfComponents=\""<<comps<<"\" format=\"ascii\">\n";
            for (int e=0;e<meshD.BulkElmtsNum;e++) {
                for (int c=0;c<comps;++c) {
                    out << fieldVals[f][static_cast<std::size_t>(e)*static_cast<std::size_t>(comps)
                                        +static_cast<std::size_t>(c)];
                    out << (c+1<comps?' ':'\n');
                }
            }
            out<<"</DataArray>\n";
        }
    }
    out << "</CellData>\n";

    //***************************************
    //*** End of output
    //***************************************
    out << "</Piece>\n";
    out << "</UnstructuredGrid>\n";
    out << "</VTKFile>" << endl;

    out.close();

    // append to the time-series and rewrite the .pvd index
    m_ElmtOutputs.emplace_back(time,stripDir(filename));
    const string pvdPath = stripJsonExt(inputfilename)+"-Element-results.pvd";
    writePVDFile(pvdPath,m_ElmtOutputs);
}

void PDProblem::saveExodusResults(const string &inputfilename,const int &step,const double &time) {
    (void)step;
    const int ndof=m_ElmtSystem.getMaxDofsPerNode();
    const auto &meshD=m_Mesh.getMeshDataRef();
    const std::vector<std::string> dofNames=resolveDofNames(m_ElmtSystem,ndof);
    const auto &fields=m_OutputSystem.getResolvedFields();
    const int nelem=meshD.BulkElmtsNum;

    // result field names = one per DoF, then one per projection component
    // (split a multi-component projection into scalar element variables).
    std::vector<std::string> varNames=dofNames;
    for (const auto &fld : fields)
        for (int c=0;c<fld.Components;c++)
            varNames.push_back(fld.Components>1 ? fld.Name+"_"+std::to_string(c+1) : fld.Name);

    // write the mesh + schema once, then append a record each output step.
    if (!m_ExodusBegun) {
        const string fname=stripJsonExt(inputfilename)+".e";
        if (!m_ExodusWriter.begin(fname,meshD,varNames,stripDir(stripJsonExt(inputfilename)))) return;
        m_ExodusBegun=true;
    }

    // per-element values: each bulk element carries the PD point at its centroid
    // (BulkElmtPDNodeID), exactly as the Element-results vtu.
    std::vector<std::vector<double>> elemVals(varNames.size(),
                                              std::vector<double>(static_cast<std::size_t>(nelem),0.0));
    for (int a=1;a<=ndof;a++)
        for (int e=1;e<=nelem;e++) {
            const int NodeID=meshD.BulkElmtPDNodeID[static_cast<std::size_t>(e-1)];
            elemVals[static_cast<std::size_t>(a-1)][static_cast<std::size_t>(e-1)]=m_U((NodeID-1)*ndof+a);
        }
    if (!fields.empty() && m_ElmtSystem.getElementsNum()>0) {
        // per-field provider (see savePDResults): the advertising kernel computes it
        std::vector<const ElementBase*> providers(fields.size());
        for (std::size_t f=0;f<fields.size();++f) {
            providers[f]=m_ElmtSystem.findProjectionProvider(fields[f].Name);
            if (!providers[f]) providers[f]=&m_ElmtSystem.getElementByIndex(0);
        }
        std::vector<double> nodeBuf;
        for (int e=1;e<=nelem;e++) {
            const int NodeID=meshD.BulkElmtPDNodeID[static_cast<std::size_t>(e-1)];
            m_Operators.calcAMatrix(NodeID,m_PDMesh.getDataConstRef());
            std::size_t vi=static_cast<std::size_t>(ndof);
            std::size_t f=0;
            for (const auto &fld : fields) {
                nodeBuf.assign(static_cast<std::size_t>(fld.Components),0.0);
                providers[f]->computeNodalProjection(fld.Name,m_PDMesh,m_Operators,m_U,NodeID,ndof,nodeBuf);
                ++f;
                for (int c=0;c<fld.Components;c++)
                    elemVals[vi++][static_cast<std::size_t>(e-1)]=nodeBuf[static_cast<std::size_t>(c)];
            }
        }
    }
    m_ExodusWriter.appendStep(time,elemVals);
}

void PDProblem::writePVDFile(const string &pvdFilePath,
                             const std::vector<std::pair<double,string>> &entries) const {
    std::ofstream out(pvdFilePath);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open file " << pvdFilePath << '\n';
        return;
    }
    out << std::setprecision(16);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";
    for (const auto &[time,filename] : entries) {
        out << "    <DataSet timestep=\"" << time
            << "\" group=\"\" part=\"0\" file=\"" << filename << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
    out.close();
}
