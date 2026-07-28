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
//+++ Function: the sparse matrix class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MathUtils/SparseMatrix.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

SparseMatrix::SparseMatrix() = default;

SparseMatrix::SparseMatrix(const SparseMatrix &other) {
    m_Data=other.m_Data;
}

SparseMatrix::SparseMatrix(const int &size) {
    setSize(size);
}

void SparseMatrix::copy2Matrix(SparseMatrix &other) {
    other.m_Data=m_Data;
}

void SparseMatrix::insertColID2Row(const int &RowID, const int &ColID, vector<int> &ColIDVec) {
    (void)RowID;
    const int ZeroBasedColID=ColID-1;

    if (ColIDVec.empty()) {
        ColIDVec.push_back(ZeroBasedColID);
        return;
    }
    if (ZeroBasedColID>ColIDVec.back()) {
        ColIDVec.push_back(ZeroBasedColID);
        return;
    }
    if (ZeroBasedColID<ColIDVec.front()) {
        ColIDVec.insert(ColIDVec.begin(),ZeroBasedColID);
        return;
    }

    const auto it=lower_bound(ColIDVec.begin(),ColIDVec.end(),ZeroBasedColID);
    if (it==ColIDVec.end() || *it!=ZeroBasedColID) {
        ColIDVec.insert(it,ZeroBasedColID);
    }
}

void SparseMatrix::insert(const int &RowID, const int &ColID) {
    if (RowID<=0 || RowID>m_Data.RowsNum || ColID<=0 || ColID>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("*** insert fails, ("+to_string(RowID)+","+to_string(ColID)+") is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
    // The pattern scaffold is consumed (freed) by setup(); inserting afterwards
    // would index a released vector. Rebuilding a pattern requires setSize().
    if (static_cast<int>(m_Data.RowsColumnIndices.size())!=m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("insert fails, the sparsity pattern was already finalised by setup(); "
                                      "call setSize() first to build a new pattern");
        MessagePrinter::exitPeriX();
    }
    insertColID2Row(RowID,ColID,m_Data.RowsColumnIndices[RowID-1]);
}

void SparseMatrix::initFromPDMesh(const PDMeshData &Data,const int &DofsPerNode) {
    if (DofsPerNode<1) {
        MessagePrinter::printErrorTxt("SparseMatrix::initFromPDMesh: DofsPerNode must be >=1, got "
                                      +to_string(DofsPerNode));
        MessagePrinter::exitPeriX();
    }

    setSize(Data.NodesNum*DofsPerNode);
    if (m_Data.RowsNum==0) {
        setup();
        return;
    }

    const int n=DofsPerNode;
    // Build the unique set of column nodes for row block i, then emit each
    // n-by-n dense block once.
    std::vector<int> colNodes;
    for (int i=1;i<=Data.NodesNum;i++) {
        colNodes.clear();
        colNodes.push_back(i);                               // diagonal block (i,i)
        for (const auto &ColNodeID:Data.NodesNeighNodesID[i-1]) {
            colNodes.push_back(ColNodeID);                    // first-ring block (i,j)
        }
        // Boundary ghost rows are algebraic constraints tied to their mirror bulk
        // node; keep that column even if an initial crack deleted the ghost-bulk bond.
        if (static_cast<int>(Data.GhostMirrorBulkID.size())>=Data.NodesNum) {
            const int MirrorBulkID=Data.GhostMirrorBulkID[i-1];
            if (MirrorBulkID>=1 && MirrorBulkID<=Data.NodesNum) colNodes.push_back(MirrorBulkID);
        }
        std::sort(colNodes.begin(),colNodes.end());
        colNodes.erase(std::unique(colNodes.begin(),colNodes.end()),colNodes.end());
        for (int a=1;a<=n;a++) {
            auto &row=m_Data.RowsColumnIndices[static_cast<std::size_t>((i-1)*n+a-1)];
            row.reserve(row.size()+colNodes.size()*static_cast<std::size_t>(n));
            for (const int c:colNodes)
                for (int b=0;b<n;b++) row.push_back((c-1)*n+b);   // 0-based column id
        }
    }

    for (auto &RowCols:m_Data.RowsColumnIndices) {
        sort(RowCols.begin(),RowCols.end());
        RowCols.erase(unique(RowCols.begin(),RowCols.end()),RowCols.end());
    }

    setup();
}

void SparseMatrix::setup() {
    // setup() consumes (frees) the pattern scaffold, so a second call on the
    // same pattern would silently produce an empty matrix. Rebuilding requires
    // setSize() + insert (or initFromPDMesh, which starts with setSize()).
    if (static_cast<int>(m_Data.RowsColumnIndices.size())!=m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("SparseMatrix::setup fails, the pattern scaffold was already "
                                      "consumed by a previous setup(); call setSize() and re-insert "
                                      "the pattern before calling setup() again");
        MessagePrinter::exitPeriX();
    }

    m_Data.RowIndices.assign(m_Data.RowsNum+1,0);
    m_Data.DiagonalGlobalIndices.assign(m_Data.RowsNum,-1);
    m_Data.HasFullDiagonal=true;
    m_Data.FirstMissingDiagonalRow=-1;
    m_Data.MaxRowNNZ=0;

    // Accumulate row offsets in 64-bit. A large three-dimensional horizon can
    // exceed 2^31 nonzeros; fail before a 32-bit CSR offset can wrap.
    long long nnzAcc=0;
    for (int Row=0;Row<m_Data.RowsNum;++Row) {
        const int RowNNZ=static_cast<int>(m_Data.RowsColumnIndices[Row].size());
        if (RowNNZ>m_Data.MaxRowNNZ) {
            m_Data.MaxRowNNZ=RowNNZ;
        }
        nnzAcc+=RowNNZ;
        if (nnzAcc>static_cast<long long>(std::numeric_limits<int>::max())) {
            MessagePrinter::printErrorTxt("SparseMatrix::setup fails, the sparsity pattern has more than "
                                          "2^31-1 nonzeros (32-bit CSR index overflow at row "
                                          +to_string(Row+1)+" of "+to_string(m_Data.RowsNum)
                                          +"). The mesh/horizon combination is too large for the 32-bit "
                                          "CSR build; reduce the model size or the horizon factor.");
            MessagePrinter::exitPeriX();
        }
        m_Data.RowIndices[Row+1]=static_cast<int>(nnzAcc);
    }

    m_Data.NNZ=m_Data.RowIndices[m_Data.RowsNum];
    m_Data.ColumnIndices.resize(m_Data.NNZ);
    m_Data.Values.assign(m_Data.NNZ,0.0);

    for (int Row=0;Row<m_Data.RowsNum;++Row) {
        const auto &RowCols=m_Data.RowsColumnIndices[Row];   // sorted, unique (see initFromPDMesh)
        const int RowBegin=m_Data.RowIndices[Row];
        for (int LocalID=0;LocalID<static_cast<int>(RowCols.size());++LocalID) {
            const int GlobalID=RowBegin+LocalID;
            const int ColID=RowCols[LocalID];
            m_Data.ColumnIndices[GlobalID]=ColID;
            if (ColID==Row) {
                m_Data.DiagonalGlobalIndices[Row]=GlobalID;
            }
        }
        if (m_Data.DiagonalGlobalIndices[Row]<0 && m_Data.HasFullDiagonal) {
            m_Data.HasFullDiagonal=false;
            m_Data.FirstMissingDiagonalRow=Row+1;
        }
    }

    // The per-row column-id scaffold has done its job: the CSR ColumnIndices now
    // hold the (sorted, unique) pattern for every row. addValue/insertValue locate
    // a slot by binary-searching that sorted row (findEntry), so neither the
    // scaffold nor a per-row hash map is needed afterwards. Free the scaffold so it
    // is not resident for the matrix's entire lifetime (it is the same size as
    // ColumnIndices, i.e. one full copy of the connectivity).
    std::vector<std::vector<int>>().swap(m_Data.RowsColumnIndices);
}

int SparseMatrix::findEntry(const int &Row0Based,const int &Col0Based) const {
    // Binary search for Col0Based in the sorted CSR column slice of row Row0Based.
    // Returns the zero-based global CSR index, or -1 if the entry is not stored.
    const int lo=m_Data.RowIndices[Row0Based];
    const int hi=m_Data.RowIndices[Row0Based+1];
    const int *cols=m_Data.ColumnIndices.data();
    const int *it=std::lower_bound(cols+lo,cols+hi,Col0Based);
    if (it!=cols+hi && *it==Col0Based) return static_cast<int>(it-cols);
    return -1;
}

void SparseMatrix::addValue(const int &RowID, const int &ColID, const double &Val) {
#ifndef NDEBUG
    if (RowID<=0 || RowID>m_Data.RowsNum || ColID<=0 || ColID>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("addValue fails, ("+to_string(RowID)+","+to_string(ColID)+") is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
#endif

    const int idx=findEntry(RowID-1,ColID-1);
    if (idx>=0) {
        m_Data.Values[idx]+=Val;
    }
    else{
        MessagePrinter::printErrorTxt("addValue fails, ("+to_string(RowID)+","+to_string(ColID)+") is not exist in your sparse matrix");
        MessagePrinter::exitPeriX();
    }
}

void SparseMatrix::insertValue(const int &RowID, const int &ColID, const double &Val) {
#ifndef NDEBUG
    if (RowID<=0 || RowID>m_Data.RowsNum || ColID<=0 || ColID>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("insertValue fails, ("+to_string(RowID)+","
                     +to_string(ColID)+") is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
#endif

    const int idx=findEntry(RowID-1,ColID-1);
    if (idx>=0) {
        m_Data.Values[idx]=Val;
    }
    else{
        MessagePrinter::printErrorTxt("insertValue fails, ("+to_string(RowID)+","+to_string(ColID)+") is not exist in your sparse matrix");
        MessagePrinter::exitPeriX();
    }
}

void SparseMatrix::addDiagonalValue(const double &Scalar) {
    if (m_Data.RowsNum==0 || Scalar==0.0) {
        return;
    }
    if (!m_Data.HasFullDiagonal) {
        MessagePrinter::printErrorTxt("addDiagonalValue fails, diagonal entry ("+to_string(m_Data.FirstMissingDiagonalRow)
                  +","+to_string(m_Data.FirstMissingDiagonalRow)+") does not exist in your sparse matrix");
        MessagePrinter::exitPeriX();
    }

#ifndef NDEBUG
    if (static_cast<int>(m_Data.DiagonalGlobalIndices.size())!=m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("addDiagonalValue fails, diagonal cache is not initialized correctly");
        MessagePrinter::exitPeriX();
    }
    if (static_cast<int>(m_Data.Values.size())!=m_Data.NNZ) {
        MessagePrinter::printErrorTxt("addDiagonalValue fails, CSR values array is not initialized correctly");
        MessagePrinter::exitPeriX();
    }
#endif

    double *Values=m_Data.Values.data();
    const int *DiagonalGlobalIndices=m_Data.DiagonalGlobalIndices.data();
    const int RowsNum=m_Data.RowsNum;


#pragma omp simd
    for (int Row=0;Row<RowsNum;++Row) {
        Values[DiagonalGlobalIndices[Row]]+=Scalar;
    }
}

void SparseMatrix::addDiagonalValues(const VectorXd &DiagValues) {
    if (m_Data.RowsNum==0) {
        return;
    }
    if (DiagValues.getSize()!=m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("addDiagonalValues fails, vector size="+to_string(DiagValues.getSize())
                  +" is inconsistent with matrix size="+to_string(m_Data.RowsNum));
        MessagePrinter::exitPeriX();
    }
    if (!m_Data.HasFullDiagonal) {
        MessagePrinter::printErrorTxt("addDiagonalValues fails, diagonal entry ("+to_string(m_Data.FirstMissingDiagonalRow)
                  +","+to_string(m_Data.FirstMissingDiagonalRow)+") does not exist in your sparse matrix");
        MessagePrinter::exitPeriX();
    }

#ifndef NDEBUG
    if (static_cast<int>(m_Data.DiagonalGlobalIndices.size())!=m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("addDiagonalValues fails, diagonal cache is not initialized correctly");
        MessagePrinter::exitPeriX();
    }
    if (static_cast<int>(m_Data.Values.size())!=m_Data.NNZ) {
        MessagePrinter::printErrorTxt("addDiagonalValues fails, CSR values array is not initialized correctly");
        MessagePrinter::exitPeriX();
    }
#endif

    double *Values=m_Data.Values.data();
    const int *DiagonalGlobalIndices=m_Data.DiagonalGlobalIndices.data();
    const double *DiagPtr=DiagValues.getDataPtr();
    const int RowsNum=m_Data.RowsNum;


#pragma omp simd
    for (int Row=0;Row<RowsNum;++Row) {
        Values[DiagonalGlobalIndices[Row]]+=DiagPtr[Row];
    }
}

double SparseMatrix::getDiagonal(const int &Row) const {
#ifndef NDEBUG
    if (Row<=0 || Row>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("getDiagonal fails, row "+to_string(Row)
                     +" is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
#endif
    const int idx=m_Data.DiagonalGlobalIndices[Row-1];
    return (idx>=0) ? m_Data.Values[idx] : 0.0;
}

double SparseMatrix::getRowMaxAbs(const int &Row) const {
#ifndef NDEBUG
    if (Row<=0 || Row>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("getRowMaxAbs fails, row "+to_string(Row)
                     +" is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
#endif
    double maxabs=0.0;
    for (int k=m_Data.RowIndices[Row-1];k<m_Data.RowIndices[Row];++k) {
        const double a=std::fabs(m_Data.Values[k]);
        if (a>maxabs) maxabs=a;
    }
    return maxabs;
}

void SparseMatrix::zeroRowAndSetDiagonal(const int &Row, const double &DiagVal) {
#ifndef NDEBUG
    if (Row<=0 || Row>m_Data.RowsNum) {
        MessagePrinter::printErrorTxt("zeroRowAndSetDiagonal fails, row "+to_string(Row)
                     +" is outside matrix range [1,"+to_string(m_Data.RowsNum)+"]");
        MessagePrinter::exitPeriX();
    }
#endif
    const int idx=m_Data.DiagonalGlobalIndices[Row-1];
    if (idx<0) {
        MessagePrinter::printErrorTxt("zeroRowAndSetDiagonal fails, row "+to_string(Row)
                     +" has no stored diagonal entry; cannot impose a Dirichlet constraint on it");
        MessagePrinter::exitPeriX();
    }
    for (int k=m_Data.RowIndices[Row-1];k<m_Data.RowIndices[Row];++k) {
        m_Data.Values[k]=0.0;
    }
    m_Data.Values[idx]=DiagVal;
}

void SparseMatrix::print() const {
    const int size=78;
    char buffer[size];
    MessagePrinter::printNormalTxt("Summary info of sparse matrix:");
    snprintf(buffer,size,"size=%6d, nnz=%8d, max row nnz=%6d",m_Data.RowsNum,m_Data.NNZ,m_Data.MaxRowNNZ);
    MessagePrinter::printNormalTxt(buffer);
    MessagePrinter::printStars();
}

void SparseMatrix::printMatrix() const {
    for (int Row=0;Row<m_Data.RowsNum;++Row) {
        int k=0;
        printf("***");
        for (int GlobalID=m_Data.RowIndices[Row];GlobalID<m_Data.RowIndices[Row+1];++GlobalID) {
            ++k;
            printf("(%5d,%5d)=%14.5e; ",Row+1,m_Data.ColumnIndices[GlobalID]+1,m_Data.Values[GlobalID]);
            if (k%5==0) {
                printf("\n***");
                k=0;
            }
        }
        printf("\n");
    }
}
