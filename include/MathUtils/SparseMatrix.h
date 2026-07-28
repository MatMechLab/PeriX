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

#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

#include "MathUtils/SparseMatrixData.h"
#include "MathUtils/VectorXd.h"
#include "Utils/MessagePrinter.h"
#include "PDMesh/PDMeshData.h"

using namespace std;

/**
 * This class implement the sparse matrix class for FEM analysis
 */
class SparseMatrix {
public:
    /**
     * constructor
     */
    SparseMatrix();

    /**
     * constructor with other sparse matrix
     * @param other the input sparse matrix
     */
    SparseMatrix(const SparseMatrix& other);

    /**
     * constructor with size
     * @param size the size of the sparse matrix
     */
    SparseMatrix(const int &size);


    /**
     * initialize the sparse-matrix sparsity pattern from a PD mesh.
     * The matrix has size NodesNum*DofsPerNode. For every (i,j) pair
     * (i, neighbor j) an n_dof x n_dof block is inserted with row a
     * = (i-1)*ndof+1..i*ndof and col b = (j-1)*ndof+1..j*ndof, plus
     * the diagonal block (i,i). Ghost rows also keep a dense block to
     * their mirror bulk node so boundary-condition constraints remain in
     * the fixed CSR pattern even when a crack deletes the physical bond.
     * @param PDData the pd mesh data
     * @param DofsPerNode number of degrees of freedom per pd node, default 1
     */
    void initFromPDMesh(const PDMeshData &PDData,const int &DofsPerNode=1);

    /**
     * copy the current sparse matrix to another sparse matrix
     * @param other the input sparse matrix
     */
    void copy2Matrix(SparseMatrix &other);

    /**
     * - operator
     * @param other right hand side sparse matrix
     * @return calculated sparse matrix
     */
    SparseMatrix operator-(const SparseMatrix &other) const {
        if (other.m_Data.RowsNum != m_Data.RowsNum ||
            other.m_Data.RowIndices != m_Data.RowIndices ||
            other.m_Data.ColumnIndices != m_Data.ColumnIndices ||
            other.m_Data.Values.size() != m_Data.Values.size()) {
            cout<<"*** Error: cant't apply - to sparse matrix with different CSR structure !!!"<<endl;
            abort();
        }
        SparseMatrix result(*this);
        for (size_t i=0;i<m_Data.Values.size();++i) {
            result.m_Data.Values[i]-=other.m_Data.Values[i];
        }
        return result;
    }

    /**
     * set the size of the sparse matrix
     * @param size the size of the sparse matrix
     */
    void setSize(const int &size) {
        if (size<0) {
            cout<<"*** Error: sparse matrix size can't be negative !!!"<<endl;
            abort();
        }
        m_Data.RowsNum=size;
        m_Data.RowIndices.assign(size+1,0);
        m_Data.ColumnIndices.clear();
        m_Data.Values.clear();
        m_Data.MaxRowNNZ=0;
        m_Data.NNZ=0;
        m_Data.RowsColumnIndices.clear();
        m_Data.RowsColumnIndices.resize(size);
        m_Data.DiagonalGlobalIndices.assign(size,-1);
        m_Data.HasFullDiagonal=(size==0);
        m_Data.FirstMissingDiagonalRow=(size==0 ? -1 : 1);
    }

    /**
     * setup the sparse matrix
     */
    void setup();

    /**
     * insert row,col coordinate into the sparse matrix
     * @param row row id, start from 1
     * @param col col id, start from 1
     */
    void insert(const int &row, const int &col);

    /**
     * add value into the sparse matrix
     * @param RowID row id, start from 1
     * @param ColID col id, start from 1
     * @param Val the element value
     */
    void addValue(const int &RowID, const int &ColID, const double &Val);

    /**
     * insert value into the sparse matrix, the original one will be replaced
     * @param RowID row id, start from 1
     * @param ColID col id, start from 1
     * @param Val val id, start from 1
     */
    void insertValue(const int &RowID, const int &ColID, const double &Val);

    /**
     * add a scalar value to all diagonal entries, i.e. K_{i,i} += Scalar
     * @param Scalar the scalar value added to each diagonal entry
     */
    void addDiagonalValue(const double &Scalar);

    /**
     * add a vector value to the diagonal entries, i.e. K_{i,i} += DiagValues(i)
     * @param DiagValues the diagonal increment vector
     */
    void addDiagonalValues(const VectorXd &DiagValues);

    /**
     * value of the (Row,Row) diagonal entry, or 0.0 if this row stores no
     * diagonal. Used to pick a representative scale when imposing a Dirichlet
     * constraint by row elimination.
     * @param Row row id, start from 1
     */
    [[nodiscard]] double getDiagonal(const int &Row) const;

    /**
     * largest |entry| currently stored in Row (0.0 if the row is empty).
     * Serves as a fall-back row scale when the diagonal is (near) zero.
     * @param Row row id, start from 1
     */
    [[nodiscard]] double getRowMaxAbs(const int &Row) const;

    /**
     * impose a Dirichlet row by exact elimination: zero every stored entry of
     * Row, then set the diagonal entry (Row,Row) to DiagVal. The CSR sparsity
     * pattern is preserved (entries are zeroed in place, never removed), so the
     * factorization structure is unchanged. This replaces the ill-conditioned
     * diagonal-penalty trick: the constrained equation becomes
     * DiagVal*dU(Row)=RHS(Row) with DiagVal kept at the row's natural scale,
     * leaving the condition number of the assembled system intact.
     * @param Row     row id, start from 1
     * @param DiagVal the (nonzero) value written to the diagonal entry
     */
    void zeroRowAndSetDiagonal(const int &Row, const double &DiagVal);

    /**
     * set values vector to zeros
     */
    void setToZeros() {
        fill(m_Data.Values.begin(),m_Data.Values.end(),0.0);
    }

    /**
     * print out the sparse matrix
     */
    void print() const;

    /**
     * print out the sparse matrix with csr structure info
     */
    void printMatrix() const;

    /**
     * get the size of current sparse matrix
     * @return size of current matrix
     */
    [[nodiscard]] int getSize() const {
        return m_Data.RowsNum;
    }

    /**
     * get the pointer of row indices
     * @return the pointer of the row indices
     */
    int* getCSRRowsIndexPtr() {
        return m_Data.RowIndices.data();
    }

    /**
     * get the copy of row indices vector
     * @return copy of row index vector
     */
    [[nodiscard]] vector<int> getCSRRowsIndexCopy()const {
        return m_Data.RowIndices;
    }

    /**
     * get the pointer of the col indices
     * @return the pointer of col indices
     */
    int* getCSRColsIndexPtr() {
        return m_Data.ColumnIndices.data();
    }

    /**
     * get the copy of col indices vector
     * @return copy of col indices
     */
    [[nodiscard]] vector<int> getCSRColsIndexCopy()const {
        return m_Data.ColumnIndices;
    }

    /**
     * get the pointer of values
     * @return the pointer of values
     */
    double* getCSRValuesPtr() {
        return m_Data.Values.data();
    }

    /**
     * get the copy of csr values vector
     * @return copy of csr values
     */
    [[nodiscard]] vector<double> getCSRValuesCopy()const {
        return m_Data.Values;
    }

    /**
     * get the number of nonzero elements
     * @return nonzeros number
     */
    [[nodiscard]] int getNNZNum() const {
        return m_Data.NNZ;
    }

    /**
     * get the norm of the sparse matrix
     * @return norm of the sparse matrix
     */
    [[nodiscard]] double getNorm()const {
        double sum=0.0;
        for (const auto &val:m_Data.Values) {
            sum+=val*val;
        }
        return sqrt(sum);
    }

private:
    /**
     * locate the zero-based global CSR index of entry (Row0Based,Col0Based) by
     * binary-searching the sorted column slice of that row. Returns -1 if the
     * (row,col) pair is not part of the stored sparsity pattern. This replaces
     * the former per-row unordered_map (one hash table per row, ~40 bytes of
     * heap overhead per stored nonzero) with an O(log nnz_row) search over the
     * already-sorted, contiguous ColumnIndices — the same scheme the CUDA
     * assembler uses on the device (devFindSlot).
     * @param Row0Based zero-based row id
     * @param Col0Based zero-based column id
     */
    [[nodiscard]] int findEntry(const int &Row0Based,const int &Col0Based) const;

    /**
     * insert the col id into the specific row's coldid vector
     * @param RowID row id start from 1
     * @param ColID col id start from 1
     * @param ColIDVec colid vector of current rowid
     */
    void insertColID2Row(const int &RowID,const int &ColID,vector<int> &ColIDVec);

private:
    CSRSparseMatrixData m_Data;/**< the data of the sparse matrix */

};
