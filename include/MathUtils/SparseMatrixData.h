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
//+++ Function: the sparse matrix data class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <vector>
#include <unordered_map>

using std::vector;
using std::unordered_map;

/**
 * the data structure of csr sparse matrix
 */
struct CSRSparseMatrixData {
    int RowsNum = 0;/**< the rows number */
    vector<int> RowIndices;/**< the row indices of csr structure */
    vector<int> ColumnIndices;/**< the col indices of csr structure */
    vector<double> Values;/**< the values of csr structure */

    int MaxRowNNZ = 0;/**< the max nnz of each row */
    int NNZ = 0;/**< the nnz of the sparse matrix */

    vector<vector<int>> RowsColumnIndices;/**< per-row sorted zero-based col ids; build-time scaffold, freed at the end of SparseMatrix::setup() once the CSR pattern is materialised */
    vector<int> DiagonalGlobalIndices;/**< zero-based global csr index of each diagonal entry, -1 if absent */
    bool HasFullDiagonal = true;/**< true if every row has an explicitly stored diagonal entry */
    int FirstMissingDiagonalRow = -1;/**< first 1-based row without diagonal entry, -1 if none */
};
