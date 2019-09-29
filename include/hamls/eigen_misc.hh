#ifndef __HAMLS_EIGEN_MISC_HH
#define __HAMLS_EIGEN_MISC_HH

//----------------------------------------------------------------------------------------------
// This file is part of the HAMLS software.
// 
// MIT License
// 
// Copyright (c) 2019 Peter Gerds
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//----------------------------------------------------------------------------------------------

// Project     : HAMLS
// File        : eigen_misc.hh
// Description : auxilliary routines for the eigen decomposition routines contained 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "cluster/TClusterTree.hh"
#include "cluster/TCoordinate.hh"
#include "matrix/TMatrix.hh"
#include "matrix/TSparseMatrix.hh"
#include "matrix/TDenseMatrix.hh"


// ========================Temporary Start ===================================//
#include <iostream>
#include <sstream>

#include "base/types.hh"
#include "cluster/TClusterTree.hh"
#include "matrix/TMatrix.hh"
#include "matrix/THMatrix.hh"
#include "matrix/TSparseMatrix.hh"
#include "matrix/TDenseMatrix.hh"
#include "algebra/mat_fac.hh"
#include "algebra/mat_add.hh"
#include "io/TMatrixIO.hh"
#include "misc/TTimer.hh"


#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tbb.h"

// ========================Ende ===================================//


namespace HAMLS
{

using namespace HLIB;

namespace EIGEN_MISC
{

    
    
//==============================================================================
//
// several auxiliary functions
//
//==============================================================================


//! return TDenseMatrix version of the input matrix
//! (Note: Internal index numbering is used )
TDenseMatrix * get_dense( const TMatrix * M ) ;

//! return number of nonzero entries
size_t nnz ( const TMatrix * M ) ;

//! check for invalid data
bool has_inf_nan_entry ( const TMatrix * M ) ;

//! A symmetric TBlockMatrix is represented in the HLIBpro by a lower triangalur TBlockMatrix. 
//! The matrix blocks above the block diagonal are set to nullptr. In the H-Matrix Algebra the blocks 
//! above the diagonal are identified by the corresponding blocks below the diagonal. This function 
//! transforms a symmetric TBlockMatrix into a equal TBlockMatrix which is not symmetric. This means
//! that matrix blocks above the diagonal are represented by real copies of their corresponding blocks 
//! below the diagonal. The upper triangular matrix of the TBlockMatrix is not anymore zero.
TMatrix * get_nonsym_of_sym ( const TMatrix * M ) ;
TMatrix * get_sym_of_nonsym ( const TMatrix * M ) ;

//! Delete the upper block triangular part of the given matrix
void delete_upper_block_triangular( TMatrix * M ) ;

//! Compute from a given symmetric matrix M the unique decomposition 
//! M = M_half + M_half^{T} with lower triangular matrix M_half 
//! (Note: The upper triangular part of M_half is represented by Nullptr)
TMatrix * get_halve_matrix( const TMatrix * M ) ;

//! Upper triangular part of lower triangular matrix gets represented 
//! by matrices with zero-valued entries instead of using Nullptr
TMatrix * get_full_format_of_lower_triangular( const TMatrix * M ) ;




///==========================================================================================
/// TODO: Better accuracy of the 'check_ortho' routine can be obtained when instead of the 
///       normal dot-product implementation the stable-version of the dot-product implementation 
///       is used. This should to be tested and implemented.
///==========================================================================================

//! return true if the matrix \a Z is orthogonal 
bool check_ortho ( const TMatrix * Z ) ;

//! return true if the \a m vectors contained in the data \a q are orthogonal 
bool check_ortho ( const std::vector < TVector * > & q,
                   const size_t                      m ) ;

//! return true if the matrix \a Z is M-orthogonal 
bool check_M_ortho ( const TMatrix * Z,
                     const TMatrix * M ) ;
                     
//! return true if M is positive definite otherwise false                     
bool is_pos_def ( const TMatrix * M ) ;

//! check the matrix entries and return true if M is symmetric otherwise false
bool is_symmetric ( const TMatrix * M ) ;

//! Computes the Rayleigh Quotients of the column vectors contained
//! in \a Z according to the general eigenvalue problem (\a K, \a M).
//! Note: If ct != nullptr then it is assumed that K is a sparse matrix 
//! and the entries of D and Z have to be permutated
void comp_rayleigh_quotients ( const TMatrix *       K,
                               const TMatrix *       M,
                               TDenseMatrix *        D,
                               const TDenseMatrix *  Z,
                               const TClusterTree *  ct = nullptr ) ;
                               
                               
//! compute the Rayleigh wuotients of the column vectors contained in \a Z_approx 
//! according to the general eigenvalue problem (\a K_sparse, \a M_sparse).
//! If \a stable_rayleigh is true then the stable dot product is used when the 
//!  Rayleigh quotient (x^T*K*x) / (x^T*M*x) is computed
void comp_rayleigh_quotients ( const TSparseMatrix * K_sparse, 
                               const TSparseMatrix * M_sparse, 
                               TDenseMatrix *        D_rayleigh,
                               const TDenseMatrix *  Z_approx,
                               const bool            stable_rayleigh,
                               const bool            do_parallel ) ;


//==============================================================================
//
// matrix input/output routines
//
//==============================================================================

//! load and save matrix from file (filename with absolute path)
void load_matrix ( TMatrix * &          M, 
                   const std::string &  absolute_filename ) ;
void save_matrix ( const TMatrix *      M, 
                   const std::string &  absolute_filename ) ;
                    
//==============================================================================
//
// Sylvester's law of inertia
//
//==============================================================================
    
                    
                    
//! Compute the number of eigenvalues of the symmetric matrix \a K which are smaller than \a alpha. 
size_t number_ev ( const TMatrix *   K,
                   const real        alpha,
                   const TTruncAcc & acc ) ;

//! Compute the number of eigenvalues of the symmetric matrix \a K which are in the half-open 
//! intervall [ \a alpha, \a beta ) where \a beta > \a alpha.
size_t number_ev ( const TMatrix *   K,
                   const real        alpha,
                   const real        beta,
                   const TTruncAcc & acc ) ;

//! Compute the number of eigenvalues of the general eigenvalue problem ( \a K, \a M ) which are smaller
//! than \a alpha. Here \a K and \a M are symmetric matrices and where \a M has to be positive definite.
size_t number_ev ( const TMatrix *   K,
                   const TMatrix *   M,
                   const real        alpha,
                   const TTruncAcc & acc ) ;

//! Compute the number of eigenvalues of the general eigenvalue problem ( \a K, \a M ) which are in the 
//! half-open intervall [ \a alpha, \a beta ). Here \a K and \a M are symmetric matrices and where \a M 
//! has to be positive definite and \a beta > \a alpha.
size_t number_ev ( const TMatrix *   K,
                   const TMatrix *   M,
                   const real        alpha,
                   const real        beta,
                   const TTruncAcc & acc ) ;
                   
                   
}// namespace EIGEN_MISC

}// namespace HAMLS

#endif  // __HAMLS_EIGEN_MISC_HH
