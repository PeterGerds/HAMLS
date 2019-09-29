#ifndef __HLIB_TEIGENLAPACK_HH
#define __HLIB_TEIGENLAPACK_HH

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
// File        : TEigenLapack.hh
// Description : class for the eigen decomposition of symmetric eigenvalue problems using LAPACK
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "blas/Algebra.hh"
#include "blas/Algebra.hh"
#include "blas/Matrix.hh"

#include "base/types.hh"

#include "matrix/TMatrix.hh"
#include "matrix/TDenseMatrix.hh"
#include "matrix/TBlockMatrix.hh"

#include "matrix/THMatrix.hh"
#include "algebra/mat_norm.hh"
#include "algebra/mat_inv.hh"
#include "algebra/mat_fac.hh"
#include "algebra/mat_mul.hh"
#include "algebra/mat_add.hh"

#include "hamls/eigen_misc.hh"
#include "hamls/TEigenAnalysis.hh"


namespace HAMLS
{

    
/********************************
* types of eigenvalue selection *
********************************/   

typedef enum { EV_SEL_FULL,  /* compute all eigenvalues */
               EV_SEL_INDEX, /* compute all eigenvalues corresponding to an given index set */
               EV_SEL_BOUND  /* compute all eigenvalues within an given intervall */                       
} ev_sel_t;  
    
 
//!
//! \ingroup  AMLS_Module
//! \class    TEigenLapack
//! \brief    class for the eigen decomposition of symmetric eigenvalue problems using LAPACK
//!
class TEigenLapack
{
private:

    //! If true, then it is tested if the input matrices K respectively K, M are symmetric. 
    //! In the case of the generalized eigenvalue problem (K,M) it is tested as well 
    //! that the matrix M is positive defininit.
    bool      _test_pos_def;
    
    //! if true the residuals of the computed eigenpairs are tested and large errors are reported
    bool      _test_residuals;

    //! type of eigenvalue selection
    ev_sel_t  _ev_selection;
 
     //! lower bound of the range of interest of the wanted eigenvalues
    real      _lbound;
 
    //! upper bound of the range of interest of the wanted eigenvalues
    real      _ubound;
        
    //! upper index of the eigenvalues wanted 
    idx_t     _uindex;
    
    //! lower lower of the eigenvalues wanted
    idx_t     _lindex; 
    
public:

    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // constructor and destructor
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    //! constructor of the class to compute all eigenvalues/eigenvectors
    TEigenLapack () 
    {
        _test_pos_def   = false;
        _test_residuals = false;
        _ev_selection   = EV_SEL_FULL;
        _lbound         = real(0);
        _ubound         = real(0);
        _lindex         = 0;
        _uindex         = 0;
    }
    
    //! constructor of the class to compute all eigenvalues/eigenvectors where  
    //! the eigenvalues are in the half-open interval ( \a _lbound, \a _ubound ]
    TEigenLapack ( const real lbound,
                    const real ubound ) 
    {
        _test_pos_def   = false;
        _test_residuals = false;
        _ev_selection   = EV_SEL_BOUND;
        _lbound         = lbound;
        _ubound         = ubound;
        _lindex         = 0;
        _uindex         = 0;
    }
        
    //! constructor of the class to compute the \a lindex -th through  
    //! the \a uindex -th  eigenvalue/eigenvector where the eigenvalues  
    //! are numbered from '1' to 'n' beginning with the smallest one
    TEigenLapack ( const idx_t lindex, 
                    const idx_t uindex ) 
    {
        _test_pos_def   = false;
        _test_residuals = false;
        _ev_selection   = EV_SEL_INDEX;
        _lbound         = real(0);
        _ubound         = real(0);
        _lindex         = lindex;
        _uindex         = uindex;
    }

    //! dtor
    virtual ~TEigenLapack () {}
    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // access local variables
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    real get_ubound () const { return _ubound; }
    void set_ubound ( const real ubound ) { _ubound = ubound; }
    
    real get_lbound () const { return _lbound; }
    void set_lbound ( const real lbound ) { _lbound = lbound; }
 
    idx_t get_uindex () const { return _uindex; }
    void  set_uindex ( const real uindex ) { _uindex = uindex; }
    
    idx_t get_lindex () const { return _lindex; }
    void  set_lindex ( const real lindex ) { _lindex = lindex; }
    
    ev_sel_t get_ev_selection () const { return _ev_selection; }
    void     set_ev_selection ( const ev_sel_t ev_selection ) { _ev_selection = ev_selection; }
    
    bool get_test_pos_def () const { return _test_pos_def; }
    void set_test_pos_def ( const bool test_pos_def ) { _test_pos_def = test_pos_def; } 
    
    bool get_test_residuals () const { return _test_residuals; }
    void set_test_residuals ( const bool test_residuals ) { _test_residuals = test_residuals; } 
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // miscellaneous methods 
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
        
    //! Solving the generalized eigenvalue problem K*Z = M*Z*D where Z^{T}*M*Z = Id. 
    //! Input:  K and M are symmetric matrices and M is positive definite
    //! Output: D is a diagonal matrix containing the selected eigenvalues in 
    //!         ascending order. The matrix Z contains the corresponding eigenvectors.
    //!         The return value of the function is the number of computed eigenvalues.
    size_t comp_decomp ( const TMatrix * K,
                         const TMatrix * M,
                         TDenseMatrix  * D,
                         TDenseMatrix  * Z ) const;
                      
                      
    //! Solving the standard eigenvalue problem K*Z = Z*D where Z^{T}*Z = Id. 
    //! Input:  K is symmetric matrix
    //! Output: D is a diagonal matrix containing the selected eigenvalues in 
    //!         ascending order. The Z matrix contains the corresponding eigenvectors.
    //!         The return value of the function is the number of computed eigenvalues.
    size_t comp_decomp ( const TMatrix * K,
                         TDenseMatrix  * D,
                         TDenseMatrix  * Z ) const;                               
};

}// namespace

#endif  // __HLIB_TEIGENLAPACK_HH
