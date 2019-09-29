#ifndef __HAMLS_TEIGENARPACK_HH
#define __HAMLS_TEIGENARPACK_HH

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
// File        : TEigenArpack.hh
// Description : eigensolver for H-matrices based on ARPACK library 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "matrix/TMatrix.hh"
#include "matrix/TDenseMatrix.hh"
#include "matrix/THMatrix.hh"
#include "algebra/mat_norm.hh"
#include "algebra/mat_fac.hh"
#include "algebra/mat_mul.hh"
#include "algebra/solve_tri.hh"
#include "solver/TSolver.hh"
#include "solver/TAutoSolver.hh"
#include "vector/TScalarVector.hh"

#include "hamls/TEigenAnalysis.hh"
#include "hamls/TEigenArnoldi.hh"
#include "hamls/TEigenLapack.hh"
#include "hamls/eigen_misc.hh"
#include "hamls/arpack.hh"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tbb.h"



namespace HAMLS
{

using namespace HLIB;

//!
//! \ingroup  HAMLS_Module
//! \class    TEigenArpack
//! \brief    eigensolver for H-matrices based on the the ARPACK library
//!
class TEigenArpack : public TEigenBase
{
private:

    //!
    //! \struct  parameter_arpack_t
    //! \brief   Datatype to summarize the parameters 
    //!
    struct parameter_arpack_t
    {      
        /////////////////////////////////////////////////////////
        //
        // parameter and auxiliary data
        //
        /////////////////////////////////////////////////////////
        
        //! stopping criteria
        //! ==================================================================================================
        //!     the relative accuracy of the Ritz value 'lambda[i]' is considered as acceptable if 
        //! 
        //!                     bound[i] <= tol * |lambda[i]|
        //!
        //!     If 'tol<=0' than in ARPACK the tolerance is set automatically to machine precission 
        //!     (i.e., intern LAPACK is used to compute this machine precission)
        //! ==================================================================================================
        real stopping_tol;        
        
        //! relevant ARPACK counters
        int iparam_3;
        int iparam_5;
        int iparam_9;
        int iparam_10;
        int iparam_11;
        int info_seupd;
        int info_saupd;
        
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_arpack_t ( )
        {            
            //--------------------------------         
            // set the default parameter
            //--------------------------------
            stopping_tol = real(1e-8);
            
            //--------------------------------         
            // initialise auxiliary data
            //--------------------------------    
            iparam_3     = -1;
            iparam_5     = -1;
            iparam_9     = -1;
            iparam_10    = -1;
            iparam_11    = -1;
            info_seupd   = -1;
            info_saupd   = -1;
        }
    };
    
    
    //!
    //! data cotaining all the parameters
    //!
    parameter_arpack_t  _parameter_arpack;
        
protected:

    //! transform real array into a vector consistent to the underlying system
    TScalarVector * get_vector_from_array ( const real * in,
                                            const size_t n ) const;
                         
    //! apply ARPACK operator B: 'in' is input array of size 'n' and 'out' is corresponding output array
    void apply_B ( const real * in, 
                   real *       out,
                   const size_t n ) const; 
                   
    //! apply ARPACK operator OP: 'in' is input array of size 'n' and 'out' is corresponding output array
    void apply_OP ( const real * in, 
                   real *       out,
                   const size_t n ) const; 
                   
    //! apply ARPACK operator OP: 'in' is input array of size 'n' and 'out' is corresponding output array
    void apply_OP_partial ( const real * in, 
                            real *       out,
                            const size_t n ) const;
                              
                                    
    //! Print summary of the finished method 
    //! ( 'D' is the matrix containing the approximated eigenvalues)
    void print_summary ( const TMatrix * D_approx ) const;
             
    //! Print the options used for the eigensolver
    void print_options () const;
                                                                                                       
public:
   
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // constructor and destructor
    //
    //////////////////////////////////////////////////////////////////////////////////////

    //! constructor 
    TEigenArpack () {}
    
    //! dtor
    virtual ~TEigenArpack () {}
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // access local variables
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    real   get_stopping_tol () const { return _parameter_arpack.stopping_tol; }
    void   set_stopping_tol ( const real stopping_tol ) { _parameter_arpack.stopping_tol = stopping_tol; }
               
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // miscellaneous methods 
    //
    //////////////////////////////////////////////////////////////////////////////////////
    

    //! Solving the generalized eigenvalue problem K*Z = M*Z*D where Z^{T}*M*Z = Id. 
    //! Input:  K and M are symmetric matrices and M is positive definite
    //! Output: D is a diagonal matrix containing the selected eigenvalues in 
    //!         ascending order. The matrix Z contains the corresponding eigenvectors.
    //!         The return value of the function is the number of computed eigenvalues.
    size_t comp_decomp ( const TMatrix *      K,
                         const TMatrix *      M,
                         TDenseMatrix  *      D,
                         TDenseMatrix  *      Z ) ;
   
};        

}// namespace HAMLS

#endif  // __HAMLS_TEIGENARPACK_HH
