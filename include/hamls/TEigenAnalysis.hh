#ifndef __HAMLS_TEIGENANALYSIS_HH
#define __HAMLS_TEIGENANALYSIS_HH

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
// File        : TEigenAnalysis.hh
// Description : class for the analysis of eigensolutions
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "cluster/TClusterTree.hh"

#include "matrix/TMatrix.hh"
#include "matrix/TDenseMatrix.hh"
#include "matrix/TSparseMatrix.hh"

#include "hamls/eigen_misc.hh"

namespace HAMLS
{

using namespace HLIB;

//!
//! \ingroup  HAMLS_Module
//! \class    TEigenAnalysis
//! \brief    class for the analysis of eigensolutions
//!
class TEigenAnalysis
{
private:

    //! @cond
    
    //! relative H-Matrix accuracy used during computations
    real _eps_rel;
        
    //! error threshold: errors larger than this error are considered as too large
    real _error_threshold;
        
    //! verbosity of the output printed to the logfile regarding the measured errors: 0=nothing,1=medium,2=detail
    int  _verbosity;
    
    //! apply computations in parallel if possible
    bool _do_parallel;

    //! @endcond

public:
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // constructor and destructor
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    //! constructor
    TEigenAnalysis () 
    {
        _eps_rel          = real(1e-8);
        _error_threshold  = real(1e-7);
        _verbosity        = 2;
        
        if ( CFG::nthreads() == 1 )
            _do_parallel = false;
        else
            _do_parallel = true;
    }
        
    //! dtor
    virtual ~TEigenAnalysis () {}
        
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // access local variables
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
    //
    int    get_verbosity () const { return _verbosity; }
    void   set_verbosity ( const int verbosity ) { _verbosity = verbosity; }
    
    real   get_error_threshold () const { return _error_threshold; }
    void   set_error_threshold ( const real error_threshold ) { _error_threshold = error_threshold; }
    
    real   get_eps_rel () const { return _eps_rel; }
    void   set_eps_rel ( const real eps_rel_exact ) { _eps_rel = eps_rel_exact; }
    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // miscellaneous methods 
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////
                               
                           
    //!=====================================================================================
    //!
    //! Analyse the eigensolution (D,Z) of the standard eigenvalue problem K*x=lambda*x
    //! respectively of the generalized eigenvalue problem K*x=lambda*M*x where \a D is
    //! a diagonal matrix containing the eigenvalues and \a Z the matrix containing the 
    //! corresponding eigenvectors.                   
    //!
    //!=====================================================================================
                 
    //! Analyse the matrix residual, return \a true if large errrors were detected
    bool analyse_matrix_residual ( const TMatrix * K,
                                   const TMatrix * D,
                                   const TMatrix * Z ) const;
                           
    //! Analyse the matrix residual, return \a true if large errrors were detected
    bool analyse_matrix_residual ( const TMatrix * K,
                                   const TMatrix * M,
                                   const TMatrix * D,
                                   const TMatrix * Z ) const;
                                                                                   
    //! Analyse the matrix residual, return \a true if large errrors were detected
    //! Note: If ct != nullptr then the column vectors contained in Z are permutated 
    //! according to the perumtation given in clustertree ct
    bool analyse_vector_residual ( const TMatrix *       K,
                                   const TDenseMatrix *  D,
                                   const TDenseMatrix *  Z,
                                   const TClusterTree *  ct = nullptr ) const;
                                   
    //! Analyse the matrix residual, return \a true if large errrors were detected
    //! Note: If ct != nullptr then the column vectors contained in Z are permutated 
    //! according to the perumtation given in clustertree ct
    bool analyse_vector_residual ( const TMatrix *       K,
                                   const TMatrix *       M,
                                   const TDenseMatrix *  D,
                                   const TDenseMatrix *  Z,
                                   const TClusterTree *  ct = nullptr ) const;
                   
};    
                
}// namespace HAMLS

#endif  // __HAMLS_TEIGENANALYSIS_HH
