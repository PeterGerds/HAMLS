#ifndef __HAMLS_TEIGENARNOLDI_HH
#define __HAMLS_TEIGENARNOLDI_HH

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
// File        : TEigenArnoldi.hh
// Description : eigensolver for H-matrices based on Arnoldi method
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
//! \class    TEigenBase
//! \brief    base class of eigensolvers for H-matrices which handles in particular the transformation of the EVP
//!
class TEigenBase
{
private:

    //!
    //! \struct  parameter_base_t
    //! \brief   Datatype to summarize the basic parameters
    //!
    struct parameter_base_t
    {      
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // basic parameters
        //
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //! if true it is tested that the matrix 'M' respectively the shifted problem is positive defininit
        bool test_pos_def;
                
        //! if true the residuals of the computed eigenpairs are tested and large errors are reported
        bool test_residuals;
        
        //! if true then print summarised information of program execution to the logfile
        bool  print_info;
        
        //! number of searched eigenpairs
        size_t n_ev_searched;
        
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_base_t ( )
        {            
            //--------------------------------
            // set the default basic parameter
            //--------------------------------
            print_info     = false;
            n_ev_searched  = 1;
            test_pos_def   = false;
            test_residuals = false;
        }
            
    };
    
    //!
    //! \struct  parameter_EVP_transformation_t
    //! \brief   Datatype to summarize the parameters for the EVP transformation
    //!
    struct parameter_EVP_transformation_t
    {              
        ///------------------------------------------------------------------------------------------------------
        /// TODO: Implement routines for standard eigenvalue problems
        ///------------------------------------------------------------------------------------------------------
    
    
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // parameter handling the transformation of the EVP
        //
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        
        //! If true the general eigenvalue problem (K,M) is IMPLICITELY transformed into a symmetric shifted 
        //! standard EVP otherwise it is IMPLICITELY transformed into a nonsymmetric shifted standard EVP.
        //! Note that K and M have to be symmetric matrices.
        //! =================================================================================================
        //!    symmetric: K*x = lambda*M*x  <==>  M_tilde*y = mu*y  with 
        //!                    
        //!               mu = 1/(lambda-shift), M_tilde = L^{-1}*M*L^{-T}, y = L^{T}*x, K-shift*M = L*L^{T}
        //!
        //!               NOTE: This transfomration works only if (K-shift*M) is positive definite
        //!
        //! nonsymmetric: K*x = lambda*M*x  <==>  C*x = mu*x  with 
        //!                    
        //!               mu = 1/(lambda-shift), C = (K-shift*M)^{-1} * M
        //!                 
        //!               NOTE: The matrix C is typically unsymmetric. However, if ARPACK is used as eigensolver
        //!                     this problem is acutally handled as a symmetric problem where symmetry is induced
        //!                     by the inner product <x,y>:=x^T*M*y. See ARPACK userguide for further information.
        //!
        //! =================================================================================================
        bool transform_symmetric;
        real shift;
        
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // parameter handling the accuracy of the EVP transformation
        //
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        
        //! relative H-matrix accuracy for the computation of the nearly exact H-matrix 
        //! operations, especially used for the almost exact factorisation of K-shift*M = L*L^{T}
        real eps_rel_exact;
        
        //! start value for the relative H-matrix accuracy of the computation of the preconditioner and
        //! final value of the relative H-matrix accurcy which has beenn finnally used for the compuation
        real eps_rel_precond_start;
        real eps_rel_precond_final;
        
        //! maximal number of tries which can be made to get a preconditioner which is good 
        //! enough and number of tries which have been acutal made to obtain this preconditioner
        size_t precond_max_try;
        size_t precond_count_try;
        
        //! error of the computed preconditioner, i.e., this is the value is the error of ‖ I - B A ‖_2 
        //! where A = K-shift*M and B = (K-shift*M)^{-1} is the preconditioner of A
        real   precond_error;
        
        
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_EVP_transformation_t ( )
        {            
            //-------------------------------------------------
            // set the default parameter for the transformation
            //-------------------------------------------------
            transform_symmetric   = true;
            shift                 = real(0);
            eps_rel_exact         = real(1e-10);
            eps_rel_precond_start = real(1e-3);
            
            //--------------------------------------
            // initialiase counters/auxiliary values
            //--------------------------------------
            eps_rel_precond_final = eps_rel_precond_start;
            precond_error         = real(100);
            precond_max_try       = 5;
            precond_count_try     = 0;
        }
    };
    
                
protected:


    /*
    /// TODO: Use sparse matrices for computation as well?
    
    //! pointer to the underlying cluster tree 
    const TClusterTree *  _ct;
    
    //! pointer to the matrix K (sparse matrix version)
    const TSparseMatrix * _K_sparse;
    
    //! pointer to the matrix M (sparse matrix version)
    const TSparseMatrix * _M_sparse;
    */



    //!
    //! data cotaining all the parameters relating to the EVP transformation
    //!
    parameter_EVP_transformation_t  _parameter_EVP_transformation;
    
    //!
    //! data cotaining all the basic parameters
    //!
    parameter_base_t                _parameter_base;
    
    
    //!--------------------------------------------------------------------------------------------------
    //! pointer to the different matrices involved during transformation and matrix vector multiplication
    //!--------------------------------------------------------------------------------------------------
    const TMatrix *  _K;
    const TMatrix *  _M;
    
    //! ---> only used if \a shift is not equal to zero
    TMatrix *  _K_minus_shift_M;
    
    //! ---> only used if \a transform_symmetric is true:  
    //!      pointer to factor L of the "exact" factorisation of K-shift*M = L*L^{T} 
    TMatrix *  _L;
    
    //! ---> only used if \a transform_symmetric is false:  
    //!      pointer to the approximated factorisation of K-shift*M
    TMatrix *  _K_minus_shift_M_factorised;
    
    //! ---> only used if \a transform_symmetric is false: 
    //!      pointer to preconditionier of (K-shift*M) which is derivated from _K_minus_shift_M_factorised
    TLinearOperator *  _K_minus_shift_M_precond;


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // auxialiarry routines
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

                                                                                                               
    //! Check the input consistency of the data
    void check_input_consistency ( const TMatrix *  K,
                                   const TMatrix *  M,
                                   TDenseMatrix  *  D,
                                   TDenseMatrix  *  Z ) ;
                                   
    //! Auxiliary routine which checks for trivial problems and solves it if the problem 
    //! is trivial. If a trivial problem is solved true is returned, otherwise false
    bool solve_trivial_problems ( const TMatrix *  K,
                                  const TMatrix *  M,
                                  TDenseMatrix  *  D,
                                  TDenseMatrix  *  Z ) const;
                                  
    //! Compute the matrix (K-shift*M) with predefined accuracy 'eps_rel_exact'
    void comp_shifted_matrix ( const TMatrix * K, 
                               const TMatrix * M, 
                               const real      shift, 
                               TMatrix * &     K_minus_shift_M,
                               const real      eps_rel_exact ) const;
                                
    //! Compute factorisation of K_minus_shift_M with predefined accuracy 'eps_rel'
    void factorise ( const TMatrix *  K_minus_shift_M, 
                     TMatrix * &      L,
                     const real       eps_rel_exact ) ;
                     
    //! Compute a preconditioner of K_minus_shift_M with predefined starting accuracy 'eps_rel_precond_start'
    //! (Note: existence of K_minus_shift_M_precond is coupled to existence of K_minus_shift_M_factorised)
    void comp_precond ( const TMatrix *      K_minus_shift_M,
                        TMatrix * &          K_minus_shift_M_factorised,
                        TLinearOperator * &  K_minus_shift_M_precond,
                        const real           eps_rel_precond_start ) ;
                                  
                                                                     
    //! Transform the generalized eigenvalue problem (K,M) to implictely a standard eigenvalue problem
    void transform_problem ( const TMatrix *  K,
                             const TMatrix *  M ) ;
                             
    //! Can only be used if \a transform_symmetric is true: Transform the eigenvectors of the 
    //! standard eigenvalue problem back to eigenvectors of the original generalized problem.
    //! On input \a Z_standard contains the eigenvectors of the standard EVP and when function
    //! is finished it contains the eigenvectors of the generalized EVP.
    void backtransform_eigenvectors_sym ( const TMatrix * L,
                                          TDenseMatrix *  Z_standard ) const;
                                          
    //! Compute: v <== L^{-1} * M * L^{-T} * v 
    void iterate_sym ( TVector * v ) const;
    
    //! Compute: v <== (K-shift*M)^{-1} * M * v   
    void iterate_nonsym ( TVector * v ) const;
    
    //! Compute: v <== (K-shift*M)^{-1} * v 
    void iterate_nonsym_partial ( TVector * v ) const;

public:
   
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // constructor and destructor
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //! constructor 
    TEigenBase () 
    {
        _K                          = nullptr;
        _M                          = nullptr;
        
        _K_minus_shift_M            = nullptr;
        _L                          = nullptr;
        _K_minus_shift_M_factorised = nullptr;
        _K_minus_shift_M_precond    = nullptr;
    }
    
    //! dtor
    virtual ~TEigenBase ()
    {    
        if ( _K_minus_shift_M            != nullptr ) { delete _K_minus_shift_M;            _K_minus_shift_M = nullptr; }
        if ( _L                          != nullptr ) { delete _L;                          _L = nullptr; }
        if ( _K_minus_shift_M_factorised != nullptr ) { delete _K_minus_shift_M_factorised; _K_minus_shift_M_factorised = nullptr; }
        if ( _K_minus_shift_M_precond    != nullptr ) { delete _K_minus_shift_M_precond;    _K_minus_shift_M_precond    = nullptr; }
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // access local variables
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //------------------------------------------------------------------------------------------------------
    // basic parameter
    //------------------------------------------------------------------------------------------------------
    bool   get_print_info () const { return _parameter_base.print_info; }
    void   set_print_info ( const bool print_info ) { _parameter_base.print_info = print_info; }
    
    bool   get_test_pos_def () const { return _parameter_base.test_pos_def; }
    void   set_test_pos_def ( const bool test_pos_def ) { _parameter_base.test_pos_def = test_pos_def; }
    
    bool   get_test_residuals () const { return _parameter_base.test_residuals; }
    void   set_test_residuals ( const bool test_residuals ) { _parameter_base.test_residuals = test_residuals; }
    
    size_t get_n_ev_searched () const { return _parameter_base.n_ev_searched; }
    void   set_n_ev_searched ( const size_t n_ev_searched ) { _parameter_base.n_ev_searched = n_ev_searched; }
    
    //------------------------------------------------------------------------------------------------------
    // parameter eigenvalueproblem transformation
    //------------------------------------------------------------------------------------------------------
    bool   get_transform_symmetric () const { return _parameter_EVP_transformation.transform_symmetric; }
    void   set_transform_symmetric ( const bool transform_symmetric ) { _parameter_EVP_transformation.transform_symmetric = transform_symmetric; } 
    
    real   get_shift () const { return _parameter_EVP_transformation.shift; }
    void   set_shift ( const real shift ) { _parameter_EVP_transformation.shift = shift; }
    
    real   get_eps_rel_exact () const { return _parameter_EVP_transformation.eps_rel_exact; }
    void   set_eps_rel_exact ( const real eps_rel_exact ) { _parameter_EVP_transformation.eps_rel_exact = eps_rel_exact; }
    
    real   get_eps_rel_precond_start () const { return _parameter_EVP_transformation.eps_rel_precond_start; }
    void   set_eps_rel_precond_start ( const real eps_rel_precond_start ) { _parameter_EVP_transformation.eps_rel_precond_start = eps_rel_precond_start; }
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // miscellaneous methods 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                                       
    //! Solving the generalized eigenvalue problem K*Z = M*Z*D where Z^{T}*M*Z = Id (approximatively)
    //! Input:  K and M are symmetric matrices and M is positive definite
    //! Output: D is a diagonal matrix containing the selected eigenvalues in 
    //!         ascending order. The matrix Z contains the corresponding eigenvectors.
    //!         The return value of the function is the number of computed eigenvalues.
    virtual size_t comp_decomp ( const TMatrix *      K,
                                 const TMatrix *      M,
                                 TDenseMatrix  *      D,
                                 TDenseMatrix  *      Z ) = 0;   
};        






//!
//! \ingroup  HAMLS_Module
//! \class    TEigenArnoldi
//! \brief    eigensolver for H-matrices based on Arnoldi method
//!
class TEigenArnoldi : public TEigenBase
{
private:
    ///------------------------------------------------------------------------------------------------------
    /// TODO: This eigensolver is only a basic implementation of the Arnoldi Algorithm and works 
    ///       quite well. Possibly better accuracy of the computed eigenpair approximations could
    ///       be obtained when instead of the normal dot-product implementation
    ///       the stable-version of the dot-product implementation is used. 
    ///------------------------------------------------------------------------------------------------------
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Different re-orthogonalisation strategies while computing Arnoldi basis, cf. [Templates for Solution of Algebraic EVPs]
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum re_ortho_strategy_t
    { 
        FULL_ORTHO          = 0,  // apply full orthogonalisation
        FULL_ORTHO_PARALLEL = 1,  // apply full orthogonalisation in parallel
        SELECTIVE_ORTHO     = 2,  // apply selective orthogonalisation, cf. [Templates for Solution of Algebraic EVPs]
        PARTIAL_ORTHO       = 3,  // instable orhtogonalisation strategy
        MIXED_ORTHO         = 4   // instable orhtogonalisation strategy
    };  
        
        
    //!
    //! \struct  parameter_arnoldi_t
    //! \brief   Datatype to summarize the parameters
    //!
    struct parameter_arnoldi_t
    {      
        /////////////////////////////////////////////////////////
        //
        // parameter and auxiliary data
        //
        /////////////////////////////////////////////////////////
        
        //! Upper bound of the relative error of an eigenpair approximation (Note: This parameter shouldn't 
        //! be choosen to large. If the error bound is choosen too large it can be happen, that bad eigenpair 
        //! approximations are computed. The problem is that then all other iteration vectors are orthogonalized
        //! with this bad approximation and convergency could be destroyed)
        real    rel_error_bound;
        
        //! seed/start value for initialising random values
        size_t  seed;
        
        //! auxiliary parameter counting the iterations
        size_t  count_iterate;
        
        //! if 'n_ev' eigenvalues are searched, the maximal size of the computed Arnoldi basis is 'n_ev * factor_basis_size'
        size_t  factor_basis_size;
        
        //! the minimal value for the maximal size of the computed Arnoldi basis
        size_t  min_basis_size;
        
        //! actual size of the computed Arnoldi basis
        size_t  basis_size;
        
        //! strategy for reorthogonalisation of the iteration vectors of Arnoldi basis
        re_ortho_strategy_t re_ortho_strategy;
        
        //! maximum number of vectors where reorthogonalisation is applied serial and not parallel
        size_t  size_serial_ortho;
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_arnoldi_t ( )
        {   
            //--------------------------------         
            // set the default parameter
            //--------------------------------
            rel_error_bound   = real(1e-8);
            factor_basis_size = 4;
            min_basis_size    = 40;
            
            ///NOTE: If this value 'size_serial_ortho' is chosen too small the orthoganlity of the basis vectors 
            ///      deteriorates a little bit when the parallel implementation is used. This is the same effect 
            ///      as when normal and not modified Gram-Schmidt is used. Benchmarks have shown that a value larger 
            ///      or equal to 10 is fine. Best results are obtained, however, if sequential implementation is used
            ///      (re_ortho_strategy is set to FULL_ORTHO)
            size_serial_ortho = 20; 
            
            // Adjust re-orthogonalisation strategy to number of threads
            if ( CFG::nthreads() == 1 || true )
                re_ortho_strategy = FULL_ORTHO;
            else
                re_ortho_strategy = FULL_ORTHO_PARALLEL;
            
            //--------------------------------         
            // initialise auxiliary data
            //--------------------------------
            basis_size    = 0;
            seed          = 0;
            count_iterate = 0;
        }
    };
    
    //!
    //! data cotaining all the parameters
    //!
    parameter_arnoldi_t  _parameter_arnoldi;
        
    
protected:

    //! Compute y = L^{-1} * M * L^{-T} * v
    void iterate_arnoldi ( const TVector * v, 
                           TVector *       y ) ;
                           
    //! Test routine which reorthogonalizes Q and updates A_hat
    void debug_update_Q_and_A_hat ( TDenseMatrix * A_hat, 
                                    TDenseMatrix * Q ) ;
                           
    //! auxialiarry routine for "comp_arnoldi_basis" applying different reorthogonalisation strategies
    void comp_arnoldi_basis_ortho_and_update ( const std::vector < TVector * > & q,
                                               const size_t                      m,
                                               TVector *                         q_tilde,
                                               TDenseMatrix *                    A_hat_temp ) const;
                           
    //! Determine the Arnoldi basis of the vector 'v' and the matrix A:= L^{-1}*M*L^{-T}.
    //! The Arnoldi basis is contained in 'Q' and the projected EVP in 'A_hat:=Q^{T}*A*Q'. 
    //! The parameter 'rel_error_bound' controls the relative accuracy of the associated subspace
    real   comp_arnoldi_basis (  const size_t     m_max,
                                 const TVector *  v, 
                                 TDenseMatrix *   A_hat, 
                                 TDenseMatrix *   Q, 
                                 const real       rel_error_bound ) ;
                       
    //! Auxiliary function of 'comp_decomp'
    size_t comp_decomp_arnoldi ( const size_t  n_ev_searched, 
                                 TMatrix * &   D_standard,
                                 TMatrix * &   Z_standard ) ;
                                 

    //! Can only be used if \a transform_symmetric is true: 
    //! Transform the eigenpairs of the standard EVP back to eigenpairs of the original generalized EVP.
    //! \a D_standard and \a Z_standard contains the eigenvalues respectively eigenvectors of the 
    //! standard problem and \a D and \a Z the data of the generalized EVP.
    void backtransform_eigensolutions_arnoldi ( const TMatrix *  L,
                                                const TMatrix *  D_standard,
                                                const TMatrix *  Z_standard,
                                                TDenseMatrix *   D,
                                                TDenseMatrix *   Z ) const;
                                                
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
    TEigenArnoldi () {}
    
    //! dtor
    virtual ~TEigenArnoldi () {}
    
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

#endif  // __HAMLS_TEIGENARNOLDI_HH
