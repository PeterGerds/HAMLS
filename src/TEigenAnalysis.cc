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
// File        : TEigenAnalysis.cc
// Description : class for the analysis of eigensolutions 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include <iostream>

#include <tbb/parallel_for.h>

#include "algebra/mat_mul.hh"
#include "algebra/mat_norm.hh"

#include "hamls/TEigenAnalysis.hh"

namespace HAMLS
{

using std::unique_ptr;
using std::make_unique;
using std::scientific;
using std::fixed;
using std::cout;
using std::endl;
using std::vector;


/////////////////////////////////////////////////////////////////////////////
//
// local defines
//
/////////////////////////////////////////////////////////////////////////////

// macro that handles basic output 
#define OUT( msg )  LOG::print( msg )
#define HLINE       LOG::print("    -------------------------------------------------")

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
// TEigenAnalysis
//
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool
TEigenAnalysis::analyse_matrix_residual ( const TMatrix *  K,
                                          const TMatrix *  D,
                                          const TMatrix *  Z ) const
{
    if ( K == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenAnalysis) analyse_matrix_residual", "argument is nullptr" );

    auto  tic = Time::Wall::now();
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Compute the norm of K*Z - Z*D
    //
    ////////////////////////////////////////////////////////////////////////////

    unique_ptr< TDenseMatrix > KZ( new TDenseMatrix( K->row_is(), Z->col_is() ) );
    unique_ptr< TDenseMatrix > ZD( new TDenseMatrix( Z->row_is(), D->col_is() ) );
    
    TTruncAcc acc( _eps_rel );
    
    real abs_error =  real(0);
    real rel_error =  real(0);
    
    if ( D->rows() > 0 )
    {
        multiply( real(1), MATOP_NORM, K, MATOP_NORM, Z, real(0), KZ.get(), acc );        
        multiply( real(1), MATOP_NORM, Z, MATOP_NORM, D, real(0), ZD.get(), acc );
        
        const real temp_norm = norm_2( KZ.get() );
        
        abs_error = diffnorm_2( KZ.get() , ZD.get() );
        
        rel_error = abs_error / temp_norm;
    }// if
    
    auto  toc = Time::Wall::since( tic );
    
    if ( _verbosity >= 1 )
    {
        LOG::printf( " " );
        LOG::printf( "(TEigenAnalysis::analyse_matrix_residual) done in %.2fs", toc.seconds() );
        HLINE;
        LOG::printf( "    size of K = %d", K->rows() );
        LOG::printf( "    %d eigenpairs are given in D and Z", D->rows() );
        LOG::printf( "    abs_error := |K*Z-Z*D|_2       = %.4e", abs_error );
        LOG::printf( "    rel_error := abs_error/|K*Z|_2 = %.4e", rel_error );
        HLINE;
    }// if
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Return true if a too large error is detected
    //
    ////////////////////////////////////////////////////////////////////////////
    bool too_large_error_detected = false;
    
    if ( rel_error > _error_threshold )
        too_large_error_detected = true;
        
    return too_large_error_detected;
}
                    
                    
                                                 
                             
bool
TEigenAnalysis::analyse_matrix_residual( const TMatrix *  K,
                                         const TMatrix *  M,
                                         const TMatrix *  D,
                                         const TMatrix *  Z ) const
{
    if ( K == nullptr || M == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenAnalysis) analyse_matrix_residual", "argument is nullptr" );   
         
    auto  tic = Time::Wall::now();

    ////////////////////////////////////////////////////////////////////////////
    //
    // Compute the norm of K*Z - M*Z*D
    //
    ////////////////////////////////////////////////////////////////////////////

    auto  KZ  = make_unique< TDenseMatrix >( K->row_is(), Z->col_is() );
    auto  MZ  = make_unique< TDenseMatrix >( M->row_is(), Z->col_is() );
    auto  MZD = make_unique< TDenseMatrix >( M->row_is(), D->col_is() );
    
    TTruncAcc acc( _eps_rel );
    
    real abs_error =  real(0);
    real rel_error =  real(0);
    
    if ( D->rows() > 0 )
    {
        multiply( real(1), MATOP_NORM, K,        MATOP_NORM, Z, real(0), KZ.get(),  acc );        
        multiply( real(1), MATOP_NORM, M,        MATOP_NORM, Z, real(0), MZ.get(),  acc );
        multiply( real(1), MATOP_NORM, MZ.get(), MATOP_NORM, D, real(0), MZD.get(), acc );
        
        const real temp_norm = norm_2( KZ.get() );
        
        abs_error = diffnorm_2( KZ.get() , MZD.get() );
        
        rel_error = abs_error / temp_norm;
    }// if
    
    auto  toc = Time::Wall::since( tic );
    
    if ( _verbosity >= 1 )
    {
        LOG::printf( " " );
        LOG::printf( "(TEigenAnalysis::analyse_matrix_residual) done in %.2fs", toc.seconds() );
        HLINE;
        LOG::printf( "    size of (K,M) = %d", K->rows() );
        LOG::printf( "    %d eigenpairs are given in D and Z", D->rows() );
        LOG::printf( "    abs_error := |K*Z-M*Z*D|_2     = %.4e", abs_error );
        LOG::printf( "    rel_error := abs_error/|K*Z|_2 = %.4e", rel_error );
        HLINE;
    }// if
     
    ////////////////////////////////////////////////////////////////////////////
    //
    // Return true if a too large error is detected
    //
    ////////////////////////////////////////////////////////////////////////////
    
    bool too_large_error_detected = false;
    
    if ( rel_error > _error_threshold )
        too_large_error_detected = true;
        
    return too_large_error_detected;
}
                                    



bool
TEigenAnalysis::analyse_vector_residual ( const TMatrix *       K,
                                          const TDenseMatrix *  D,
                                          const TDenseMatrix *  Z,
                                          const TClusterTree *  ct ) const
{
    if ( K == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenAnalysis) analyse_vector_residual", "argument is nullptr" );
            
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine the residuals of the eigenvalues and eigenvectors contained
    // in (D,Z) according to the standard eigenvalue problem K*v=lambda*v
    //
    /////////////////////////////////////////////////////////////////////////////////////
    
    bool permutation_performed = false;
    
    if ( ct != nullptr )
        permutation_performed = true;    
    
    const size_t n_ev = Z->cols();
    
    vector < real >  abs_error( n_ev, real(0) );
    vector < real >  rel_error( n_ev, real(0) );
    vector < real >  lambda   ( n_ev, real(0) );
    
    auto comp_residual = 
        [ K, D, Z, ct, & abs_error, & rel_error, & lambda, & permutation_performed ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                unique_ptr< TVector >  K_v ( K->col_vector() );
                unique_ptr< TVector >  v   ( K->col_vector() );
                
                //----------------------------------------------------------------------
                // Get eigenvector 'v' corresponding to external numbering (if possible)
                //----------------------------------------------------------------------
                const TScalarVector  v_intern( Z->column( j ) );
                
                if ( permutation_performed )
                {   
                    const TPermutation * perm_intern_2_extern = ct->perm_i2e();
            
                    perm_intern_2_extern->permute( & v_intern, v.get() );
                }// if 
                else
                    v->assign( real(1), & v_intern );
                
                //------------------------------------------------
                // Compute the vector K_v = K*v
                //------------------------------------------------
                K->mul_vec( real(1), v.get(), real(0), K_v.get() );
                
                const real K_v_norm  = K_v->norm2(); 
                
                //------------------------------------------------
                // Compute the vector K_v = K*v - lambda*v
                //------------------------------------------------
                lambda[j] = D->entry( j,j );
                
                K_v->axpy( -lambda[j], v.get() );
                
                abs_error[j] = K_v->norm2();
                rel_error[j] = abs_error[j] / K_v_norm;
            }// for
        };
            
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute residuals
    //
    /////////////////////////////////////////////////////////////////////////////////////

    auto  tic = Time::Wall::now();

    if ( _do_parallel )
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_ev) ), comp_residual );
    else
        comp_residual( tbb::blocked_range< uint >( uint(0), uint(n_ev) ) );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine max/min residual
    //
    /////////////////////////////////////////////////////////////////////////////////////
    real   max_rel_error = real(0);
    real   max_abs_error = real(0);
    real   max_lambda    = real(0);
    size_t max_index     = 0;
    
    real   min_rel_error = real(10000000); //just make a initial guess...
    real   min_abs_error = real(0);
    real   min_lambda    = real(0);
    size_t min_index     = 0;
    
    for ( size_t j = 0; j < n_ev; j++ )
    {
        if ( rel_error[j] > max_rel_error )
        {
            max_rel_error = rel_error[j];
            max_abs_error = abs_error[j];
            max_lambda    = lambda   [j];
            max_index     = j;
        }// if
        
        if ( rel_error[j] < min_rel_error )
        {
            min_rel_error = rel_error[j];
            min_abs_error = abs_error[j];
            min_lambda    = lambda   [j];
            min_index     = j;
        }// if
    }// for

    auto  toc = Time::Wall::since( tic );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Output
    //
    /////////////////////////////////////////////////////////////////////////////////////
    
    if ( _verbosity >= 1 )
    {   
        LOG::printf( " " );
        LOG::printf( "(TEigenAnalysis::analyse_vector_residual) done in %.2fs", toc.seconds() );
        HLINE;
        LOG::printf( "    analyse eigenvector residuals according to EVP K*v=lambda*v" );
        HLINE;
        LOG::printf( "    permutation_performed = %d", int(permutation_performed) );
        LOG::printf( "    size of K             = %d", Z->rows() );
        LOG::printf( "    number of eigenpairs  = %d", Z->cols() );
        HLINE;
        LOG::printf( "    max. relative error: j=%d, abs_error=%.4e, rel_error=%.4e, lambda=%.4e",
                     max_index+1, max_abs_error, max_rel_error, max_lambda );
        LOG::printf( "    min. relative error: j=%d, abs_error=%.4e, rel_error=%.4e, lambda=%.4e",
                     min_index+1, min_abs_error, min_rel_error, min_lambda );
        HLINE;
        
        if ( _verbosity >= 2 )
        {
            for ( size_t j = 0; j < n_ev; j++ )
                LOG::printf( "    j=%d, abs_error=%.4e, rel_error=%.4e  (lambda = %.4e)",
                             j+1, abs_error[j], rel_error[j], lambda[j] );
                
            HLINE;
        }// if
    }// if
    
    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Return true if a too large error is detected
    //
    /////////////////////////////////////////////////////////////////////////////////////
    bool too_large_error_detected = false;
    
    if ( max_rel_error > _error_threshold )
        too_large_error_detected = true;
        
    return too_large_error_detected;
}
             
             
             
             



bool
TEigenAnalysis::analyse_vector_residual ( const TMatrix *       K,
                                          const TMatrix *       M,
                                          const TDenseMatrix *  D,
                                          const TDenseMatrix *  Z,
                                          const TClusterTree *  ct ) const
{
    if ( K == nullptr || M == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenAnalysis) analyse_vector_residual", "argument is nullptr" );
            
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine the residuals of the eigenvalues and eigenvectors contained
    // in (D,Z) according to the general eigenvalue problem K*v=lambda*M*v
    //
    /////////////////////////////////////////////////////////////////////////////////////
    bool permutation_performed = false;
    
    if ( ct != nullptr )
        permutation_performed = true;    
    
    const size_t n_ev = Z->cols();
    
    vector < real >  abs_error( n_ev, real(0) );
    vector < real >  rel_error( n_ev, real(0) );
    vector < real >  lambda   ( n_ev, real(0) );
    
    auto comp_residual = 
        [ K, M, D, Z, ct, & abs_error, & rel_error, & lambda, & permutation_performed ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                unique_ptr< TVector >  K_v ( K->col_vector() );
                unique_ptr< TVector >  M_v ( K->col_vector() );
                unique_ptr< TVector >  v   ( K->col_vector() );
                
                //----------------------------------------------------------------------
                // Get eigenvector 'v' corresponding to external numbering (if possible)
                //----------------------------------------------------------------------
                const TScalarVector  v_intern( Z->column( j ) );
                
                if ( permutation_performed )
                {   
                    const TPermutation * perm_intern_2_extern = ct->perm_i2e();
            
                    perm_intern_2_extern->permute( & v_intern, v.get() );
                }// if 
                else
                    v->assign( real(1), & v_intern );
                
                //------------------------------------------------
                // Compute the vector K_v = K*v and M_v = M*v
                //------------------------------------------------
                K->mul_vec( real(1), v.get(), real(0), K_v.get() );
                M->mul_vec( real(1), v.get(), real(0), M_v.get() );
                
                const real K_v_norm  = K_v->norm2(); 
                
                //------------------------------------------------
                // Compute the vector K_v = K*v - lambda*M*v
                //------------------------------------------------
                lambda[j] = D->entry( j,j );
                
                K_v->axpy( -lambda[j], M_v.get() );
                
                abs_error[j] = K_v->norm2();
                rel_error[j] = abs_error[j] / K_v_norm;
            }// for
        };
            
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute residuals
    //
    /////////////////////////////////////////////////////////////////////////////////////

    auto  tic = Time::Wall::now();
    
    if ( _do_parallel )
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_ev) ), comp_residual );
    else
        comp_residual( tbb::blocked_range< uint >( uint(0), uint(n_ev) ) );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine max/min residual
    //
    /////////////////////////////////////////////////////////////////////////////////////
    real   max_rel_error = real(0);
    real   max_abs_error = real(0);
    real   max_lambda    = real(0);
    size_t max_index     = 0;
    
    real   min_rel_error = real(10000000); //just make a initial guess...
    real   min_abs_error = real(0);
    real   min_lambda    = real(0);
    size_t min_index     = 0;
    
    for ( size_t j = 0; j < n_ev; j++ )
    {
        if ( rel_error[j] > max_rel_error )
        {
            max_rel_error = rel_error[j];
            max_abs_error = abs_error[j];
            max_lambda    = lambda   [j];
            max_index     = j;
        }// if
        
        if ( rel_error[j] < min_rel_error )
        {
            min_rel_error = rel_error[j];
            min_abs_error = abs_error[j];
            min_lambda    = lambda   [j];
            min_index     = j;
        }// if
    }// for
    
    auto  toc = Time::Wall::since( tic );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Output
    //
    /////////////////////////////////////////////////////////////////////////////////////
    
    if ( _verbosity >= 1 )
    {   
        LOG::printf( " " );
        LOG::printf( "(TEigenAnalysis::analyse_vector_residual) done in %.2fs", toc.seconds() );
        HLINE;
        LOG::printf( "    analyse eigenvector residuals according to EVP K·v=lambda·M·v" );
        HLINE;
        LOG::printf( "    permutation_performed = %d", int(permutation_performed) );
        LOG::printf( "    size of K             = %d", Z->rows() );
        LOG::printf( "    number of eigenpairs  = %d", Z->cols() );
        HLINE;
        LOG::printf( "    max. relative error: j=%d, abs_error=%.4e, rel_error=%.4e, lambda=%.4e",
                     max_index+1, max_abs_error, max_rel_error, max_lambda );
        LOG::printf( "    min. relative error: j=%d, abs_error=%.4e, rel_error=%.4e, lambda=%.4e",
                     min_index+1, min_abs_error, min_rel_error, min_lambda );
        HLINE;
        
        if ( _verbosity >= 2 )
        {
            for ( size_t j = 0; j < n_ev; j++ )
                LOG::printf( "    j=%d, abs_error=%.4e, rel_error=%.4e, lambda = %.4e",
                             j+1, abs_error[j], rel_error[j], lambda[j] );
                
            HLINE;
        }// if
    }// if
    
    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Return true if a too large error is detected
    //
    /////////////////////////////////////////////////////////////////////////////////////

    bool too_large_error_detected = false;
    
    if ( max_rel_error > _error_threshold )
        too_large_error_detected = true;
        
    return too_large_error_detected;
}
             
}// namespace
