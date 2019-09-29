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
// File        : eigen_misc.cc
// Description : auxilliary routines for the eigen decomposition routines contained 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include <iostream>

#include <tbb/parallel_for.h>

#include "matrix/structure.hh"
#include "matrix/TBlockMatrix.hh"
#include "algebra/mat_add.hh"
#include "algebra/mat_fac.hh"

#include "hamls/eigen_misc.hh"

namespace HAMLS
{

using std::unique_ptr;
using std::cout;
using std::endl;
using std::string;
using std::vector;


/////////////////////////////////////////////////////////////////////////////
//
// local defines
//
/////////////////////////////////////////////////////////////////////////////

// macro that handles basic output 
#define OUT( msg )  LOG::print( msg )
#define HLINE       LOG::print("    -------------------------------------------------")

// define to print information during computation
//   0 : do nothing
//   1 : do timings and print info at the end of computation
//   2 : print info messages during computation
#define DO_TEST   0

// timing macros
#if DO_TEST >= 1
#  define  TIC            auto  tic = Time::Thread::now()
#  define  TOC            auto  toc = Time::Thread::since( tic )
#  define  TICC( timer )           auto  (timer) = Time::Thread::now()
#  define  TOCC( timer, result )   (result) += double( Time::Thread::since( (timer) ) )
#else
#  define  TIC
#  define  TOC
#  define  TICC( timer )
#  define  TOCC( timer, result )
#endif

// define to print detailed informations during computations
#if DO_TEST >= 2
#  define LOG( msg )  LOG::print( msg )
#  define LOGHLINE    LOG::print("    -------------------------------------------------")
#else
#  define LOG( msg )   
#  define LOGHLINE
#endif


namespace EIGEN_MISC
{

namespace 
{
    
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// local defines and constants
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
    
//
// values smaller than this are considered as zero
//
const real ACCEPTED_AS_ZERO = real(1e-8);

//
// relative H-matrix accuracy which is seen as nearly exact
//
const real EPS_REL_EXACT    = real(1e-10);
    
    
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//
// local functions
//
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


                      
TMatrix *
get_halve_matrix_local( const TMatrix * M ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_halve_matrix_local", "argument is nullptr" );
             
             
    if ( is_blocked( M ) )
    {
        auto  M_b = cptrcast( M, TBlockMatrix ); 
        
        TBlockMatrix * root = new TBlockMatrix;
        
        //
        // Copy matrix structure of the blockmatrix 
        // but set the new matrix to symmetric
        //
        root->copy_struct_from( M_b );
        root->set_nonsym();
        
        if ( M_b->block_rows() != M_b->block_cols() )
            HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_halve_matrix_local", "blockrows different to blockcols" ); 
     
        //
        // Copy only blocks of the lower triangular matrix
        //
        for ( uint i = 0; i < M_b->block_rows(); i++ )
        {
            for ( uint j = 0; j < M_b->block_cols(); j++ )
            {
                root->set_block( i, j, nullptr );
                
                if ( i > j )
                {        
                    if ( M_b->block(i,j) != nullptr )
                    {
                        unique_ptr< TMatrix >  M_ij( M_b->block(i,j)->copy() );
                        
                        root->set_block( i, j, M_ij.release() );
                    }// if
                }// if
                else if ( i == j )
                {
                    root->set_block( i, j, get_halve_matrix_local( M_b->block(i,j) ) );
                }// else if
                else 
                {
                    if ( M_b->block(i,j) != nullptr )
                        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_halve_matrix", "" ); 
                
                    root->set_block( i, j, nullptr );
                }// else
            }// for
        }// for
        
        return root;
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = ptrcast( M->copy().release(), TDenseMatrix ); 
        
        for ( size_t j = 0; j < DM->cols(); j++ )
        {
            for ( size_t i = 0; i < DM->rows(); i++ )
            {
                if ( i > j )
                {
                    //
                    // do nothing
                    //
                }// if
                else if ( i == j )
                {
                    //
                    // halve diagonal
                    //
                    const real entry = DM->entry( i, j );
            
                    DM->set_entry( i, j, entry/real(2) );
                }// else if
                else
                {
                    //
                    // set entry to zero
                    //
                    DM->set_entry( i, j, real(0) );
                }// else
            }// for
        }// for
        
        //
        // set matrix nonsymmetic
        //
        DM->set_nonsym();
        
        return DM;
    }// else if
    else 
    {
        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_halve_matrix_local", "unexpected matrix type" ); 
    }// else
}

TMatrix *
get_nonsym_of_sym_local ( const TMatrix * M,
                          const int       max_id_plus_1 ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_nonsym_of_sym_local", "argument is nullptr" );
            
    if ( is_blocked( M ) )
    {
        auto  M_b = cptrcast( M, TBlockMatrix ); 
        
        TBlockMatrix * root = new TBlockMatrix;
        
        //
        // Copy matrix structure of the blockmatrix 
        // but set the new matrix to nonsymmetric
        //
        root->copy_struct_from( M_b );
        root->set_nonsym();
        
        if ( M_b->block_rows() != M_b->block_cols() )
            HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_nonsym_of_sym_local", "blockrows different to blockcols" ); 
     
        //
        // Copy the lower block triangular part of the blockmatrix to the
        // the lower and also to the upper triangular part of the new matrix
        //
        for ( uint i = 0; i < M_b->block_rows(); i++ )
        {
            for ( uint j = 0; j <= i; j++ )
            {
                root->set_block( i, j, nullptr );
                root->set_block( j, i, nullptr );
            
                if ( i != j )
                {                        
                    if ( M_b->block(i,j) != nullptr )
                    {
                        //
                        // set M_ij
                        //
                        unique_ptr< TMatrix >  M_ij( M_b->block(i,j)->copy() );
                    
                        root->set_block( i, j, M_ij.release() );
                        
                                                
                        //
                        // set M_ji = (M_ij)^{T}
                        //
                        unique_ptr< TMatrix >  M_ji( M_b->block(i,j)->copy() );
                        
                        M_ji->transpose();
                    
                        // ensure unique ID in matrix (not globally unique!)
                        M_ji->set_id( max_id_plus_1 + M_ji->id() );
                        
                        
                        root->set_block( j, i, M_ji.release() );
                    }// if
                }// if
                else
                {
                    root->set_block( i, j, get_nonsym_of_sym_local( M_b->block(i,j), max_id_plus_1 ) );
                }// else
            }// for
        }// for
        
        return root;
    }// if
    else
    {
        unique_ptr< TMatrix > M_copy( M->copy() );
        
        M_copy->set_nonsym();
        
        return M_copy.release();
    }// else
}   
     



TMatrix *
get_full_format_of_lower_triangular_local ( const TMatrix * M,
                                            const int       max_id_plus_1 ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_full_format_of_lower_triangular_local", "argument is nullptr" );
        
    if ( is_blocked( M ) )
    {
        auto  M_b = cptrcast( M, TBlockMatrix ); 
        
        TBlockMatrix * root = new TBlockMatrix;
        
        //
        // Copy matrix structure of the blockmatrix 
        // but set the new matrix to nonsymmetric
        //
        root->copy_struct_from( M_b );
        root->set_nonsym();
        
        if ( M_b->block_rows() != M_b->block_cols() )
            HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_full_format_of_lower_triangular_local", "blockrows different to blockcols" ); 
     
        //
        // Copy the lower block triangular part of the blockmatrix to the
        // the lower and also to the upper triangular part of the new matrix
        //
        for ( uint i = 0; i < M_b->block_rows(); i++ )
        {
            for ( uint j = 0; j <= i; j++ )
            {
                root->set_block( i, j, nullptr );
                root->set_block( j, i, nullptr );
            
                if ( i != j )
                {                        
                    if ( M_b->block(i,j) != nullptr )
                    {
                        //
                        // set M_ij
                        //
                        unique_ptr< TMatrix >  M_ij( M_b->block(i,j)->copy() );
                    
                        root->set_block( i, j, M_ij.release() );
                        
                                                
                        //
                        // set entries of M_ji to Zero
                        //
                        unique_ptr< TMatrix >  M_ji( M_b->block(i,j)->copy() );
                        
                        M_ji->transpose();
                        M_ji->scale( real(0) );
                    
                        // ensure unique ID in matrix (not globally unique!)
                        M_ji->set_id( max_id_plus_1 + M_ji->id() );
                        
                        
                        root->set_block( j, i, M_ji.release() );
                    }// if
                }// if
                else
                {
                    root->set_block( i, j, get_full_format_of_lower_triangular_local( M_b->block(i,j), max_id_plus_1 ) );
                }// else
            }// for
        }// for
        
        return root;
    }// if
    else
    {
        unique_ptr< TMatrix > M_copy( M->copy() );
        
        M_copy->set_nonsym();
        
        return M_copy.release();
    }// else
}   



string
tot_dof_size ( const TMatrix * A )
{
    const size_t  mem  = A->byte_size();
    const size_t  pdof = size_t( double(mem) / double(A->rows()) );

    return Mem::to_string( mem ) + " (" + Mem::to_string( pdof ) + "/dof)";
}


real
matrix_dot ( const TSparseMatrix *  M,
             const TVector *        x,
             const bool             use_stable_version )
{
    //
    // check for nullptr, index set and type
    //

    if ( M == nullptr ) HERROR( ERR_ARG, "matrix_dot", "M = nullptr" );
    if ( x == nullptr ) HERROR( ERR_ARG, "matrix_dot", "x = nullptr" );

    if ( M->is_complex() )
        HERROR( ERR_REAL_CMPLX, "matrix_dot", "" );

    if (( M->row_is() != x->is() ) || ( M->col_is() != x->is() ))
        HERROR( ERR_VEC_STRUCT, "matrix_dot", "incompatible vector index set" );


    if ( ! IS_TYPE( x, TScalarVector ) )
        HERROR( ERR_VEC_TYPE, "matrix_dot", "only scalar vectors supported" );

    if ( M->n_non_zero() == 0 )
        return real(0);

    //
    // entry-wise multiplication
    // - only for real matrix
    //

    auto                 sx = cptrcast( x, TScalarVector );
    const size_t         n  = M->rows();
    std::vector< real >  tmp( n );

    for ( idx_t  i = 0; i < idx_t(n); i++ )
    {
        const idx_t  lb = M->rowptr(i);
        const idx_t  ub = M->rowptr(i+1);
        real         s  = 0;

        for ( idx_t j = lb; j < ub; j++ )
            s += M->rcoeff(j) * sx->blas_rvec()( M->colind(j) );


        tmp[i] = sx->blas_rvec()( i ) * s;
    }// for

    if ( use_stable_version )
    {
        // sort tmp w.r.t. absolute value of element
        std::sort( tmp.begin(), tmp.end(), BLAS::abs_lt< real > );
    }// if

    // compute final result
    real  res = real(0);

    for ( idx_t  i = 0; i < idx_t(n); ++i )
        res += tmp[i];

    return res;
}
 
}// namespace anonymous
    


  
size_t 
number_ev ( const TMatrix *   K,
            const real        alpha, 
            const TTruncAcc & acc ) 
{
    if ( K == nullptr )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "argument is nullptr" );
       
    //
    // compute the L*D*L^T decomposition of the matrix K - alpha * Id 
    //
    unique_ptr< TMatrix >  K_a( K->copy() );
    
    add_identity( K_a.get(), -alpha );
    
    fac_options_t  fac_opts;
    
    fac_opts.eval = point_wise;
    
    LDL::factorise( K_a.get(), acc, fac_opts);
       
    TMatrix *  L = nullptr;
    TMatrix *  D = nullptr;
        
    LDL::split( K_a.get(), L, D, fac_opts );
        
    //
    // count number of negative diagonal entries of D
    //
    size_t number = 0;
    
    for ( size_t i = 0; i < D->rows(); i++ )
    {
        if ( D->entry(i,i) < 0.0 )
            number++;
    }
        
    delete D;
    delete L;
    
    return number;
}


size_t
number_ev ( const TMatrix *   K,
            const real        alpha,
            const real        beta,
            const TTruncAcc & acc ) 
{
    if ( K == nullptr )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "argument is nullptr" );
       
    if ( !( beta > alpha ) )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "beta is not bigger than alpha" );
        
    const size_t n_b = number_ev( K, beta, acc );
    const size_t n_a = number_ev( K, alpha, acc );
    
    if ( n_b < n_a )
        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::number_ev", "number of eigenvalues inconsistent" ); 
    
    return n_b - n_a;
}


size_t 
number_ev ( const TMatrix *   K,
            const TMatrix *   M,
            const real        alpha,
            const TTruncAcc & acc ) 
{
    if ( K == nullptr || M == nullptr )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "argument is nullptr" );
       
    //   
    // compute the matrix C = K - alpha * M 
    //
    unique_ptr< TMatrix >  C( M->copy() );
    
    add( real(1), K, -alpha, C.get(), acc ); 
       
    //
    // compute L*D*L^T decomposition of the C
    //
    fac_options_t  fac_opts;
    
    fac_opts.eval = point_wise;
    
    LDL::factorise( C.get(), acc, fac_opts );
       
    TMatrix *  L = nullptr;
    TMatrix *  D = nullptr;
        
    LDL::split( C.get(), L, D, fac_opts );
        
    //
    // count number of negative diagonal entries of D
    //
    size_t number = 0;
    
    for ( size_t i = 0; i < D->rows(); i++ )
    {
        if ( D->entry(i,i) < 0.0 )
            number++;
    }
        
    delete D;
    delete L;
    
    return number;
}
                    

size_t 
number_ev ( const TMatrix *   K,
            const TMatrix *   M,
            const real        alpha,
            const real        beta,
            const TTruncAcc & acc ) 
{
    if ( K == nullptr || M == nullptr )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "argument is nullptr" );
       
    if ( !( beta > alpha ) )
       HERROR( ERR_ARG, "EIGEN_MISC::number_ev", "beta is not bigger than alpha" );
        
    const size_t n_b = number_ev( K, M, beta, acc );
    const size_t n_a = number_ev( K, M, alpha, acc );
    
    if ( n_b < n_a )
        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::number_ev", "number of eigenvalues inconsistent" ); 
    
    return n_b - n_a;

}




TMatrix *
get_nonsym_of_sym( const TMatrix * M ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_nonsym_of_sym", "argument is nullptr" );
        
    if ( ! M->is_symmetric() )
        LOG( "(eigen_misc) get_nonsym_of_sym : Warning: M is not symmetric" );
        
    const int max_id_plus_1 = max_id( M ) + 1;
    
    return get_nonsym_of_sym_local( M, max_id_plus_1 );
}           
   
   
TMatrix *
get_full_format_of_lower_triangular( const TMatrix * M ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_full_format_of_lower_triangular", "argument is nullptr" );
        
    if ( ! M->is_nonsym() )
        LOG( "(eigen_misc) get_full_format_of_lower_triangular : Warning: M is not nonsym" );
        
    const int max_id_plus_1 = max_id( M ) + 1;
    
    return get_full_format_of_lower_triangular_local( M, max_id_plus_1 );
}     
   
                   
                      
TMatrix *
get_sym_of_nonsym( const TMatrix * M ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_sym_of_nonsym", "argument is nullptr" );
        
    if ( M->is_symmetric() )
        LOG( "(eigen_misc) get_sym_of_nonsym : Warning: M is symmetric" );
        
    if ( is_blocked( M ) )
    {
        auto  M_b = cptrcast( M, TBlockMatrix ); 
        
        TBlockMatrix * root = new TBlockMatrix;
        
        //
        // Copy matrix structure of the blockmatrix 
        // but set the new matrix to symmetric
        //
        root->copy_struct_from( M_b );
        root->set_symmetric();
        
        if ( M_b->block_rows() != M_b->block_cols() )
            HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_sym_of_nonsym", "blockrows different to blockcols" ); 
     
        //
        // Copy only blocks of the lower triangular matrix
        //
        for ( uint i = 0; i < M_b->block_rows(); i++ )
        {
            for ( uint j = 0; j <= i; j++ )
            {
                root->set_block( i, j, nullptr );
                root->set_block( j, i, nullptr );
            
                if ( i != j )
                {        
                    if ( M_b->block(i,j) != nullptr )
                    {
                        unique_ptr< TMatrix >  M_ij( M_b->block(i,j)->copy() );
                        
                        root->set_block( i, j, M_ij.release() );
                        root->set_block( j, i, nullptr );
                    }// if
                }// if
                else
                {
                    root->set_block( i, j, get_sym_of_nonsym( M_b->block(i,j) ) );
                }// else
            }// for
        }// for
        
        return root;
    }// if
    else
    {
        unique_ptr< TMatrix > M_copy( M->copy() );
        
        M_copy->set_symmetric();
        
        return M_copy.release();
    }// else
}
                   
 

bool 
check_ortho ( const TMatrix * Z )
{
    if ( Z == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::check_ortho", "argument is nullptr" );
        
    bool is_ortho = true;    
    
    const size_t n = Z->rows();        
    const size_t p = Z->cols();    
        
    //
    // Handle special case of zero-sized matrix
    //
    if ( n*p == 0 )
        return false;
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Represent columns of matrix Z is vectors to ease following computations
    //
    ////////////////////////////////////////////////////////////////////////////
    vector< TVector * >  z_array( p, nullptr );
    
    // initialise vectors
    for ( size_t j = 0; j < p; j++ )
            z_array[j] = Z->row_vector().release();
    
    // copy vectors
    for ( size_t j = 0; j < p; j++ )
    {
        for ( size_t i = 0; i < n; i++ )
        {
            const real entry = Z->entry( i, j );
            
            z_array[j]->set_entry( i, entry );
        }// for
    }// for
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Check the norm of each vector
    //
    ////////////////////////////////////////////////////////////////////////////
    
    for ( size_t i = 0; i < p; i++ )
    {
        const real norm = z_array[i]->norm2();
        
        if ( Math::abs( norm - real(1) ) > ACCEPTED_AS_ZERO )
        {
            is_ortho = false;   
            
            LOG( to_string("(eigen_misc) check_ortho : vector in col = %d not normalized (norm = %.2e",i,norm) );
        }// if
    }// for
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Check the orthogonality of all vectors
    //
    ////////////////////////////////////////////////////////////////////////////
    for ( size_t i = 1; i < p; i++ )
    {   
        for ( size_t j = 0; j < i; j++ )
        {       
            const real prod = re( dot( z_array[i], z_array[j] ) );
            
            if ( Math::abs( prod ) > ACCEPTED_AS_ZERO)
            {
                is_ortho = false;  
                
                LOG( to_string("(eigen_misc) check_ortho : vector i=%d and j=%d not orthogonal (prod= %.2e",i,j,prod) );
            }// if
        }// for
    }// for
    
    ////////////////////////////////////////////////////////////////////////////
    //
    // Clean up data and return result
    //
    ////////////////////////////////////////////////////////////////////////////
    for ( size_t j = 0; j < p; j++ )
        delete z_array[j];
            
    return is_ortho;
}


bool 
check_ortho ( const vector < TVector * > & q,
              const size_t                 m )
{
    const size_t n = q[0]->size();
    
    unique_ptr< TDenseMatrix > Q( new TDenseMatrix(n,m) );
    
    for ( size_t j = 0; j < m; j++ )
    {
        for ( size_t i = 0; i < n; i++ )
        {
            const real entry = q[j]->entry(i);
            
            Q->set_entry( i, j, entry );
        }// for
    }// for
    
    return check_ortho( Q.get() );
}




bool 
check_M_ortho ( const TMatrix * , // Z
                const TMatrix * ) // M
{
    HERROR( ERR_CONSISTENCY, "EIGEN_MISC::check_M_ortho", "not yet supported" ); 
}




void
comp_rayleigh_quotients ( const TSparseMatrix * K_sparse, 
                          const TSparseMatrix * M_sparse,
                          TDenseMatrix *        D_rayleigh,
                          const TDenseMatrix *  Z_approx,
                          const bool            stable_rayleigh,
                          const bool            do_parallel ) 
{
    if ( K_sparse == nullptr || M_sparse == nullptr || Z_approx == nullptr || D_rayleigh == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::comp_rayleigh_quotients", "argument is nullptr" );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine the Rayleigh Quotients of the eigenvectors contained column-wise 
    // in 'Z_approx' according to the general eigenvalue problem (K_sparse,M_sparse)
    //
    /////////////////////////////////////////////////////////////////////////////////////
    const size_t nev = Z_approx->cols();
    
    D_rayleigh->set_size( nev, nev );
    D_rayleigh->scale( real(0) );
    D_rayleigh->set_ofs( K_sparse->col_ofs(), K_sparse->col_ofs() );
         
    #if 0
    const size_t n   = Z_approx->rows();
    const size_t ofs = Z_approx->row_ofs();
    
    auto compute_rayleigh = 
        [ K_sparse, M_sparse, Z_approx, n, ofs, D_rayleigh ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                unique_ptr< TScalarVector >  K_v ( new TScalarVector( n, ofs ) );
                unique_ptr< TScalarVector >  M_v ( new TScalarVector( n, ofs ) );
                
                //---------------------------------------------------------------
                // Get eigenvector (v is a reference to the column j of Z_approx)
                //---------------------------------------------------------------
                const TScalarVector v( Z_approx->column( j ) );
                
                //---------------------------------------------------------------
                // Compute K*v and M*v
                //---------------------------------------------------------------
                K_sparse->mul_vec( real(1), & v, real(0), K_v.get() );
                M_sparse->mul_vec( real(1), & v, real(0), M_v.get() );
                
                //---------------------------------------------------------------
                // Compute rayleigh = (v^T)*K*v / (v^T)*M*v
                //---------------------------------------------------------------
                const real rayleigh = re( dot( & v, K_v.get() ) ) / re( dot( & v, M_v.get() ) );
                
                D_rayleigh->set_entry( j, j, rayleigh );
            }// for
        };
    #endif
        
    auto compute_rayleigh = 
    [ K_sparse, M_sparse, Z_approx, D_rayleigh, & stable_rayleigh ] ( const tbb::blocked_range< uint > & r )
    {
        for ( auto  j = r.begin(); j != r.end(); ++j )
        {
            //---------------------------------------------------------------
            // Get eigenvector (v is a reference to the column j of Z_approx)
            //---------------------------------------------------------------
            const TScalarVector v( Z_approx->column( j ) );
            
            //---------------------------------------------------------------
            // Compute rayleigh = (v^T)*K*v / (v^T)*M*v
            //---------------------------------------------------------------
            const real vKv = matrix_dot( K_sparse, & v, stable_rayleigh );
            const real vMv = matrix_dot( M_sparse, & v, stable_rayleigh );
            //NOTE: In order to save memory bandwidth an extra routine has been written to compute
            //      (v^T)*M*v and which provides the option to compute this value in an stable version 
            
            D_rayleigh->set_entry( j, j, vKv / vMv );
        }// for
    };
        
        
    if ( do_parallel && (CFG::nthreads() > 1) ) 
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(nev) ), compute_rayleigh );
    else 
        compute_rayleigh ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
}



void
comp_rayleigh_quotients ( const TMatrix *       K,
                          const TMatrix *       M,
                          TDenseMatrix *        D,
                          const TDenseMatrix *  Z,
                          const TClusterTree *  ct )
{
    if ( K == nullptr || M == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::comp_rayleigh_quotients", "argument is nullptr" );

    bool sparse_arithmetic = false;
    
    if ( IS_TYPE( K, TSparseMatrix) && IS_TYPE( M, TSparseMatrix) && ct != nullptr )
        sparse_arithmetic = true;
  
    TIC;

    //////////////////////////////////////////////////////////////////////////////
    //
    // Determine the Rayleigh Quotients of the column vectors contained
    // in the matrix 'Z' according to the general eigenvalue problem (K,M)
    //
    //////////////////////////////////////////////////////////////////////////////
    const size_t n   = Z->rows();
    const size_t nev = Z->cols();
    const size_t ofs = Z->row_ofs();
    
    const TPermutation * perm_i2e = ct->perm_i2e();
    
    D->set_size( size_t(nev), size_t(nev) );
    D->set_ofs ( K->col_ofs(), K->col_ofs() );
    D->scale( real(0) );
    
    auto compute_rayleigh = 
    [ K, M, Z, perm_i2e, n, ofs, sparse_arithmetic, D ] ( const tbb::blocked_range< uint > & r )
    {
        for ( auto  j = r.begin(); j != r.end(); ++j )
        {
            unique_ptr< TScalarVector >  K_v ( new TScalarVector( n, ofs ) );
            unique_ptr< TScalarVector >  M_v ( new TScalarVector( n, ofs ) );
            
            if ( sparse_arithmetic )
            {
                //======================================================================
                // Get eigenvector (v_unperm is a reference to the column j of Z)
                //======================================================================
                const TScalarVector v_unperm( Z->column( j ) );
                
                //========================================================
                // Permutate eigenvector to external numbering
                //========================================================
                unique_ptr< TScalarVector >  v ( new TScalarVector( n, ofs ) );
                
                perm_i2e->permute( & v_unperm, v.get() );
                
                //========================================================
                // Compute K*v and M*v
                //========================================================
                K->mul_vec( real(1), v.get(), real(0), K_v.get() );
                M->mul_vec( real(1), v.get(), real(0), M_v.get() );
                
                //========================================================
                // Compute rayleigh = (v^T)*K*v / (v^T)*M*v
                //========================================================
                const real rayleigh = re( dot( v.get(), K_v.get() ) ) / re( dot( v.get(), M_v.get() ) );
                
                D->set_entry( j, j, rayleigh );
            }// if 
            else
            {
                //===============================================================
                // Get eigenvector (v is a reference to the column j of Z)
                //===============================================================
                const TScalarVector v( Z->column( j ) );
                
                //========================================================
                // Compute K*v and M*v
                //========================================================
                K->mul_vec( real(1), & v, real(0), K_v.get() );
                M->mul_vec( real(1), & v, real(0), M_v.get() );
                
                //========================================================
                // Compute rayleigh = (v^T)*K*v / (v^T)*M*v
                //========================================================
                const real rayleigh = re( dot( & v, K_v.get() ) ) / re( dot( & v, M_v.get() ) );
                
                D->set_entry( j, j, rayleigh );
            }// else
        }// for
    };
        
    bool  do_parallel = true;
    
    if ( CFG::nthreads() == 1 )
        do_parallel = false;
    
    if ( do_parallel ) 
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(nev) ), compute_rayleigh );
    else 
        compute_rayleigh ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
    
    
    TOC;
    
    /////////////////////////////////////////////////////////////////
    //
    // print information
    //
    /////////////////////////////////////////////////////////////////
    if ( false )
    {
        LOG( "" );
        LOG( to_string("(eigen_misc)  comp_rayleigh_quotients : done in %.2fs", toc.seconds() ));
        LOGHLINE;
        LOG( to_string("    n                 = %d",n) );
        LOG( to_string("    nev               = %d",nev) );
        LOG( to_string("    sparse_arithmetic = %d",sparse_arithmetic) );
        LOGHLINE;
    }// if
}


bool
is_pos_def ( const TMatrix * M )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::is_pos_def", "argument is nullptr" );

    TIC;

    TTruncAcc acc = TTruncAcc( EPS_REL_EXACT );

    const size_t n_minus = EIGEN_MISC::number_ev ( M, ACCEPTED_AS_ZERO, acc );
    
    TOC;
                   
    bool is_pos_def = true;
    
    if ( n_minus > 0 )
    {
        is_pos_def = false;
        
        LOG( "" );
        LOG( to_string("(eigen_misc) is_pos_def : done in %.2fs", toc.seconds() ));
        LOGHLINE;
        LOG( to_string("    ACCEPTED_AS_ZERO = %.2e",ACCEPTED_AS_ZERO) );
        LOG( to_string("    EPS_REL_EXACT    = %.2e",EPS_REL_EXACT) );
        LOG( to_string("    size of M        = %d",M->rows()) );
        LOG( to_string("    n_minus          = %d",n_minus) );
        LOG( to_string("    is_pos_def       = %d",is_pos_def) );
        LOGHLINE;
    }// if
    
    return is_pos_def;
}



bool
is_symmetric ( const TMatrix * M )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::is_symmetric", "argument is nullptr" );
        
    bool is_symmetric = true;

    if ( M->rows() == M->cols() )
    {
        for ( size_t j = 0; j < M->cols(); j++ )
        {
            for ( size_t i = 0; i < j; i++ )
            {
                const real diff = M->entry(i,j) - M->entry(j,i);
                
                if ( Math::abs( diff ) > ACCEPTED_AS_ZERO )
                {
                    is_symmetric = false;
                    
                    return is_symmetric;
                }// if 
            }// for
        }// for
    }// if
    else
    {
        is_symmetric = false;
    }// else
    
    return is_symmetric;
}
        


void
delete_upper_block_triangular( TMatrix * M )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::delete_upper_block_triangular", "argument is NULL" );
        
    if ( M->rows() != M->cols() ) 
        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::delete_upper_block_triangular", "rows and cols are different" ); 
              
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( M, TBlockMatrix ); 
                
        if ( BM->block_rows() != BM->block_cols() )
            HERROR( ERR_CONSISTENCY, "EIGEN_MISC::delete_upper_block_triangular", "blockrows and blockcols are different" ); 
     
        //
        // delete the upper block triangular part of the blockmatrix 
        //
        for ( uint i = 0; i < BM->block_rows(); i++ )
        {
            for ( uint j = i; j < BM->block_cols(); j++ )
            {
                if ( i < j )
                {        
                    if ( BM->block(i,j) != nullptr )
                    {
                        delete BM->block( i, j );
                        
                        BM->set_block( i, j, nullptr );
                    }// if
                }// if
                else
                    delete_upper_block_triangular( BM->block( i, j ) );
            }// for
        }// for   
    }// if
}


TMatrix *
get_halve_matrix ( const TMatrix * M ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::get_halve_matrix", "argument is nullptr" );
        
    if ( ! M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "EIGEN_MISC::get_halve_matrix", "" ); 
        
    unique_ptr< TMatrix >  M_half( get_halve_matrix_local ( M ) );
    
    M_half->set_nonsym();
    
    return M_half.release();
}




bool
has_inf_nan_entry ( const TMatrix * M )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::has_inf_nan_entry", "argument is NULL" );
        
    bool has_bad_data = false;

    for ( size_t j = 0; j < M->cols(); j++ )
    {
        for ( size_t i = 0; i < M->rows(); i++ )
        {
            if ( Math::is_inf( M->entry(i,j) ) )
            {
                LOG( to_string("(eigen_misc) has_inf_nan_entry : entry( %d, %d ) = Inf",i,j) );
                
                has_bad_data = true;
            }// if
            
            if ( Math::is_nan( M->entry(i,j) ) )
            {
                LOG( to_string("(eigen_misc) has_inf_nan_entry : entry( %d, %d ) = NaN",i,j) );
                
                has_bad_data = true;
            }// if
        }// for
    }// for
    
    return has_bad_data;
}



size_t
nnz ( const TMatrix * M )
{
    size_t count = 0;
    
    for ( size_t j = 0; j < M->cols(); j++ )
    {
        for ( size_t i = 0; i < M->rows(); i++ )
        {
            if ( M->entry(i,j) != real(0) )
                count++;
        }// for
    }// for
    
    return count;
}





void
load_matrix( TMatrix * &     M, 
             const string &  absolute_filename ) 
{        
    THLibMatrixIO   mio;
    
    TIC;
    M = mio.read( absolute_filename ).release();
    TOC;
    
    LOG( "" );
    LOG( to_string("(eigen_misc) load_matrix : done in %.2fs", toc.seconds() ));
    LOGHLINE;
    LOG( to_string("    rows     = %d",M->rows()) );
    LOG( to_string("    cols     = %d",M->cols()) );
    LOG(           "    memory   = "+tot_dof_size( M ) );
    LOG(           "    filename = '"+absolute_filename+"'" );
    LOGHLINE;
}



void
save_matrix ( const TMatrix *  M, 
              const string &   absolute_filename ) 
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "EIGEN_MISC::save_matrix", "argument is nullptr" );
        
    THLibMatrixIO   mio;
    
    TIC;
    mio.write( M, absolute_filename );
    TOC;
    
    LOG( "" );
    LOG( to_string("(eigen_misc) save_matrix : done in %.2fs", toc.seconds() ));
    LOGHLINE;
    LOG( to_string("    rows     = %d",M->rows()) );
    LOG( to_string("    cols     = %d",M->cols()) );
    LOG(           "    memory   = "+tot_dof_size( M ) );
    LOG(           "    filename = '"+absolute_filename+"'" );
    LOGHLINE;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// Useful routines used for debugging and development
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void print ( const TIndexSet & is ) ;

void print ( const TBlockIndexSet & bis ) ;

//! print matrix to console
void print_matrix ( const TMatrix * M, 
                    const uint      precision     = 2,
                    const bool      scientific    = true,
                    const bool      print_entries = true ) ;
                    
//! print matrix to console
void print_matrix ( const BLAS::Matrix< real > & M,
                    const uint                   precision  = 2,
                    const bool                   scientific = false,
                    const bool                   print_entries = true ) ;
                                        
//! print coordinates to console
void print_coordinates ( const TCoordinate * coord ) ;

void
print ( const TIndexSet & is )
{
    OUT( to_string( "(eigen_misc) print : indexset (%d,%d)",is.first(),is.last()) );
}


void
print ( const TBlockIndexSet & bis )
{
    OUT( to_string("(eigen_misc) print :   size: %d x %d  offset: (%d,%d)",
         bis.row_is().size(),bis.col_is().size(),bis.row_is().first(),bis.row_is().last()) );
}

// void
// print_matrix ( const TMatrix * M, 
//                const uint      precision,
//                const bool      scientific,
//                const bool      print_entries )
// {
//     if ( M == nullptr )
//     {
//         cout<<endl;
//         cout<<endl<<"(EIGEN_MISC::print_matrix)";
//         cout<<endl<<"\t matrix is nullptr";
//         cout<<endl;
//         cout<<endl;  
//               
//         return;
//     }// if
// 
//     cout<<endl;
//     cout<<endl<<"(EIGEN_MISC::print_matrix) \t size: "<<M->rows()<<" x "<<M->cols()<<" \t offset: ("<<M->row_ofs()<<","<<M->col_ofs()<<")";
//     cout<<endl;
//     cout<<endl;
// 
//     if ( print_entries )
//     {
//         cout.precision( precision );
//         
//         if ( scientific )
//             cout<<std::scientific;
//         else
//             cout<<std::fixed;
//     
//         for ( size_t i = 0; i < M->rows(); i++ )
//         {
//             for ( size_t j = 0; j < M->cols(); j++ )
//             {
//                     cout<<M->entry(i,j)<<"   ";
//             }// for
//     
//             cout<<endl;
//         }// for
//         
//         cout<<endl;
//     }// if
// }
// 
// 
// void
// print_matrix ( const BLAS::Matrix< real > & M,
//                const uint                   precision,
//                const bool                   scientific,
//                const bool                   print_entries ) 
// {    
//     cout<<endl;
//     cout<<endl;
//     cout<<"(EIGEN_MISC::print_matrix) \t size: "<<M.nrows()<<" x "<<M.ncols();
//     cout<<endl;
//     cout<<endl;
//     
//     if ( print_entries )
//     {
//         cout.precision( precision );
//         
//         if ( scientific )
//             cout<<std::scientific;
//         else
//             cout<<std::fixed;
//         
//         for ( size_t i = 0; i < M.nrows(); i++ )
//         {
//             for ( size_t j = 0; j < M.ncols(); j++ )
//             {
//                 cout<<M(i,j)<<"   ";
//             }// for
//     
//             cout<<endl;
//         }// for
//         
//         cout<<endl;
//     }// if
// }
// 
// 
// void
// print_coordinates ( const TCoordinate * coord )
// {
//     if ( coord == nullptr )
//         HERROR( ERR_ARG, "EIGEN_MISC::print_coordinates", "argument is nullptr" );
//         
//     cout<<endl;
//     cout<<endl<<"(EIGEN_MISC::print coordinates) \t dim = "<< coord->dim() <<", number of coordinates = "<< coord->ncoord();
//     cout<<endl;
//     cout<<endl;
//     
//     for ( size_t i = 0; i < coord->ncoord(); i++ )
//     {
//         cout<<endl<<" x_"<< i <<"= (";
//         
//         for ( size_t k = 0; k < coord->dim(); k++ )
//         {
//             cout<<" "<< coord->coord(i)[k] <<", ";
//         }// for
//         
//         cout<<")";
//     }// for
//     
//     for ( size_t i = 0; i < coord->ncoord(); i++ )
//     {
//         cout<<endl<<" (bbmax-bbmin)_"<< i <<"= (";
//         
//         for ( size_t k = 0; k < coord->dim(); k++ )
//         {
//             cout<<" "<< coord->bbmax(i)[k] - coord->bbmin(i)[k] <<", ";
//         }// for
//         
//         cout<<")";
//     }// for
//     
//     cout<<endl;
//     cout<<endl;
// }
//
//
//
// struct combi_timer_t
// {     
//         TCPUTimer   cpu;
//         TWallTimer  wall;
//        
//         
//         /////////////////////////////////////////////////////////
//         //
//         // constructor and destructor
//         //
//         combi_timer_t () {}
//         
//         
//         /////////////////////////////////////////////////////////
//         //
//         // misc functions
//         //
//         combi_timer_t &  start () { cpu.start(); wall.start(); return *this; }
// 
//         combi_timer_t &  pause () { cpu.pause(); wall.pause(); return *this; }
//     
//         combi_timer_t &  cont  () { cpu.cont (); wall.cont (); return *this; }
//         
//         void             add   ( const combi_timer_t & timer_2_add ) { /*cpu.add( timer_2_add.cpu ); wall.add( timer_2_add.wall );*/ return; }
//         
//         real             speedup    () const { return cpu.elapsed() / wall.elapsed(); }
//         
//         real             efficiency () const { return speedup() * real(100) / CFG::nthreads(); }
//         
//         std::string      to_string  () const 
//                             {                                 
//                                 std::ostringstream str; 
//                                 
//                                 str.precision( 1 );
//                                 str.flags( std::ios_base::fixed );
//                                 
//                                 str<<"\t S = "<<speedup();
//                                 
//                                 str.precision( 0 );
//                                 str<<"\t E = "<<efficiency()<<"%";
//                                                 
//                                 return str.str();
//                             }
// };    
// 
// 
// real
// percent_time( const TTimer part,
//               const TTimer total )
// {
//     const real percent = std::max( real(0) , part.elapsed() ) / std::max( real(0) , total.elapsed() ) * real(100);
//     
//     return percent;
// }
// 
// std::string 
// print_perf_help ( const combi_timer_t & part,
//                   const combi_timer_t & all,
//                   const bool            print_WALL_TIME )
// {
//     std::ostringstream str; 
//     
//     str.precision( 2 );
//     str.flags( std::ios_base::fixed );
// 
//     if ( print_WALL_TIME )
//         str<<part.wall<<"\t ("<<percent_time( part.wall, all.wall )<<"%)"<<part.to_string();
//     else
//         str<<part.cpu<<"\t ("<<percent_time( part.cpu, all.cpu )<<"%)";
// 
//     return str.str();
// }
// 
// real
// avg_time ( const combi_timer_t  timer,
//            const TMatrix *      Z_approx ) 
// {
//     const size_t N    = Z_approx->rows();
//     const size_t n_ev = Z_approx->cols();
//     
//     const real factor   = Math::pow( real(10), real(6) ) / ( real(N) * real(n_ev) );
//     
//     const real average_time = std::max( real(0) , timer.wall.elapsed() ) * factor;
//     
//     return average_time;
// }




}// namespace EIGEN_MISC


}// namespace HAMLS

