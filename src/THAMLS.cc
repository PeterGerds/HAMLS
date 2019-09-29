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
// File        : THAMLS.cc
// Description : class for the HAMLS algortihm
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "hlib-config.h"

#if USE_MKL == 1
#include <mkl_service.h>
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <sys/resource.h>  // for routine print_mem_usage()

#include "algebra/solve_tri.hh"

#include "matrix/TSparseMatrix.hh"
#include "matrix/TZeroMatrix.hh"

#include "io/TClusterVis.hh"
#include "io/TMatrixIO.hh"
#include "io/TMatrixVis.hh"

#include "solver/TSolver.hh"
#include "solver/TAutoSolver.hh"

#include "hamls/TEigenArnoldi.hh"
#include "hamls/TEigenArpack.hh"
#include "hamls/TEigenLapack.hh"
#include "hamls/TSeparatorTree.hh"
#include "hamls/THAMLS.hh"
#include "hamls/eigen_misc.hh"


namespace HAMLS
{

using std::unique_ptr;
using std::string;
using std::vector;
using std::list;


using EIGEN_MISC::get_sym_of_nonsym;
using EIGEN_MISC::get_nonsym_of_sym;

using tbb::affinity_partitioner;

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

namespace
{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// debug functions
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct timer_HAMLS_t
{
    double all;
    double T2_T3;
    double T4;
    double condensing;
    double T6;
    double T7;
    double T8;

    //constructor
    timer_HAMLS_t ()
    {
        all        = 0;
        T2_T3      = 0;
        T4         = 0;
        condensing = 0;
        T6         = 0;
        T7         = 0;
        T8         = 0;
    }
    
    void print_performance() 
    {
        OUT( "" );
        OUT( "(THAMLS) print_performance :" );
        HLINE;
        OUT( to_string("    task_scheduler_init::default_num_threads() = %d",tbb::task_scheduler_init::default_num_threads()) );
        OUT( to_string("    CFG::nthreads()                            = %d",CFG::nthreads()) );
        HLINE;
        OUT( to_string("    all done in %fs",all) );
        HLINE;
        OUT( to_string("    T2_T3      = %f %",T2_T3/all*100) );
        OUT( to_string("    T4         = %f %",T4/all*100) );
        OUT( to_string("    condensing = %f %",condensing/all*100) );
        OUT( to_string("    T6         = %f %",T6/all*100) );
        OUT( to_string("    T7         = %f %",T7/all*100) );
        OUT( to_string("    T8         = %f %",T8/all*100) );
        HLINE;
    }
};
 
    
    
real
comp_smallest_evalue( const TMatrix * M )
{
    unique_ptr< TDenseMatrix >   D( new TDenseMatrix() );
    unique_ptr< TDenseMatrix >   Z( new TDenseMatrix() );

    TEigenLapack eigensolver( idx_t(1), idx_t(1) );
        
    eigensolver.set_test_pos_def  ( false );    
    eigensolver.set_test_residuals( false );
    
    eigensolver.comp_decomp( M, D.get(), Z.get() );
    
    return D->entry(0,0);
}




string
tot_dof_size ( const TMatrix * A )
{
    const size_t  mem  = A->byte_size();
    const size_t  pdof = size_t( double(mem) / double(A->rows()) );

    return Mem::to_string( mem ) + " (" + Mem::to_string( pdof ) + "/dof)";
}

string
tot_dof_size ( const unique_ptr< TMatrix > & A )
{
    return tot_dof_size( A.get() );
}

string
mem_in_MB ( const TMatrix * A )
{
    const size_t  bytes  = A->byte_size();
    std::ostringstream  str;
    
    size_t  mb, kb;

    // Mememory Consumption in MiB
    // mb = bytes / (1024*1024);
    // kb = size_t( ((double(bytes) / double(1024*1024)) - double(mb)) * 100.0 );
    
    // Mememory Consumption in MB
    mb = bytes / (1000*1000);
    kb = size_t( ((double(bytes) / double(1000*1000)) - double(mb)) * 100.0 );

    str << mb;
            
    if ( kb < 10 ) str << ".0";
    else           str << '.' ;

    str << kb << " MB";
    
    return str.str();
}


void
print_mem_usage ()
{
    rusage usage;
    getrusage( RUSAGE_SELF, &usage);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // The value 'ru_maxrss' delivers the most reliable measure for the maximal 
    // used main memory (non-swapped physical memory) The value 'Mem::max_usage()' 
    // delivers just the maximal by the function 'Mem::to_string()' measured memory.
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    OUT( "" );
    OUT( "(THAMLS) print_mem_usage : " );
    //NOTE: return value of 'usage.ru_maxrss'is measuered in kB 
    HLINE;
    OUT( "    ru_maxrss is                        "+Mem::to_string( usage.ru_maxrss * 1024 ) );
    //NOTE: units are using IEC prefixes
    OUT( "    current measured memory consumption "+Mem::to_string() );
    // NOTE: This value has not to be the maximal memory consumption at all. It is 
    // only the maximal value which was measured by the function 'Mem::to_string()'
    OUT( "    maximal measured memory consumption "+Mem::to_string( Mem::max_usage() ) );
    HLINE;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Auxilliary routines for "comp_M_red_ij_aux_step_1"
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void mul_block_with_dense_to_dense  ( const matop_t         op_A,
                                      const TBlockMatrix *  A,
                                      const TDenseMatrix *  B,
                                      TDenseMatrix *        C,
                                      const size_t          mul_block_with_dense_max_size ) ;

void
special_multi ( const matop_t        op_A,
                const TMatrix *      A,
                const TDenseMatrix * B,
                TDenseMatrix *       C, 
                const size_t         mul_block_with_dense_max_size ) 
{
    if ( A == nullptr || B == nullptr || C == nullptr )
        HERROR( ERR_ARG, "(THAMLS) special_multi", "argument is nullptr" );
        
    ////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the matrix C = C + op_A(A) * B  
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////

    if ( ! IS_TYPE( A, TBlockMatrix ) )
    {
        multiply( real(1), op_A, A, MATOP_NORM, B, real(1), C, acc_exact );
    }// if
    else
    {   
        auto  Block_A = cptrcast( A, TBlockMatrix );
        
        mul_block_with_dense_to_dense( op_A, Block_A, B, C, mul_block_with_dense_max_size );
    }// else
}


bool
do_parallel_mul_block_with_dense_to_dense ( const matop_t         op_A,
                                            const TBlockMatrix *  A,
                                            const TMatrix *       B,
                                            const size_t          mul_block_with_dense_max_size )  
{
    bool do_parallel = false;
        
    const bool       trans_A    = ( op_A != MATOP_NORM );
    const uint       nblockrows = ( trans_A ? A->block_cols() : A->block_rows() );
    
    if ( nblockrows > 1 )
    {
        if ( real(std::max( A->rows(), A->cols() )) > (real(mul_block_with_dense_max_size)/real(B->cols())) )
            do_parallel = true;
    }// if 
    
    return do_parallel;
}




void
mul_block_with_dense_to_dense  ( const matop_t         op_A,
                                 const TBlockMatrix *  A,
                                 const TDenseMatrix *  B,
                                 TDenseMatrix *        C,
                                 const size_t          mul_block_with_dense_max_size )  
{
    ////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the matrix C = C + op_A(A) * B  
    //
    // multiply block matrix A with dense matrix B to dense matrix C
    //
    // slightly changed routine from mat_mul.cc! 
    //
    ////////////////////////////////////////////////////////////////////////////////////
    
    //
    // multiply row wise, where row is defined w.r.t. <op_A>
    //

    const bool       trans_A    = ( op_A != MATOP_NORM );
    const uint       nblockrows = ( trans_A ? A->block_cols() : A->block_rows() );
    const uint       nblockcols = ( trans_A ? A->block_rows() : A->block_cols() );
    const TIndexSet  colis_C( C->col_is() );
    const TIndexSet  rowis_B( B->row_is() );
    const TIndexSet  colis_B( B->col_is() );

    auto  sub_mult =
        [nblockcols,colis_C,rowis_B,colis_B,op_A,A,B,C,mul_block_with_dense_max_size]
        ( const uint  i )
        {
            unique_ptr< TDenseMatrix >  C_sub;
        
            for ( uint  j = 0; j < nblockcols; ++j )
            {
                const TMatrix *  A_ij  = nullptr;
                matop_t          top_A = op_A;

                if ( op_A == MATOP_NORM )
                {
                    if      ( A->is_nonsym()    || ( i >= j )) { A_ij = A->block( i, j ); top_A = MATOP_NORM; }
                    else if ( A->is_symmetric() && ( i <  j )) { A_ij = A->block( j, i ); top_A = MATOP_TRANS; }
                    else if ( A->is_hermitian() && ( i <  j )) { A_ij = A->block( j, i ); top_A = MATOP_ADJ; }
                }// if
                else
                {
                    if      ( A->is_nonsym()    || ( i <= j )) { A_ij = A->block( j, i ); top_A = op_A; }
                    else if ( A->is_symmetric() && ( i >  j )) { A_ij = A->block( i, j ); top_A = MATOP_NORM; }
                    else if ( A->is_hermitian() && ( i >  j )) { A_ij = A->block( i, j ); top_A = MATOP_NORM; }
                }// else

                if ( A_ij == nullptr )
                    continue;

                //
                // set up destination block, if not yet done
                //
            
                if ( C_sub.get() == nullptr )
                {
                    const TIndexSet       subis_C( top_A == MATOP_NORM ? A_ij->row_is() : A_ij->col_is() );
                    BLAS::Matrix< real >  M( blas_mat< real >( C ), subis_C - C->row_ofs(), BLAS::Range::all );
                
                    C_sub = std::make_unique< TDenseMatrix >( subis_C, colis_C, M );
                }// if

                //
                // define B source
                //

                unique_ptr< TDenseMatrix >  B_sub;

                {
                    const TIndexSet  subis_B( top_A == MATOP_NORM ? A_ij->col_is() : A_ij->row_is() );

                    BLAS::Matrix< real >  M( blas_mat< real >( B ), subis_B - B->row_ofs(), BLAS::Range::all );
                
                    B_sub = std::make_unique< TDenseMatrix >( subis_B, colis_B, M );                    
                }

                //
                // multiply
                //
                special_multi( top_A, A_ij, B_sub.get(), C_sub.get(), mul_block_with_dense_max_size );
            }// for
        };

    if ( do_parallel_mul_block_with_dense_to_dense( op_A, A, B, mul_block_with_dense_max_size ) )
    {
        tbb::parallel_for( uint(0), nblockrows, sub_mult );
    }// if
    else
    {
        multiply( real(1), op_A, A, MATOP_NORM, B, real(1), C, acc_exact );
    }// else
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Parallel routines for the computation of A^{T} * B where A and B are dense matrices with
//  A->rows() >> A->cols() and  B->rows() >> B->cols()
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///
/// Alt: kann weg
///
/// Nicht mehr benoetigt, aber eine nette Routine und sollte aufgehoben werden fuer alle Faelle
/// Die Routine hatte ich mal gebraucht als ich eine Variante der 'multiply_parallel_dense' 
/// implementiert hatte --> siehe Workspace S. 17 beispielsweise
///
TMatrix * 
get_subdivided_copy_by_reference ( const TDenseMatrix *   M,
                                   const TBlockIndexSet & block_is,
                                   const size_t           max_size )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "(THAMLS) get_subdivided_copy_by_reference", "argument is nullptr" );
        
    if ( ! M->block_is().is_sub( block_is ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subdivided_copy_by_reference", "" ); 
        
    
        
    /////////////////////////////////////////////////////////////////////////////////////////
    //
    // Subdivide the submatrix of M which is indicated by  
    // the given block indexset recursively into subblocks
    //
    // Note: The recursive subdivision into subblocks is necessary because this scales better
    //       using multiple threads instead of subdividing the matrix just one time in
    //       many small subblocks. Recursively subdividing allows to compute the involved
    //       matrix sums partially in parallel. If no recursive approach is used this is 
    //       not less possible.
    //
    /////////////////////////////////////////////////////////////////////////////////////////
    
    const TIndexSet row_is =  block_is.row_is();
    const TIndexSet col_is =  block_is.col_is();
        
    const size_t rows = row_is.size();    
    const size_t cols = col_is.size();
    
    if ( rows <= max_size && cols <= max_size )
    {
        //-----------------------------------------------------------------------------------
        //
        // No subdivision is necessary, just a copy by reference is returned
        //
        //-----------------------------------------------------------------------------------
        BLAS::Matrix< real >  M_BLAS_sub( M->blas_rmat(),
                                          row_is - M->row_ofs(),
                                          col_is - M->col_ofs() );
        
        unique_ptr< TDenseMatrix > M_sub( new TDenseMatrix( row_is, col_is, M_BLAS_sub ) );
                        
        return M_sub.release();
    }// if 
    else
    {
        /////////////////////////////////////////////////////////////////////////////////////
        //
        // Subdivide the indicated submatrix into blocks
        //
        /////////////////////////////////////////////////////////////////////////////////////
    
        TBlockMatrix * root = new TBlockMatrix;
        
        //----------------------------------------------------
        // adjust the TBlockMatrix::copy_struct() directitives 
        //----------------------------------------------------
        ///root->set_id( M->id() );
        ///root->set_form( M->form() );
        root->set_nonsym();
        root->set_complex( M->is_complex() );
        ///root->set_procs( M->procs() );
        root->set_block_is( block_is );
        
        //---------------------------------------------------------------------------------------
        //
        // Subdivide into blockstructure 2x1 
        //
        //---------------------------------------------------------------------------------------
        if ( rows > max_size && cols <= max_size )
        {
            //-------------------------------
            // Initialise the block structure
            //-------------------------------
            root->set_block_struct( 2, 1 );
            
            //----------------------------------------------------------
            // Determine the corresponding index sets of the submatrices
            //----------------------------------------------------------
            
            const idx_t row_mid = (row_is.first() + row_is.last())/2;
            
            const TIndexSet row_is_up  ( row_is.first(), row_mid  );
            const TIndexSet row_is_down( row_mid+1, row_is.last() );
            
            for ( uint i = 0; i < 2; i++ )
            {
                root->set_block( i, 0, nullptr );
                
                //-------------------------------------------------
                // Create a referece to the corresponding submatrix 
                //-------------------------------------------------
                TIndexSet row_is_sub;
                
                if ( i == 0 )
                    row_is_sub = row_is_up;
                else
                    row_is_sub = row_is_down;
                    
                const TBlockIndexSet block_is_sub ( row_is_sub, col_is );
                                
                //---------------------------------------------------
                // Apply the subdivision into blocks recursively to 
                // the submatrix indicated by the new block index set
                //---------------------------------------------------
                unique_ptr< TMatrix > M_sub_subdivided( get_subdivided_copy_by_reference ( M, block_is_sub, max_size ) );
                    
                root->set_block( i, 0, M_sub_subdivided.release() );
            }// for 
                      
            return root;
        }// if
        
        //---------------------------------------------------------------------------------------
        //
        // Subdivide into blockstructure 1x2 
        //
        //---------------------------------------------------------------------------------------
        if ( rows <= max_size && cols > max_size )
        {
            //-------------------------------
            // Initialise the block structure
            //-------------------------------
            root->set_block_struct( 1, 2 );
            
            //----------------------------------------------------------
            // Determine the corresponding index sets of the submatrices
            //----------------------------------------------------------
            
            const idx_t col_mid = (col_is.first() + col_is.last())/2;
            
            const TIndexSet col_is_left  ( col_is.first(), col_mid  );
            const TIndexSet col_is_right(  col_mid+1, col_is.last() );
            
            for ( uint j = 0; j < 2; j++ )
            {
                root->set_block( 0, j, nullptr );
                
                //-------------------------------------------------
                // Create a referece to the corresponding submatrix 
                //-------------------------------------------------
                TIndexSet col_is_sub;
                
                if ( j == 0 )
                    col_is_sub = col_is_left;
                else
                    col_is_sub = col_is_right;
                    
                const TBlockIndexSet block_is_sub ( row_is, col_is_sub );
                                
                //---------------------------------------------------
                // Apply the subdivision into blocks recursively to 
                // the submatrix indicated by the new block index set
                //---------------------------------------------------
                unique_ptr< TMatrix > M_sub_subdivided( get_subdivided_copy_by_reference ( M, block_is_sub, max_size ) );
                    
                root->set_block( 0, j, M_sub_subdivided.release() );
            }// for 
            
            return root;
        }// if
    
        
        //---------------------------------------------------------------------------------------
        //
        // Subdivide into blockstructure 2x2 
        //
        //---------------------------------------------------------------------------------------
        if ( rows > max_size && cols > max_size )
        {
            //-------------------------------
            // Initialise the block structure
            //-------------------------------
            root->set_block_struct( 2, 2 );
            
            //-------------------------------------------------------------
            // Determine the corresponding index sets of the submatrices
            //-------------------------------------------------------------
            
            const idx_t col_mid = (col_is.first() + col_is.last())/2;
            const idx_t row_mid = (row_is.first() + row_is.last())/2;
            
            const TIndexSet col_is_left  ( col_is.first(), col_mid  );
            const TIndexSet col_is_right(  col_mid+1, col_is.last() );
            
            const TIndexSet row_is_up  ( row_is.first(), row_mid  );
            const TIndexSet row_is_down( row_mid+1, row_is.last() );
                
            for ( uint j = 0; j < 2; j++ )
            {
                for ( uint i = 0; i < 2; i++ )
                {
                    root->set_block( i, j, nullptr );
                    
                    //--------------------------------------------------
                    // Create a referece to the corresponding submatrix 
                    //--------------------------------------------------
                    TIndexSet col_is_sub;
                    TIndexSet row_is_sub;
                    
                    if ( j == 0 )
                        col_is_sub = col_is_left;
                    else
                        col_is_sub = col_is_right;
                        
                    if ( i == 0 )
                        row_is_sub = row_is_up;
                    else
                        row_is_sub = row_is_down;
                        
                    
                    const TBlockIndexSet block_is_sub ( row_is_sub, col_is_sub );
                                
                    //---------------------------------------------------
                    // Apply the subdivision into blocks recursively to 
                    // the submatrix indicated by the new block index set
                    //---------------------------------------------------
                    unique_ptr< TMatrix > M_sub_subdivided( get_subdivided_copy_by_reference ( M, block_is_sub, max_size ) );
                        
                    root->set_block( i, j, M_sub_subdivided.release() );
                }// for
            }// for 
                     
            return root;
        }// if
        
        
        //---------------------------------------------------------------------------------------
        //
        // No other case should occur
        //
        //---------------------------------------------------------------------------------------
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subdivided_copy_by_reference", "no other case should occur" );
    }// else
}






void 
sequential_multiply_parallel_dense_sub ( const TDenseMatrix * A,
                                         const TDenseMatrix * B,
                                         const TIndexSet &    A_B_is, 
                                         TDenseMatrix *       C_sub ) 
{
    ////////////////////////////////////////////////////////////////////////////////
    //
    // Restict the matrices A and B to the corresponding indexsets and compute 
    //
    //      C_sub = A_restict^{T} * B_restrict 
    //
    // sequentially. Use a copy by reference for the restriction instead of a copy 
    // by value in order to  spare memory bandwith.
    //
    ////////////////////////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------
    // restrict A 
    //--------------------------------------------------------------
    BLAS::Matrix< real >  A_BLAS_restricted ( A->blas_rmat(), 
                                              A_B_is - A->row_ofs(),
                                              C_sub->row_is() - A->col_ofs() );

    unique_ptr< TDenseMatrix > A_restricted ( new TDenseMatrix( A_B_is, C_sub->row_is(), A_BLAS_restricted ) );
    
    //--------------------------------------------------------------
    // restrict B 
    //--------------------------------------------------------------
    BLAS::Matrix< real >  B_BLAS_restricted ( B->blas_rmat(),
                                              A_B_is - B->row_ofs(),
                                              C_sub->col_is() - B->col_ofs() );
    
    unique_ptr< TDenseMatrix > B_restricted( new TDenseMatrix( A_B_is, C_sub->col_is(), B_BLAS_restricted ) );
        
    //--------------------------------------------------------------
    // Compute C_sub = A_resticted^{T} * B_restricted 
    //--------------------------------------------------------------
    const TTruncAcc acc( real(0) );

    multiply( real(1), MATOP_TRANS, A_restricted.get(), MATOP_NORM, B_restricted.get(), real(0), C_sub, acc ); 
}




class multiply_parallel_dense_sub_task_t : public tbb::task
{
  private:
   const TDenseMatrix * _A;
   const TDenseMatrix * _B;
   const TIndexSet &    _A_B_is; 
   TDenseMatrix *       _C_sub;
   const size_t &       _min_size;

  public:

    multiply_parallel_dense_sub_task_t( const TDenseMatrix * A,
                                        const TDenseMatrix * B,
                                        const TIndexSet &    A_B_is, 
                                        TDenseMatrix *       C_sub, 
                                        const size_t &       min_size )
        : _A( A )
        , _B( B )
        , _A_B_is( A_B_is )
        , _C_sub( C_sub )
        , _min_size( min_size )
    {}

    task * execute()
    {
        ////////////////////////////////////////////////////////////////////////////////////
        //
        // Restict the matrices A and B to the corresponding indexsets and compute 
        //
        //      C_sub = A_restict^{T} * B_restrict 
        //
        // in parallel or sequentially
        //
        ////////////////////////////////////////////////////////////////////////////////////
    
        if ( _A_B_is.size() <= _min_size )
        {
            //----------------------------------------------------------------------
            //
            // Seriell computation of the product
            //
            //----------------------------------------------------------------------
            sequential_multiply_parallel_dense_sub ( _A, _B, _A_B_is, _C_sub );
            
            return nullptr;
        }// if
        else
        {
            //---------------------------------------------------------------------------
            // Split A_restrict and B_restrict implicitely each into two subblocks, i.e., 
            //
            // A_restrict = [A_1] and B_restict = [B1]
            //              [A_2]                 [B2]
            //
            // just by splitting the indexset 'A_B_is' into the subsets 'is_1' and 'is_2'
            //---------------------------------------------------------------------------
            
            const idx_t first = _A_B_is.first();
            const idx_t last  = _A_B_is.last();
            const idx_t mid   = idx_t( (last + first)/2 );
            
            TIndexSet is_1( first,        mid  );
            TIndexSet is_2( idx_t(mid+1), last );
        
        
            ///TODO: Continuation Task implementieren mit Bypass
        
            #if 1
            //---------------------------------------------------------------------------
            // Compute C1 = A1^{T} * B1 and C2 = A2^{T} * B2
            // 
            // Note: It is sufficient to use the input matrix C_sub
            //       instead of using an extra copy C1
            //---------------------------------------------------------------------------
            unique_ptr< TDenseMatrix > C2( new TDenseMatrix( _C_sub->row_is(), _C_sub->col_is() ) );
        
            tbb::task & child1 = * new( allocate_child() ) multiply_parallel_dense_sub_task_t( _A, _B, is_1, _C_sub,   _min_size );
            tbb::task & child2 = * new( allocate_child() ) multiply_parallel_dense_sub_task_t( _A, _B, is_2, C2.get(), _min_size );
            
            set_ref_count( 3 );
            spawn( child2 );
            spawn_and_wait_for_all( child1 );
        
            //---------------------------------------------------------------------------
            // Compute C = C1 + C2
            //---------------------------------------------------------------------------
            _C_sub->add( real(1), C2.get() );
            
            return nullptr;
            #endif
        }// else
    }
};


///
/// Huebsche aber anscheinend ineffiziente Routine (nocheinmal Benchmarken)
///
void
multiply_parallel_dense_sub ( const TDenseMatrix * A,
                              const TDenseMatrix * B,
                              TDenseMatrix *       C_sub,
                              const size_t         min_size )
{
    
    ////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix C_sub = (A^{T} * B)_sub in parallel 
    // 
    ////////////////////////////////////////////////////////////////////////////////////    
    const idx_t A_B_first = B->row_is().first();
    const idx_t A_B_last  = B->row_is().last();
        
    const TIndexSet A_B_is( A_B_first, A_B_last );
    
    if ( B->rows() > min_size )
    {
        //----------------------------------------------------------------------
        //
        // Parallel computation of the product
        //
        //----------------------------------------------------------------------
            
        tbb::task & root = * new( tbb::task::allocate_root() ) multiply_parallel_dense_sub_task_t( A, B, A_B_is, C_sub, min_size );

        tbb::task::spawn_root_and_wait( root );
    }// if
    else    
    {
        //----------------------------------------------------------------------
        //
        // Seriell computation of the product
        //
        //----------------------------------------------------------------------
        
        sequential_multiply_parallel_dense_sub ( A, B, A_B_is, C_sub );
                
    }// else
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Miscellaneous
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void
sparse_matrix_multi ( const TSparseMatrix * M_sparse, 
                      const TDenseMatrix *  Q_old,
                      TDenseMatrix *        Q_new,
                      const bool            do_parallel ) 
{
    if ( M_sparse == nullptr || Q_old == nullptr || Q_new == nullptr )
        HERROR( ERR_ARG, "(THAMLS) sparse_matrix_multi", "argument is nullptr" );

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   Q_new = M_sparse * Q_old
    //
    /////////////////////////////////////////////////////////////////////////////////////        
    const size_t n   = Q_old->rows();
    const size_t nev = Q_old->cols();
        
    Q_new->set_size( n, nev );
    Q_new->set_ofs( Q_old->col_ofs(), Q_old->col_ofs() );
    
    auto compute_column = 
        [ M_sparse, Q_old, Q_new ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //-------------------------------------------------------
                // uses references for the corresponding columns
                //-------------------------------------------------------
                const TScalarVector x( Q_old->column( j ) );
                TScalarVector       y( Q_new->column( j ) );
                
                //-------------------------------------------------------
                // Compute y = M * x
                //-------------------------------------------------------
                M_sparse->mul_vec( real(1), & x, real(0), & y );
            }// for
        };
        
    if ( do_parallel ) 
    {
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(nev) ), compute_column );
    }// if 
    else 
        compute_column ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
}


void
solve_lower_right_symmetric ( TMatrix *               A,
                              const matop_t           op_L,
                              const TMatrix *         L,
                              const TMatrix *         D,
                              const TTruncAcc &       acc,
                              const solve_option_t    options,
                              const TSeparatorTree *  sep_tree,
                              TProgressBar *          progress = nullptr )
{
    if ( A->is_symmetric() ) 
        HERROR( ERR_CONSISTENCY, "solve_lower_right_symmetric", "matrix is symmetric" ); 

    //
    // The upper block diagonal part of A will be deleted. This saves a lot of 
    // computational time and this part is not needed when the result is symmetric anyway
    //           
    EIGEN_MISC::delete_upper_block_triangular( A );
    
    solve_lower_right ( A, op_L, L, D, acc, options, progress );
    
    //
    // set the result symmetric (Note: the set_symmetric routine is applied automatically to the block
    // diagonal matrices if A is a TBlockMatrix -> see Implementation of "set_symmtric()" )
    // 
    A->set_symmetric();
}






bool
is_zero_size ( const TMatrix * M )
{
    if ( M->rows() * M->cols() == 0 )
        return true;
    else
        return false;
}


//
// Set to identity matrix 
//
void
set_Id_dense( TDenseMatrix * M )
{        
    if ( M == nullptr )
       HERROR( ERR_ARG, "set_Id_dense", "argument is nullptr" );
       
    if ( M->rows() != M->cols() )
       HERROR( ERR_CONSISTENCY, "set_Id_dense", "" );

    M->scale( real(0) );
    
    for ( size_t i = 0; i < M->rows(); i++ )
        M->set_entry( i, i, real(1) );
}


// 
// Set the diagonal of the given dense matrix
//
void
set_diagonal_dense( const TDenseMatrix * D, 
                    TDenseMatrix *       M )
{            
    if ( M == nullptr || D == nullptr )
       HERROR( ERR_ARG, "set_diagonal_dense", "argument is nullptr" );
       
    const size_t n = M->rows();
    
    if ( n != M->cols() )
       HERROR( ERR_CONSISTENCY, "set_diagonal_dense", "" );
       
    if ( D->rows() != n || D->cols() != n )
       HERROR( ERR_CONSISTENCY, "set_diagonal_dense", "" );
    
    M->scale( real(0) );
    
    for ( size_t i = 0; i < n; i++ )
    {
        const real entry = D->entry(i,i);
        
        M->set_entry( i, i, entry );
    }// for
}






const TMatrix *
search_submatrix ( const TMatrix *   M, 
                   const TIndexSet & row_is,
                   const TIndexSet & col_is )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "(THAMLS) search_submatrix", "argument is nullptr" );
        
    if ( ! col_is.is_left_or_equal_to( row_is ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) search_submatrix", "" ); 
    
    if ( ! M->col_is().is_sub( col_is ) )  
        HERROR( ERR_CONSISTENCY, "(THAMLS) search_submatrix", "" ); 
    
    if ( ! M->row_is().is_sub( row_is ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) search_submatrix", "" ); 
        
    if ( col_is == M->col_is() && row_is == M->row_is() )
        return M;
                
    if ( IS_TYPE( M, TBlockMatrix ) )
    {
        auto  BM = cptrcast( M, TBlockMatrix );
        
        for ( uint j = 0; j < BM->block_cols(); ++j )
        {
            for ( uint i = 0; i < BM->block_rows(); ++i )
            {
                const TMatrix  * M_ij = BM->block(i,j);
            
                if ( M_ij == nullptr )
                    continue;
                        
                //
                // If the submatrix matchs to the indexsets 
                // I and J then return this submatrix
                //                 
                if ( M_ij->col_is().is_sub( col_is ) && M_ij->row_is().is_sub( row_is ) )
                    return search_submatrix( M_ij, row_is, col_is );
            }// for
        }// for
        
        //
        // If no submatrix found which matchs to the 
        // indexsets I and J then return the block matrix
        //
        return M;
    }// if
    else
        return M;   
}





        
TMatrix *
get_restricted_copy_by_reference ( const TMatrix *        M,
                                   const TBlockIndexSet & block_is )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "(THAMLS) get_restricted_copy_by_reference", "argument is nullptr" );
            
    if ( ! M->block_is().is_sub( block_is ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "" ); 
        

    ///////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Create restricted version of M_tilde_ij using only copies by reference and NOT copies
    // by value in order to spare memory bandwidth. This is getting important when many cores 
    // are used in a shared memory system and memory bandwidth is becoming a bottleneck of 
    // the computer system architecture and preventing further scalability.
    //
    //
    ///////////////////////////////////////////////////////////////////////////////////////////
    const TIndexSet row_is =  block_is.row_is();
    const TIndexSet col_is =  block_is.col_is();
        
    const TIndexSet intersect_row_is = intersect( row_is, M->row_is() );
    const TIndexSet intersect_col_is = intersect( col_is, M->col_is() );
    

    if( IS_TYPE( M, TDenseMatrix ) )
    {          
    
        /////////////////////////////////////////////////////////////////////////////////////// 
        //
        //
        // Case: TDenseMatrix
        //
        //
        ///////////////////////////////////////////////////////////////////////////////////////
    
        auto  DM = cptrcast( M, TDenseMatrix );
                                                            
        //-------------------------------------------------------------
        //
        // Create restricted TDenseMatrix version of M which is a copy 
        // by reference and not a copy by value of the orignal matrix.
        //
        //-------------------------------------------------------------
        
        BLAS::Matrix< real >  DM_BLAS_restricted( DM->blas_rmat(),
                                                  intersect_row_is - M->row_ofs(),
                                                  intersect_col_is - M->col_ofs() );
        
        unique_ptr< TDenseMatrix > DM_restricted( new TDenseMatrix( intersect_row_is, intersect_col_is, DM_BLAS_restricted ) );
        
        return DM_restricted.release();    
        
    }// else if
    else if ( IS_TYPE( M, TRkMatrix ) )
    {
        /////////////////////////////////////////////////////////////////////////////////////// 
        //
        //
        // Case: TRkMatrix
        //
        //
        ///////////////////////////////////////////////////////////////////////////////////////
    
        auto  RM = cptrcast( M, TRkMatrix );
        
        //---------------------------------------------------------------------
        //
        // Restrict the factors A and B of the original matrix M = A * B^{T}. 
        // Use copy by reference and not a copy by value of the orignal matrix.
        //
        //---------------------------------------------------------------------
        
        const size_t rank = RM->rank();
        
        BLAS::Matrix< real >  A_BLAS_restricted( RM->blas_rmat_A(),
                                                 intersect_row_is - M->row_ofs(),
                                                 BLAS::Range( 0, rank-1 ) );
                                
        BLAS::Matrix< real >  B_BLAS_restricted( RM->blas_rmat_B(),
                                                 intersect_col_is - M->col_ofs(),
                                                 BLAS::Range( 0, rank-1 ) );
                                
        unique_ptr< TRkMatrix > RM_restricted( new TRkMatrix( intersect_row_is, intersect_col_is, A_BLAS_restricted, B_BLAS_restricted ) );
            
        /// Not necessary because the constructor of TRkMatrix already sets the rank to A.ncols()
        RM_restricted->set_rank( rank );
        
        return RM_restricted.release();
        
    }// else if
    else if ( IS_TYPE( M, TBlockMatrix ) )
    {
    
        /////////////////////////////////////////////////////////////////////////////////////// 
        //
        //
        // Case: TBlockMatrix
        //
        //
        ///////////////////////////////////////////////////////////////////////////////////////
          
        auto            BM = cptrcast( M, TBlockMatrix ); 
        const TMatrix * BM_sub;
        
        //-------------------------------------------------------------------------------------
        //
        // Determine the block structure of the restricted matrix 
        //
        //-------------------------------------------------------------------------------------
        
        //
        // Initialise axuilliary values
        //
        size_t row_index_first = BM->block_rows();
        size_t col_index_first = BM->block_cols();
        size_t row_index_last  = 0;
        size_t col_index_last  = 0;

        size_t number_of_intersections = 0;
        
        for ( uint j = 0; j < BM->block_cols(); j++ )
        {
            for ( uint i = 0; i < BM->block_rows(); i++ )
            {
                BM_sub = BM->block(i,j);
                
                if ( BM_sub == nullptr )
                    continue;
                    
                //-----------------------------------------------------------------------------------------
                // Check if block indexset of BM_sub is intersected with the given block indexset block_is
                //-----------------------------------------------------------------------------------------
                const TIndexSet intersect_sub_row_is = intersect( row_is, BM_sub->row_is() );
                const TIndexSet intersect_sub_col_is = intersect( col_is, BM_sub->col_is() );
                
                bool has_row_intersection = false;
                bool has_col_intersection = false;
                
                if ( intersect_sub_row_is.last() - intersect_sub_row_is.first() + 1 > 0 )
                    has_row_intersection = true;
                    
                if ( intersect_sub_col_is.last() - intersect_sub_col_is.first() + 1 > 0 )
                    has_col_intersection = true;
                    
                //----------------------------------------------------------
                // If an intersection is found adjust the auxilliary values
                //----------------------------------------------------------
                if ( has_row_intersection && has_col_intersection )
                {
                    number_of_intersections++;
                    
                    if ( i > row_index_last )
                        row_index_last = i;
                        
                    if ( i < row_index_first )
                        row_index_first = i;
                        
                    if ( j > col_index_last )
                        col_index_last = j;
                        
                    if ( j < col_index_first )
                        col_index_first = j;
                }// if 
            }// for
        }// for  
        
        
        //-------------------------------------------------------------------------------------
        //
        // Consistency checks
        //
        //-------------------------------------------------------------------------------------
        
        
        const size_t restricted_block_rows = row_index_last - row_index_first + 1;
        const size_t restricted_block_cols = col_index_last - col_index_first + 1;
        
        /*
        if ( false )
        {
            if ( number_of_intersections < 1 )
                HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "this should not happen" );
            
            if ( row_index_last < row_index_first )
                HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "" );
                
            if ( col_index_last < col_index_first )
                HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "" );
                
            if ( restricted_block_rows * restricted_block_cols != number_of_intersections )
                HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "this should not happen" );
        }// if
        */
          
        //-------------------------------------------------------------------------------------
        //
        // Initialise the restricted matrix 
        //
        //-------------------------------------------------------------------------------------
          
        TBlockMatrix * root = new TBlockMatrix;
        
        //------------------------------------------------------------
        // TBlockMatrix::copy_struct() directitives which can be kept
        //------------------------------------------------------------
        root->set_id( M->id() );
        root->set_form( M->form() );
        root->set_complex( M->is_complex() );
        root->set_procs( M->procs() );
        
        //--------------------------------------------------------------------
        // TBlockMatrix::copy_struct() directitives which have to be adjusted
        //--------------------------------------------------------------------
        
        // set block rows and block cols
        root->set_block_struct( restricted_block_rows, restricted_block_cols );
              
              
        //-------------------------------------------------------------------------------------
        //
        // Create the restricted matrix 
        //
        //-------------------------------------------------------------------------------------
          
        for ( uint j = 0; j < root->block_cols(); j++ )
        {
            for ( uint i = 0; i < root->block_rows(); i++ )
            {
                root->set_block( i, j, nullptr );
                
                //-------------------------------------------------------
                // Select corresponding submatrix of the original matrix
                //-------------------------------------------------------
                BM_sub = BM->block( i + row_index_first, j + col_index_first );
                
                if ( BM_sub == nullptr )
                    continue;
                    
                //--------------------------------------
                // Determine the intersected index sets
                //--------------------------------------
                const TIndexSet intersect_sub_row_is = intersect( row_is, BM_sub->row_is() );
                const TIndexSet intersect_sub_col_is = intersect( col_is, BM_sub->col_is() );
                
                //--------------------------------------
                // Consistency Checks
                //--------------------------------------
                
                /*
                if ( false )
                {
                    bool has_row_intersection = false;
                    bool has_col_intersection = false;
                    
                    if ( intersect_sub_row_is.last() - intersect_sub_row_is.first() + 1 > 0 )
                        has_row_intersection = true;
                        
                    if ( intersect_sub_col_is.last() - intersect_sub_col_is.first() + 1 > 0 )
                        has_col_intersection = true;
                        
                    if ( !has_row_intersection || !has_col_intersection )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "this should not happen" );
                        
                }// if 
                */
                    
                //----------------------------------------------------
                // Create the restricted submatrix (copy by reference)
                //----------------------------------------------------

                const TBlockIndexSet block_is_restricted ( intersect_sub_row_is, intersect_sub_col_is );
            
                unique_ptr< TMatrix >  BM_sub_restricted( get_restricted_copy_by_reference( BM_sub, block_is_restricted ) );
                
                /// Note: the TBlockMatrix::set_block routine expects a non const input matrix 
                ///       --> Implementation got a little bit circumstantially 
                ///       --> actually I would like to have all matrices const! Also the output matrix root
                root->set_block( i, j, BM_sub_restricted.release() );
                                    
            }// for
        }// for 
         
         
        //-------------------------------------------------------------------------------------
        //
        // TBlockMatrix::copy_struct() directitives which have to be adjusted
        //
        //-------------------------------------------------------------------------------------
        
        // set row offset and col offset
        const TMatrix * root_sub_first = root->block( 0, 0 );
        
        root->set_ofs( root_sub_first->row_ofs(), root_sub_first->col_ofs() );
        
        // set rows and cols
        size_t n_rows_root = 0;
        size_t n_cols_root = 0;
        
        for ( uint j = 0; j < root->block_cols(); j++ )
            n_cols_root +=  root->block( 0, j )->cols();
            
        for ( uint i = 0; i < root->block_rows(); i++ )
            n_rows_root +=  root->block( i, 0 )->rows();
        
        root->set_size( n_rows_root, n_cols_root );
         
        //-------------------------------------------------------------------------------------
        //
        // Return restricted TBlockMatrix
        //
        //-------------------------------------------------------------------------------------
         
         
        return root;
        
    }// else if 
    else
    {
    
        /////////////////////////////////////////////////////////////////////////////////////// 
        //
        //
        // No other case should occur 
        //
        //
        ///////////////////////////////////////////////////////////////////////////////////////
    
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_restricted_copy_by_reference", "" );
        
    }//else
}



TMatrix *
get_copy_with_TZeromatrices ( const TMatrix *        M,
                              const TBlockIndexSet & block_is )
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "(THAMLS) get_copy_with_TZeromatrices", "argument is nullptr" );
            
    if ( ! M->block_is().is_sub( block_is ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_copy_with_TZeromatrices", "" ); 
        

    if ( block_is.is_sub( M->block_is() ) )
    {
        //////////////////////////////////////////////////////////////////////                  
        //
        // If the given block indexset conforms with the block 
        // indexset of the given matrix then return a copy
        //
        //////////////////////////////////////////////////////////////////////
        
        unique_ptr< TMatrix > M_copy( M->copy() );
        
        return M_copy.release();
        
    }// if
    else if ( M->is_type( TYPE_ID( TBlockMatrix) ) )
    {
        //////////////////////////////////////////////////////////////////////                  
        //
        // If M is a TBlockMatrix then replace all subblocks of M which are  
        // not associated with the block indexset block_is by TZeromatrices
        //
        //////////////////////////////////////////////////////////////////////
    
        auto           BM   = cptrcast( M, TBlockMatrix ); 
        TBlockMatrix * root = new TBlockMatrix;
        
        root->copy_struct_from( BM );
                
        for ( uint j = 0; j < BM->block_cols(); j++ )
        {
            for ( uint i = 0; i < BM->block_rows(); i++ )
            {
                root->set_block( i, j, nullptr );
                
                const TMatrix * BM_sub = BM->block(i,j);
                
                if ( BM_sub == nullptr )
                    continue;
                    
                //
                // Check if block indexset of BM_sub is intersected with the given block indexset block_is
                //
                const TIndexSet intersect_row_is = intersect( block_is.row_is(), BM_sub->row_is() );
                const TIndexSet intersect_col_is = intersect( block_is.col_is(), BM_sub->col_is() );
                
                bool row_intersection_is_empty = true;
                bool col_intersection_is_empty = true;
                
                if ( intersect_row_is.last() - intersect_row_is.first() + 1 > 0 )
                    row_intersection_is_empty = false;
                    
                if ( intersect_col_is.last() - intersect_col_is.first() + 1 > 0 )
                    col_intersection_is_empty = false;
                    
                //
                // If there is no intersection set the son to a TZeroMatrix
                //         
                if ( row_intersection_is_empty || col_intersection_is_empty )
                {
                    unique_ptr< TZeroMatrix >  BM_sub_zero( new TZeroMatrix( BM_sub->rows(), BM_sub->cols() ) );
                    
                    BM_sub_zero->set_ofs( BM_sub->row_ofs(), BM_sub->col_ofs() );
                    
                    root->set_block( i, j, BM_sub_zero.release() );
                }// if 
                else
                {
                    const TBlockIndexSet intersect_block_is( intersect_row_is, intersect_col_is );
                    
                    root->set_block( i, j, get_copy_with_TZeromatrices ( BM_sub, intersect_block_is ) );
                }// else
            }// for
        }// for   
        
        return root;
    }// else if
    else
    {
        if ( false )
        {
            //
            // Why does these case occur
            //
            // if ( M->is_type( TYPE_ID( TBlockMatrix) ) )
            //     cout<<endl<<"TBlockMatrix";
                
            // if ( M->is_type( TYPE_ID( TDenseMatrix) ) )
            //     cout<<endl<<"TDenseMatrix";
                
            // if ( M->is_type( TYPE_ID( TRkMatrix) ) )
            //     cout<<endl<<"TRkMatrix";
                
            // EIGEN_MISC::print_matrix ( M, 2,2,false);
            
            // cout<<endl<<"block_is.row_is().first() = "<<block_is.row_is().first();
            // cout<<endl<<"block_is.row_is().last()  = "<<block_is.row_is().last();
            // cout<<endl;
            // cout<<endl<<"block_is.col_is().first() = "<<block_is.col_is().first();
            // cout<<endl<<"block_is.col_is().last()  = "<<block_is.col_is().last();
            
            HERROR( ERR_CONSISTENCY, "(THAMLS) get_copy_with_TZeromatrices", "" );
        }// if
    
        /// Im Grunde passiert hier ja auch wieder die Fallunterscheidung, die ich am
        /// Anfang von comp_M_red_ij habe, das heisst eventuell muss ich Rank-k Matrizen einkuerzen
        /// oder auch TDenseMatrizen einkuerzen.
    
        unique_ptr< TMatrix > M_copy( M->copy() );
        
        return M_copy.release();
    }// else
}




}// namespace anonymous


//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!/////
//!/////
//!/////
//!/////
//!/////
//!/////                                     THAMLS protected
//!/////
//!/////
//!/////
//!/////
//!/////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////









//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Initialisation of H-AMLS
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




void
THAMLS::create_amls_clustertree ( const TClusterTree * ct ) 
{
    if ( ct == nullptr )
        HERROR( ERR_ARG, "(THAMLS) create_amls_clustertree", "argument is nullptr" );
        
    _root_amls_ct = ct->root()->copy();
              
    list< TCluster * >  cluster_list;
    TCluster *          cluster;
    
    
    /////////////////////////////////////////////////////////////////////////////
    //
    // Traverse the cluster tree and delete selected sons to obtain a cluster 
    // tree representing structure of the substructuring applied in AMLS.
    //
    // Delete all sons of clusters representing interfaces and all sons of 
    // clusters representing subdomains whose degrees of freedom are small 
    // enough. 
    //
    /////////////////////////////////////////////////////////////////////////////
   
    cluster_list.push_back( _root_amls_ct ); 
    
    while ( !cluster_list.empty() )
    {
        cluster = cluster_list.front();

        cluster_list.pop_front();
        
        const size_t nsons = cluster->nsons();
        
        
        /////////////////////////////////////////////////
        //
        //
        // Handle the different cases which can occur
        //
        //
        /////////////////////////////////////////////////
        
        
        if ( cluster->is_leaf() )
        {
            // If this case happens the cluster should represent
            // a subdomain or the domain and no sons have to be deleted
            
        }// if
        else if ( cluster->size() <= _para_base.max_dof_subdomain )
        {
            // If this case happens the cluster should represent a subdomain (or the domain) in
            // the AMLS substructuring because it is small enough and no further substructuring 
            // is needed. Correspondingly all sons of this cluster have to be deleted.
                
            for ( uint i = 0; i < nsons; i++ ) 
                cluster->set_son( i, nullptr );
               
            cluster->clear_sons();
        }// else if 
        else if ( nsons == 2 )
        {
            if ( cluster->son(0)->is_domain() && cluster->son(1)->is_domain() )
            {
                // If this case happens the cluster is divided into two subdomains
                // which are seperated in a natural way i.e. without the need of 
                // an additional interface cluster
                
                cluster_list.push_back( cluster->son(0) );
                cluster_list.push_back( cluster->son(1) );
            }// if
            else 
                HERROR( ERR_CONSISTENCY, "(THAMLS) create_amls_clustertree", "" );
        
        }// else  if
        else if ( nsons == 3 )
        {
            if ( cluster->son(0)->is_domain() && cluster->son(1)->is_domain() && !cluster->son(2)->is_domain() )
            {
                // If this case happens the cluster is separated by the classical nested dissection,
                // two subdomain separated by an interface. Append the subdomain clusters to the 
                // cluster list and delete the sons of the interface cluster.
                
                cluster_list.push_back( cluster->son(0) );
                cluster_list.push_back( cluster->son(1) );
                
                TCluster * interface_cluster = cluster->son(2);
                
                for ( uint i = 0; i < interface_cluster->nsons(); i++ )
                    interface_cluster->set_son( i, nullptr );
                       
                interface_cluster->clear_sons();
            }// if
            else
                HERROR( ERR_CONSISTENCY, "(THAMLS) create_amls_clustertree", "" );
                        
        }// else if
        else 
            HERROR( ERR_CONSISTENCY, "(THAMLS) create_amls_clustertree", "" );
    }// while
}




            
void
THAMLS::delete_amls_data () 
{
    const size_t n_subproblems = _sep_tree.n_subproblems();
        
    for ( size_t  i = 0; i < n_subproblems; i++ )
    {
        if ( _D[i] != nullptr )
        {
            delete _D[i];
            
            _D[i] = nullptr;
        }// if
    
        if ( _S[i] != nullptr )
        {
            delete _S[i];
            
            _S[i] = nullptr;
        }// if
    }// for
    
    delete _root_amls_ct;
}

                            
void
THAMLS::init_amls_data ( const TClusterTree * ct )
{
    //-------------------------------------------------
    // Adjust parallel options to number of threads
    //-------------------------------------------------
    if ( CFG::nthreads() == 1 )
        _para_parallel.set_parallel_options( false );
    
    //-------------------------------------------------
    // Initialize the Separator Tree
    //-------------------------------------------------
    create_amls_clustertree ( ct );
    
    _sep_tree = TSeparatorTree( _root_amls_ct );
        
    //-------------------------------------------------
    // Initialize the pointers to the submatrices
    //-------------------------------------------------
    const size_t n_subproblems = _sep_tree.n_subproblems();
    
    _D.resize( n_subproblems );
    _S.resize( n_subproblems );
    
    for ( size_t  i = 0; i < n_subproblems; i++ )
    {
        if ( _D[i] != nullptr )
        {
            delete _D[i];
            
            _D[i] = nullptr;
        }// if
        
        if ( _S[i] != nullptr )
        {
            delete _S[i];
            
            _S[i] = nullptr;
        }// if
    }// for
    
    //-------------------------------------------------
    // Initialise the offsets of the submatrices
    //-------------------------------------------------
    _subproblem_ofs.resize         ( 1 );
    _reduced_subproblem_ofs.resize ( 1 ); 
    
    _subproblem_ofs        [0] = -1;
    _reduced_subproblem_ofs[0] = -1;
}

void
THAMLS::set_subproblem_ofs ()
{
    const size_t n_subproblems = _sep_tree.n_subproblems();
    
    _subproblem_ofs.resize( n_subproblems );
    
    _subproblem_ofs[0] = 0;
    
    for ( size_t i = 1; i < n_subproblems; i++ )
    {
        const size_t size_i_minus_1 = _sep_tree.get_dof_is(i-1).size();
        
        //NOTE: Using 
        //
        //const size_t size_i_minus_1 = _S[i-1]->rows();
        //
        // results in a bug if for example an eigensolution S[i] has zero columns. 
        // In this case S[i] would have as well zero rows which would lead to the 
        // wrong ouput info!
         
        _subproblem_ofs[i] = _subproblem_ofs[i-1] + idx_t( size_i_minus_1 );
    }// for 
}

void
THAMLS::set_reduced_subproblem_ofs ()
{
    const size_t n_subproblems = _sep_tree.n_subproblems();
    
    _reduced_subproblem_ofs.resize( n_subproblems );
    
    _reduced_subproblem_ofs[0] = 0;
    
    for ( size_t i = 1; i < n_subproblems; i++ )
    {
        const size_t size_i_minus_1 = _S[i-1]->cols();
        
        _reduced_subproblem_ofs[i] = _reduced_subproblem_ofs[i-1] + idx_t( size_i_minus_1 );
    }// for 
}

//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Routines for computing the matrices of the transformed problem (K_tilde,M_tilde)
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






void 
THAMLS::transform_problem ( const TMatrix *  K,
                            const TMatrix *  M, 
                            TMatrix * &      K_tilde, 
                            TMatrix * &      M_tilde, 
                            TMatrix * &      L )
{
    if ( M == nullptr || K == nullptr )
        HERROR( ERR_ARG, "(THAMLS) transform_problem", "argument is nullptr" );
    
    if ( !K->is_symmetric() || !M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) transform_problem", "input matrices are not symmetric" ); 
  
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix decomposition K = L * K_tilde * L^T where 'K_tilde' is a block-diagonal 
    // matrix and 'L' is a lower triangular matrix and compute the matrix M_tilde = L^-1 * M * L^-T 
    // 
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    transform_K_version_ldl ( K, K_tilde, L );
                        
    transform_M_version_ldl ( M, M_tilde, L );
}



void
THAMLS::transform_K_version_ldl ( const TMatrix *  K,
                                  TMatrix * &      K_tilde,
                                  TMatrix * &      L )
{ 
    TIC;
    
    //-----------------------------------------------------------------
    // Set factorisation options
    //-----------------------------------------------------------------
    fac_options_t   fac_opts;
    fac_opts.eval = block_wise;
    
    if ( _para_base.coarsen )
        fac_opts.do_coarsen = true;        
        
    //-----------------------------------------------------------------
    // Compute the factorisation
    //-----------------------------------------------------------------
    unique_ptr< TMatrix >  K_factorised( K->copy() ); 
        
    LDL::factorise( K_factorised.get(), _trunc_acc.transform_K, fac_opts );
    
    //-----------------------------------------------------------------
    // Get the matrices 'K_tilde' and 'L'
    //-----------------------------------------------------------------
    LDL::split( K_factorised.get(), L, K_tilde, fac_opts );
    
    K_tilde->set_symmetric();
    
    TOC;
    LOG( to_string("(THAMLS) transform_K_version_ldl : done in %.2fs", toc.seconds() ));
    
    //-----------------------------------------------------------------
    // Derive the corresponding preconditioner of K if wanted
    //-----------------------------------------------------------------
    if ( _para_impro.use_K_inv_as_precomditioner )
    {
        const matform_t  matform = K->form();
        
        //NOTE: See implementation of preconditioner computation in "TEigenArnoldi.cc". The derived 
        //      linear operator is symmetric and thats why it is necessary to set "K_temp" symmetric 
        //      even it isn't in the factorised form. --> see "TEigenArnoldi.cc" for more details
        K_factorised->set_symmetric();
        
        unique_ptr< TLinearOperator >  K_precond( LDL::inv_matrix( K_factorised.get(), matform, fac_opts ) );
        
        K_factorised->set_nonsym();
        
        _para_impro.K_preconditioner = K_precond.release();
        _para_impro.K_factorised     = K_factorised.release();
    }// if 
    
}


void
THAMLS::transform_M_version_ldl ( const TMatrix *  M,
                                  TMatrix * &      M_tilde,
                                  const TMatrix *  L )
{
    TIC;

    /////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix B := L^{-1} * M via solving L * B = M
    //
    /////////////////////////////////////////////////////////////////////////////
    const solve_option_t  solve_L_opts( block_wise, unit_diag );
    
    
    // NOTE: The function 'solve_lower_left' doesn't support symmetric problems! Thats why the
    // implicit symmetric matrix 'M' (upper triangular part of 'M' is nullptr) has to converted 
    // to an explicit symmetric matrix (upper triangular matrix of 'M' has to be filled)
    TMatrix * B = EIGEN_MISC::get_nonsym_of_sym( M );

    solve_lower_left( MATOP_NORM, L, nullptr, B, _trunc_acc.transform_M, solve_L_opts );

    
    /////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix M_tilde = B * L^{-T} via solving M_tilde * L^{T} = B
    //
    /////////////////////////////////////////////////////////////////////////////
    
    solve_lower_right_symmetric( B, MATOP_TRANS, L, nullptr, _trunc_acc.transform_M, solve_L_opts, nullptr );
    
    
    M_tilde = B;
    
    TOC;
    LOG( to_string("(THAMLS) transform_M_version_ldl : done in %.2fs", toc.seconds() ));
}
    


void
THAMLS::transform_M_version_experimantal_1 ( const TMatrix *  M,
                                        TMatrix * &      M_tilde,
                                        const TMatrix *  L )
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Decompose the symmetric matrix M in M =: M_half + M_half^{T} with lower triangular matrix M_half
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    unique_ptr< TMatrix >  M_half( EIGEN_MISC::get_halve_matrix( M ) );
    
    ///////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix B := L^{-1} * M_half via solving L * B = M_half
    //
    ///////////////////////////////////////////////////////////////////////
    const solve_option_t  solve_L_opts( block_wise, unit_diag );
    
    solve_lower_left( MATOP_NORM, L, nullptr, M_half.get(), _trunc_acc.transform_M, solve_L_opts );
    
    unique_ptr< TMatrix > B = std::move( M_half );
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix M_tilde_half: = B * L^{-T} via solving M_tilde_half * L^{T} = B
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    
    // NOTE: Provide the right matrix format for the result M_tilde_half. The problem is that B is a lower 
    // triangular matrix whose upper triangular is represented by Nullptr. However, the product B * L^{-T}  
    // is no longer a lower triangular matrix, it has a lower and upper triangular part. Due to implementation 
    // reasons of the routine 'solve_lower_right' the upper triangular part of the matrix B has to be represented 
    // by submatrices with zero-valued entries 
    unique_ptr< TMatrix > M_tilde_half( EIGEN_MISC::get_full_format_of_lower_triangular( B.get() ) );
    
    solve_lower_right( M_tilde_half.get(), MATOP_TRANS, L, nullptr, _trunc_acc.transform_M, solve_L_opts, nullptr );
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix M_tilde = M_tilde_half + (M_tilde_half)^{T}
    //
    // Note: The result is a symmetric matrix (with upper triangular part represented by
    //       nullptr, only the lower triangular part of both summands has to be added
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    unique_ptr< TMatrix > M_tilde_half_trans( M_tilde_half->copy() );
    
    M_tilde_half_trans->transpose();
    
    EIGEN_MISC::delete_upper_block_triangular( M_tilde_half_trans.get() );
    EIGEN_MISC::delete_upper_block_triangular( M_tilde_half.get()       );
        
    add ( real(1), M_tilde_half_trans.get(), real(1), M_tilde_half.get(), _trunc_acc.transform_M );
        
    M_tilde = M_tilde_half.release();
    
    M_tilde->set_symmetric();
}



void
THAMLS::transform_M_version_experimantal_2 ( const TMatrix *  M,
                                       TMatrix * &      M_tilde,
                                       const TMatrix *  L )
{
    ////////////////////////////////////////////////////////////
    //
    // Compute the Cholesky factorisation  M = L_M * (L_M)^T 
    //    
    ////////////////////////////////////////////////////////////
    unique_ptr< TMatrix > L_M( M->copy() );
    
    fac_options_t  fac_opts_M;

    fac_opts_M.eval = point_wise;
    
    if ( _para_base.coarsen )
        fac_opts_M.do_coarsen = true; 
    
    LL::factorise( L_M.get(), _trunc_acc.transform_M, fac_opts_M );
    

    //////////////////////////////////////////////////////////////////
    //
    // Compute the matrix A := L^-1 * L_M via solving L * A = L_M
    //
    //////////////////////////////////////////////////////////////////
    L_M->set_nonsym();
    
    EIGEN_MISC::delete_upper_block_triangular( L_M.get() ); //make sure, that L_M is lower triangular matrix
    
    const solve_option_t solve_L_opts( block_wise, unit_diag );
    
 
    solve_lower_left( MATOP_NORM, L, nullptr, L_M.get(), _trunc_acc.transform_M, solve_L_opts );
 
    
    ////////////////////////////////////////////////////////////////
    //
    // Compute the matrix M_tilde = A * A^ T with A = L^-1 * L_M
    // 
    ////////////////////////////////////////////////////////////////
    
    ///TODO: Die Multiplikation kann noch optimiert werden. Da wir bereits wissen, dass das 
    /// Resultat symmetrisch ist, muss nicht die komplette Matrix berechnet werden. 
    /// (eventuell reicht es bereits aus, dass obere Dreicksmatrix der Zielmatrix nullptr ist, dass
    /// diese Komponenten erst garnicht berechnet werden.)
    M_tilde = M->copy().release();
    
 
    multiply( real(1), MATOP_NORM, L_M.get(), MATOP_TRANS, L_M.get(), real(0), M_tilde, _trunc_acc.transform_M );   
}






//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Load/Save matrices of the tranformed problem (K_tilde,M_tilde)
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void
THAMLS::save_transformed_problem( const TMatrix *  K_tilde, 
                                  const TMatrix *  M_tilde, 
                                  const TMatrix *  L ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr || L == nullptr )
        HERROR( ERR_ARG, "(THAMLS) save_transformed_problem", "argument is nullptr" );
        
    THLibMatrixIO   mio;
    
    const string location = _para_base.io_location;
    const string prefix   = _para_base.io_prefix;
    
    TIC;
    
    mio.write( K_tilde, location + prefix + "K_tilde.hm" );
    mio.write( M_tilde, location + prefix + "M_tilde.hm" );
    mio.write( L      , location + prefix + "L.hm" );
    
    TOC;
    
    LOG( "" );
    LOG( to_string("(THAMLS) save_transformed_problem : done in %.2fs", toc.seconds() ));
    LOGHLINE;
    LOG( "    L       saved in '" + _para_base.io_location + _para_base.io_prefix + "L.hm'");
    LOG( "    K_tilde saved in '" + _para_base.io_location + _para_base.io_prefix + "K_tilde.hm'");
    LOG( "    M_tilde saved in '" + _para_base.io_location + _para_base.io_prefix + "M_tilde.hm'");
    LOGHLINE;
}


void
THAMLS::load_transformed_problem( TMatrix * &  K_tilde, 
                                  TMatrix * &  M_tilde, 
                                  TMatrix * &  L ) 
{        
    THLibMatrixIO   mio;
    
    const string location = _para_base.io_location;
    const string prefix   = _para_base.io_prefix;
    
    TIC;
    
    L       = mio.read( location + prefix + "L.hm" ).release();
    K_tilde = mio.read( location + prefix + "K_tilde.hm" ).release();
    M_tilde = mio.read( location + prefix + "M_tilde.hm" ).release();
    
    TOC;
    
    LOG( "" );
    LOG( to_string("(THAMLS) load_transformed_problem : done in %.2fs", toc.seconds() ));
    LOGHLINE;
    LOG( "    L       loaded from '" + _para_base.io_location + _para_base.io_prefix + "L.hm'");
    LOG( "    K_tilde loaded from '" + _para_base.io_location + _para_base.io_prefix + "K_tilde.hm'");
    LOG( "    M_tilde loaded from '" + _para_base.io_location + _para_base.io_prefix + "M_tilde.hm'");
    LOGHLINE;
}


//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Routines for computing the partial eigensolutions of (K_tilde_ii, M_tilde_ii)
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void
THAMLS::get_subproblem_matrix ( const TMatrix *    M,
                                const idx_t        i,
                                const TMatrix * &  M_ii ) const
{
    if ( M == nullptr )
        HERROR( ERR_ARG, "(THAMLS) get_subproblem_matrix", "argument is nullptr" );    
    
    const TIndexSet is = _sep_tree.get_dof_is( i );
        
    //////////////////////////////////////////////////////////
    //
    // Search the smallest submatrices of M which contain
    // the given indexset as row and column indexset
    //    
    //////////////////////////////////////////////////////////
    const TMatrix * M_sub = search_submatrix ( M, is, is );
    
    //////////////////////////////
    //
    // Eleminate 'trivial sons'
    //
    //////////////////////////////
    
    while ( true )
    {
        if ( IS_TYPE( M_sub, TBlockMatrix ) )
        {
            auto  BM_sub = cptrcast( M_sub, TBlockMatrix );
          
            if ( BM_sub->block_rows() == 1 && BM_sub->block_cols() == 1 )
                M_sub = BM_sub->block(0,0);
            else
                break;
        }// if
        else
            break;
            
    }// while
        
    ///////////////////////////////
    //
    // Do some consistency checks 
    //
    ///////////////////////////////
    M_ii = M_sub;
    
    if ( M_ii->row_is() != is || M_ii->col_is() != is )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subproblem_matrix", "is are different" ); 
}


size_t
THAMLS::get_number_of_wanted_eigenvectors ( const subproblem_t  kind_of_subproblem,
                                            const size_t        dof ) const
{
    if ( get_mode_selection() != MODE_SEL_AUTO_H )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_number_of_wanted_eigenvectors", "" );
        
    real factor   = real(0);
    real exponent = real(0);
        
    if ( kind_of_subproblem == SUBDOMAIN_SUBPROBLEM )
    {
        exponent = _para_mode_sel.exponent_subdomain;
        
        factor   = _para_mode_sel.factor_subdomain;
    }// if
    else if ( kind_of_subproblem == INTERFACE_SUBPROBLEM )
    {
        exponent = _para_mode_sel.exponent_interface;
        
        factor   = _para_mode_sel.factor_interface;
    }//else if
    else if ( kind_of_subproblem == CONDENSE_SUBPROBLEM )
    {
        exponent = _para_mode_sel.exponent_subdomain;
        
        factor   = _para_mode_sel.factor_condense;
    }// else if 
    else
        HERROR( ERR_ARG, "(THAMLS) get_number_of_wanted_eigenvectors", "" );
        
    
    size_t number_of_eigenvectors = int( factor * Math::pow( real( dof ), exponent ) );
    
    if ( number_of_eigenvectors < 1 )
        number_of_eigenvectors = 1;
    
    if ( number_of_eigenvectors > dof )
        number_of_eigenvectors = dof;
        
    return number_of_eigenvectors;
}
    

void
THAMLS::configure_lapack_eigensolver_subproblem( TEigenLapack &     eigensolver, 
                                                 const size_t        n,
                                                 const subproblem_t  kind_of_subproblem ) const
{
    const mode_selection_t strategy = get_mode_selection();

    if ( strategy == MODE_SEL_BOUND_H )    
    {
        eigensolver.set_ev_selection( EV_SEL_BOUND );
        
        eigensolver.set_ubound( get_trunc_bound() );
        
        //
        // The lower bound is set to 'lower bound = -get_trunc_bound()' because
        // if it is set to 'lower bound = 0' only eigenvalues bigger than zero 
        // are computed. But this would cause some inconsistencies in the later 
        // analysis of the relative errors of eigenvalues (cf. Notizheft 8, S. 33)
        //
        eigensolver.set_lbound( -real(1) * get_trunc_bound() );
    }// if 
    else if ( strategy == MODE_SEL_ABS_H || strategy == MODE_SEL_REL_H )
    {
        eigensolver.set_ev_selection( EV_SEL_INDEX );
        
        size_t k;
        
        //
        // Determine the number of eigenvalues which SHOULD be computed 
        //
        if ( strategy == MODE_SEL_ABS_H )
            k = get_abs();
        
        if ( strategy == MODE_SEL_REL_H )
            k = size_t( get_rel() * n );
        
        if ( k < _para_mode_sel.nmin_ev )
            k = _para_mode_sel.nmin_ev;
        
        //
        // Determine the number of eigenvalues which WILL be computed 
        //
        idx_t number_ev_wanted;
        
        if ( k == 0 )
            number_ev_wanted = 1;
        else if ( k > n )
            number_ev_wanted = idx_t( n );
        else
            number_ev_wanted = idx_t( k );
        
        eigensolver.set_lindex( 1 );
        eigensolver.set_uindex( number_ev_wanted );
    }// else if
    else if ( strategy == MODE_SEL_AUTO_H )
    {
        eigensolver.set_ev_selection( EV_SEL_INDEX );
        
        //
        // Determine the number of wanted eigenpairs
        //
        const size_t k = get_number_of_wanted_eigenvectors( kind_of_subproblem, n );
         
        eigensolver.set_lindex( 1 );
        eigensolver.set_uindex( idx_t(k) );
    }// else if
    else 
    {
        eigensolver.set_ev_selection( EV_SEL_FULL );
    }// else 
}




bool
THAMLS::is_small_problem( const idx_t  i ) const
{
    bool is_small = false;
    
    const size_t  problem_size = _sep_tree.get_dof_is( i ).size();
    
    if ( problem_size <= _para_base.max_dof_subdomain )
        is_small = true;
               
    return is_small;
}
            
            
            
size_t 
THAMLS::eigen_decomp_small_interface_problem ( const TMatrix *  K_tilde_ii,
                                               const TMatrix *  M_tilde_ii,
                                               TDenseMatrix  *  D_i,
                                               TDenseMatrix  *  S_i,
                                               const size_t     i )
{
    if ( M_tilde_ii == nullptr || K_tilde_ii == nullptr || D_i == nullptr || S_i == nullptr )
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_small_interface_problem", "argument is nullptr" );
        
    if ( _sep_tree.is_domain(i) || !is_small_problem( i ) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_small_interface_problem", "" ); 
        
    //////////////////////////////////////////////////////////////////
    //
    // Configure the lapack eigensolver for the subproblem 
    // according to the truncation strategy
    //
    //////////////////////////////////////////////////////////////////
    
    TEigenLapack eigen_solver;
    
    configure_lapack_eigensolver_subproblem( eigen_solver, K_tilde_ii->rows(), INTERFACE_SUBPROBLEM );
    
    // Apply tests if wanted
    eigen_solver.set_test_pos_def  ( _para_base.do_debug );    
    eigen_solver.set_test_residuals( _para_base.do_debug );

    size_t number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    
    const size_t n = K_tilde_ii->rows();
    
    //////////////////////////////////////////////////////////////////
    //
    // Make sure that enough eigenvalues are computed
    //
    //////////////////////////////////////////////////////////////////
    if ( MODE_SEL_BOUND_H == get_mode_selection() && number_ev < _para_mode_sel.nmin_ev && number_ev < n )
    {
        size_t min = _para_mode_sel.nmin_ev;
        
        if ( min > n )
            min = n;
    
        eigen_solver.set_ev_selection( EV_SEL_INDEX );
        eigen_solver.set_lindex( 1 );
        eigen_solver.set_uindex( idx_t(min) );
    
        number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    }// if
            
    ///////////////////////////////////////////////////////////////////////////
    //
    // Make a note if the eigensolution is exact or approximatively
    //
    // In benchmarks it has been observed that LAPAACK solver also 
    // eigenpairs with a relative residual of the order 10^{-4}. To be on  
    // the safe side we assume that eigensolution is not numerically exact
    //
    ///////////////////////////////////////////////////////////////////////////
    _sep_tree.set_exact_eigensolution( i, false );


    return number_ev;
}


           
            
size_t 
THAMLS::eigen_decomp_large_interface_problem ( const TMatrix *  K_tilde_ii,
                                               const TMatrix *  M_tilde_ii,
                                               TDenseMatrix  *  D_i,
                                               TDenseMatrix  *  S_i,
                                               const size_t     i )
{
    if ( M_tilde_ii == nullptr || K_tilde_ii == nullptr || D_i == nullptr || S_i == nullptr )
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_large_interface_problem", "argument is nullptr" );
        
    if ( _sep_tree.is_domain(i) || is_small_problem( i )  )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_large_interface_problem", "" ); 
        
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Using ARPACK or the Arnoldi method the general EVP (K_tilde_ii,M_tilde_ii) is transformed into
    // a standard eigenvalue problem. If in task (T2) the LDL^T decomposition is applied and K_tilde 
    // is (nearly) a diagonal matrix then the transformation the general EVP (K_tilde_ii,M_tilde_ii)
    // into a standard EVP is quite cheap because of the (nearly ) diagonal structure of K_tilde_ii.
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    
    const size_t n_ev_wanted = get_number_of_wanted_eigenvectors( INTERFACE_SUBPROBLEM, K_tilde_ii->rows() );
    
    size_t number_ev;
    
    if ( _para_base.use_arnoldi )
    {
        TEigenArnoldi  eigen_solver;
        
        eigen_solver.set_print_info         ( false );
        eigen_solver.set_n_ev_searched      ( n_ev_wanted );
        eigen_solver.set_transform_symmetric( true );
        eigen_solver.set_test_pos_def       ( _para_base.do_debug );    
        eigen_solver.set_test_residuals     ( _para_base.do_debug );
        
        number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    }// if
    else
    {
        TEigenArpack  eigen_solver;
        
        eigen_solver.set_print_info         ( false );
        eigen_solver.set_n_ev_searched      ( n_ev_wanted );
        eigen_solver.set_transform_symmetric( true );
        eigen_solver.set_test_pos_def       ( _para_base.do_debug );    
        eigen_solver.set_test_residuals     ( _para_base.do_debug );
        
        number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    }// else
    
    //////////////////////////////////////////////////////////////////
    //
    // Make a note if the eigensolution is exact or approximatively
    //
    //////////////////////////////////////////////////////////////////
    _sep_tree.set_exact_eigensolution( i, false );
    
    
    return number_ev;
}
            
            
            
            
size_t 
THAMLS::eigen_decomp_subdomain_problem ( const TMatrix *  K_tilde_ii,
                                         const TMatrix *  M_tilde_ii,
                                         TDenseMatrix  *  D_i,
                                         TDenseMatrix  *  S_i,
                                         const size_t     i )
{
    if ( M_tilde_ii == nullptr || K_tilde_ii == nullptr || D_i == nullptr || S_i == nullptr )
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_subdomain_problem", "argument is nullptr" );
        
    if ( ! _sep_tree.is_domain(i) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_subdomain_problem", "" ); 
        
    size_t number_ev;
    
    //////////////////////////////////////////////////////////////////
    //
    // Configure the lapack eigensolver for the subproblem 
    // according to the truncation strategy
    //
    //////////////////////////////////////////////////////////////////
    TEigenLapack eigen_solver;
    
    configure_lapack_eigensolver_subproblem( eigen_solver, K_tilde_ii->rows(), SUBDOMAIN_SUBPROBLEM );
    

    // Apply tests if wanted
    eigen_solver.set_test_pos_def  ( _para_base.do_debug );    
    eigen_solver.set_test_residuals( _para_base.do_debug );
        
    
    number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    
    const size_t n = K_tilde_ii->rows();
    
    //////////////////////////////////////////////////////////////////
    //
    // Make sure that enough eigenvalues are computed
    //
    //////////////////////////////////////////////////////////////////
    if ( MODE_SEL_BOUND_H == get_mode_selection() && number_ev < _para_mode_sel.nmin_ev && number_ev < n )
    {
        size_t min = _para_mode_sel.nmin_ev;
        
        if ( min > n )
            min = n;
    
        eigen_solver.set_ev_selection( EV_SEL_INDEX );
        eigen_solver.set_lindex( 1 );
        eigen_solver.set_uindex( idx_t(min) );
    
        number_ev = eigen_solver.comp_decomp( K_tilde_ii, M_tilde_ii, D_i, S_i );
    }// if
    
    ///////////////////////////////////////////////////////////////////////////
    //
    // Make a note if the eigensolution is exact or approximatively
    //
    // In benchmarks it has been observed that LAPAACK solver also 
    // eigenpairs with a relative residual of the order 10^{-4}. To be on  
    // the safe side we assume that eigensolution is not numerically exact
    //
    ///////////////////////////////////////////////////////////////////////////
    _sep_tree.set_exact_eigensolution( i, false );
    
    
    return number_ev;
}


void
THAMLS::comp_partial_eigensolutions ( const TMatrix *    K_tilde,
                                      const TMatrix *    M_tilde ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_partial_eigensolutions", "argument is nullptr" );
            
    TIC;
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the partial eigendecompostion of the fixed interface eigenvalue problems (K_ii,M_ii) 
    // and of the coupling mode eigenvalue problems (K_tilde_ii, M_tilde_ii) (cf. [Bennighof])
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    struct job_t
    {
        bool do_subdomain       = true;
        bool do_small_interface = true;
        bool do_large_interface = true;
    } job_todo;

    auto comp_partial_eigensol = 
    [ this, K_tilde, M_tilde, job_todo ]  ( const tbb::blocked_range< uint > & r )
    {
        for ( auto  j = r.begin(); j != r.end(); ++j )
        {
            const idx_t i = idx_t(j);
            
            //-------------------------------------------------------------------------
            // Check if something has to be computed
            //-------------------------------------------------------------------------
            bool do_nothing = true;
            
            if ( _sep_tree.is_domain( i ) && job_todo.do_subdomain )
                do_nothing = false;
            
            if ( !_sep_tree.is_domain( i ) && is_small_problem( i ) && job_todo.do_small_interface )
                do_nothing = false;
            
            if ( !_sep_tree.is_domain( i ) && !is_small_problem( i ) && job_todo.do_large_interface )
                do_nothing = false;
            
            if ( do_nothing )
                continue;
            
            //-------------------------------------------------------------------------
            // Get the subproblems (K_ii,M_ii) or respectively (K_tilde_ii, M_tilde_ii)
            //-------------------------------------------------------------------------
            const TMatrix * K_tilde_ii = nullptr;
            const TMatrix * M_tilde_ii = nullptr;
            
            get_subproblem_matrix( K_tilde, i, K_tilde_ii );
            get_subproblem_matrix( M_tilde, i, M_tilde_ii );
            
            //-------------------------------------------------------------
            // Compute the (partial) eigen decomposition of the subproblems
            //-------------------------------------------------------------
            _D[i] = new TDenseMatrix;
            _S[i] = new TDenseMatrix;
                
            try
            {  
                //-------------------------------------
                // Do some consistency checks if wanted
                //-------------------------------------
                if ( _para_base.do_debug )
                {
                    if ( EIGEN_MISC::has_inf_nan_entry ( K_tilde_ii ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "K_tilde_ii has bad data" );
                        
                    if ( EIGEN_MISC::has_inf_nan_entry ( M_tilde_ii ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "M_tilde_ii has bad data" );
                
                    if ( ! EIGEN_MISC::is_pos_def( K_tilde_ii ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "K_tilde_ii is not positive definit" );
                    
                    if ( ! EIGEN_MISC::is_pos_def( M_tilde_ii ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "M_tilde_ii is not positive definit" );    
                }// if
    
                //---------------------------------
                // Choose corresponding eigensolver
                //---------------------------------
                if ( _sep_tree.is_domain( i ) && job_todo.do_subdomain )
                    eigen_decomp_subdomain_problem( K_tilde_ii, M_tilde_ii, _D[i], _S[i], i );
                
                if ( !_sep_tree.is_domain( i ) && is_small_problem( i ) && job_todo.do_small_interface )
                    eigen_decomp_small_interface_problem( K_tilde_ii, M_tilde_ii, _D[i], _S[i], i );
                
                if ( !_sep_tree.is_domain( i ) && !is_small_problem( i ) && job_todo.do_large_interface )
                    eigen_decomp_large_interface_problem( K_tilde_ii, M_tilde_ii, _D[i], _S[i], i );
                
                //--------------------------------------
                // Do some consistency checks if wanted
                //--------------------------------------
                if ( _para_base.do_debug )
                {
                    if ( EIGEN_MISC::has_inf_nan_entry ( _D[i] ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "_D[i] has bad data" );
                        
                    if ( EIGEN_MISC::has_inf_nan_entry ( _S[i] ) )
                        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "_S[i] has bad data" );
                }// if
            }// try
            catch ( Error & e )
            {
                if ( e.error_code() == ERR_NCONVERGED )
                {    
                    LOG( "(THAMLS) comp_partial_eigensolutions :");
                    LOGHLINE;
                    e.print();
                    LOGHLINE;
                }// if
                else
                    throw;
            }// catch
        }// for
    };
        
    const size_t n_subproblems = _sep_tree.n_subproblems();
    
    if ( _para_parallel.miscellaneous_parallel ) 
    {
        if ( _para_base.use_arnoldi )
            tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_subproblems) ), comp_partial_eigensol );
        else 
        {
            // NOTE: Note, that the LAPACK and Arnoldi eigensolver are thread-safe, however, the ARPACK eigensolver is NOT! 
            //       More precisely: The matrix-vector iteration in ARPACK can be handeled in parallel, however, several 
            //       subproblems cannot be handled in parallel by ARPACK because the ARPACK library is not thread-safe.
            job_todo.do_subdomain       = true;
            job_todo.do_small_interface = true;
            job_todo.do_large_interface = false;
            tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_subproblems) ), comp_partial_eigensol );

            job_todo.do_subdomain       = false;
            job_todo.do_small_interface = false;
            job_todo.do_large_interface = true;            
            comp_partial_eigensol ( tbb::blocked_range< uint >( uint(0), uint(n_subproblems) ) );
        }// else 
            
    }// if
    else 
        comp_partial_eigensol ( tbb::blocked_range< uint >( uint(0), uint(n_subproblems) ) );
        
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Consistency check
    //    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    for ( size_t i = 0; i < n_subproblems; i++ )
    {
        if ( _D[i] == nullptr || _S[i] == nullptr )
            HERROR( ERR_ARG, "(THAMLS) comp_partial_eigensolutions", "argument is nullptr" );
    }// for 
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Set offset data for the subsequent computations
    //    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    set_subproblem_ofs();
    set_reduced_subproblem_ofs();  
    
    TOC; 
    LOG( to_string("(THAMLS) comp_partial_eigensolutions : done in %.2fs", toc.seconds() ));
}



    
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Auxiliary routines for the function 'comp_M_red_ij_aux' which is used for the computation of the 
//!///  matrix M_red_ij = S_i^{T} * M_tilde_ij * S_j and where different features for parallelism are applied
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    





///
/// NOTE: "col_is" has to be a copy not a reference, otherwise an error 
///       occured when parent task is finished but childs are still handled
/// 
class comp_M_red_ij_aux_step_1_task_t : public tbb::task
{
  private:
   const TMatrix *      _M_tilde_ij;
   const TDenseMatrix * _S_j;
   const TIndexSet      _col_is; 
   TDenseMatrix *       _Temp;
   const size_t &       _max_col_size;
   const size_t &       _mul_block_with_dense_max_size;
   
  public:

    comp_M_red_ij_aux_step_1_task_t( const TMatrix *      M_tilde_ij,
                                     const TDenseMatrix * S_j,
                                     const TIndexSet      col_is,
                                     TDenseMatrix *       Temp,
                                     const size_t &       max_col_size,
                                     const size_t &       mul_block_with_dense_max_size )
        : _M_tilde_ij( M_tilde_ij )
        , _S_j( S_j )
        , _col_is( col_is )
        , _Temp( Temp )
        , _max_col_size( max_col_size )
        , _mul_block_with_dense_max_size( mul_block_with_dense_max_size )
    {}

    task * execute()
    {
        if ( _col_is.size() <= _max_col_size )
        {
            //////////////////////////////////////////////////////////////////
            //
            // If the column index set is small enough then compute the
            // corresponding submatrix Temp_sub = M_tilde_ij * S_j_sub 
            //
            //////////////////////////////////////////////////////////////////
        
            //-----------------------------------------------------
            // Create reference to submatrix of S_j
            //-----------------------------------------------------
            BLAS::Matrix< real >  S_j_BLAS_sub ( _S_j->blas_rmat(),
                                                 _S_j->row_is() - _S_j->row_ofs(),
                                                 _col_is        - _S_j->col_ofs() );
        
            unique_ptr< TDenseMatrix > S_j_sub( new TDenseMatrix( _S_j->row_is(), _col_is, S_j_BLAS_sub ) );
            
            //-----------------------------------------------------
            // Create reference to submatrix of Temp
            //-----------------------------------------------------
            BLAS::Matrix< real >  Temp_BLAS_sub ( _Temp->blas_rmat(),
                                                  _Temp->row_is() - _Temp->row_ofs(),
                                                  _col_is         - _Temp->col_ofs() );
        
            unique_ptr< TDenseMatrix > Temp_sub( new TDenseMatrix( _Temp->row_is(), _col_is, Temp_BLAS_sub ) );
            
            //-----------------------------------------------------
            // Compute Temp_sub = M_tilde_ij * S_j_sub
            //-----------------------------------------------------
            if ( ! IS_TYPE( _M_tilde_ij, TBlockMatrix ) )
            {
                multiply( real(1), MATOP_NORM, _M_tilde_ij, MATOP_NORM, S_j_sub.get(), real(0), Temp_sub.get(), acc_exact );
            }// if
            else
            {   
                auto  BM_tilde_ij = cptrcast( _M_tilde_ij, TBlockMatrix );
                
                Temp_sub->scale( real(0) );
                
                mul_block_with_dense_to_dense( MATOP_NORM, BM_tilde_ij, S_j_sub.get(), Temp_sub.get(), _mul_block_with_dense_max_size );
            }// else  
            
            return nullptr;
        }// if
        else
        {
            //////////////////////////////////////////////////////////////////
            //
            // If the column index set is too large then subdivide it into 
            // two parts of nearly equal size and make a recursive call
            //
            //////////////////////////////////////////////////////////////////
            
            const idx_t first = _col_is.first();
            const idx_t last  = _col_is.last();
            const idx_t mid   = idx_t( (last + first)/2 );
            
            TIndexSet is_1( first,        mid  );
            TIndexSet is_2( idx_t(mid+1), last );
            
            
            
            #if 0
            tbb::task & child1 = * new( allocate_child() ) comp_M_red_ij_aux_step_1_task_t( _M_tilde_ij, _S_j, is_1, _Temp, _max_col_size, _mul_block_with_dense_max_size );
            tbb::task & child2 = * new( allocate_child() ) comp_M_red_ij_aux_step_1_task_t( _M_tilde_ij, _S_j, is_2, _Temp, _max_col_size, _mul_block_with_dense_max_size );
            
            set_ref_count( 3 );
            spawn( child2 );
            spawn_and_wait_for_all( child1 );
            
            return nullptr;
            #endif
            
            
                        
            #if 1
            tbb::empty_task & temp = * new( allocate_continuation() ) tbb::empty_task;
            
            //-----------------------------------------------------------------------------------------------------------------------------------------
            // NOTE: The child become the index set as a value and not as a reference, just in case that the parent is destroyed before the child
            //-----------------------------------------------------------------------------------------------------------------------------------------
            tbb::task & child1 = * new( temp.allocate_child() ) comp_M_red_ij_aux_step_1_task_t( _M_tilde_ij, _S_j, is_1, _Temp, _max_col_size, _mul_block_with_dense_max_size );
            tbb::task & child2 = * new( temp.allocate_child() ) comp_M_red_ij_aux_step_1_task_t( _M_tilde_ij, _S_j, is_2, _Temp, _max_col_size, _mul_block_with_dense_max_size );
            
            temp.set_ref_count( 2 );
            
            spawn( child1 );
            
            return & child2;
            #endif
            
        }// else
    }
};





void
THAMLS::comp_M_red_ij_aux_step_1_task ( const TMatrix *      M_tilde_ij,
                                        const TDenseMatrix * S_j,
                                        TDenseMatrix *       Temp ) const
{
    if ( M_tilde_ij == nullptr || S_j == nullptr || Temp == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_M_red_ij_aux_step_1_task", "argument is nullptr" );
        
    ////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the matrix Temp = M_tilde_ij * S_j 
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////
        
    //---------------------------------------------------------
    // Decide if the matrix result 'Temp' should be partitioned 
    // into block columns which are computed in parallel 
    //---------------------------------------------------------
    const size_t max_col_size = _para_parallel.step_1___block_partition_size;
    const size_t max_row_size = _para_parallel.step_1___max_row_size;
            
    bool do_subdivide = true; 
    
    // If number of cols is too small then make no subdivision
    if ( S_j->cols() <= max_col_size )
        do_subdivide = false;
        
    // If number of rows is too small then make no subdivision
    if ( S_j->rows() <= max_row_size )
        do_subdivide = false;
    
    if ( do_subdivide )
    {
        //---------------------------------------------------------------------------------------
        // In order to compute the matrix Temp = M_tilde_ij * S_j in parallel we subdivide 'Temp'
        // and 'S_j' recursively in block columns and compute these block columns in parallel
        //---------------------------------------------------------------------------------------
        const idx_t first = S_j->col_is().first();
        const idx_t last  = S_j->col_is().last();
        
        const TIndexSet col_is( first, last );
        
        
        tbb::task & root = * new( tbb::task::allocate_root() ) 
                comp_M_red_ij_aux_step_1_task_t( M_tilde_ij, S_j, col_is, Temp, max_col_size, _para_parallel.mul_block_with_dense_max_size );

        tbb::task::spawn_root_and_wait( root );
    }// if
    else
    {
        //---------------------------------------------------------------------------------------
        // Do not compute the block columns of the matrix result 'Temp' in parallel
        //---------------------------------------------------------------------------------------
        if ( ! IS_TYPE( M_tilde_ij, TBlockMatrix ) )
        {
            multiply( real(1), MATOP_NORM, M_tilde_ij, MATOP_NORM, S_j, real(0), Temp, acc_exact );
        }// if
        else
        {   
            auto  BM_tilde_ij = cptrcast( M_tilde_ij, TBlockMatrix );
            
            Temp->scale( real(0) );
            
            mul_block_with_dense_to_dense( MATOP_NORM, BM_tilde_ij, S_j, Temp, _para_parallel.mul_block_with_dense_max_size );
        }// else   
    }// else
}
            
     





void
THAMLS::comp_M_red_ij_aux_step_1 ( const TMatrix *      M_tilde_ij,
                                   const TDenseMatrix * S_j,
                                   TDenseMatrix *       Temp ) const
{
    if ( M_tilde_ij == nullptr || S_j == nullptr || Temp == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_M_red_ij_aux_step_1", "argument is nullptr" );
        
    ////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the matrix   Temp := M_tilde_ij * S_j 
    //
    // In order to compute the matrix result in parallel we subdivide the matrix 
    // Temp in block columns which can be computed separately. 
    //
    ////////////////////////////////////////////////////////////////////////////////////
    
    //-----------------------------------------------------
    // Initialise the block structure of the result
    //-----------------------------------------------------
    const size_t max_size = _para_parallel.step_1___block_partition_size;
        
    size_t block_cols = Temp->cols() / max_size;
        
    if ( block_cols < 1 )
        block_cols = 1;
    
    //----------------------------------------------------------------------
    // Compute Temp blockcolumin-wise
    //----------------------------------------------------------------------               
    
    auto  compute_Temp_sub =
        [ this, M_tilde_ij, S_j, Temp, & block_cols, & max_size ] ( const tbb::blocked_range< uint > &  r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //----------------------------------------------------------
                // Determine the col index set of the corresponding subblock
                //----------------------------------------------------------
                const size_t col_first = Temp->col_is().first() + j*max_size;
                
                size_t col_last = Temp->col_is().first() + (j+1)*max_size - 1;
                
                if ( j == block_cols - 1 )
                    col_last = Temp->col_is().last();
                    
                const TIndexSet sub_col_is( col_first, col_last );
                
                //----------------------------------------------------------
                // Create reference to submatrix of Temp
                //----------------------------------------------------------
                BLAS::Matrix< real >  Temp_BLAS_sub ( Temp->blas_rmat(),
                                                      Temp->row_is()  - Temp->row_ofs(),
                                                      sub_col_is      - Temp->col_ofs() );
            
                unique_ptr< TDenseMatrix > Temp_sub( new TDenseMatrix( Temp->row_is(), sub_col_is, Temp_BLAS_sub ) );
                
                //----------------------------------------------------------
                // Create reference to submatrix of S_j
                //----------------------------------------------------------
                BLAS::Matrix< real >  S_j_BLAS_sub ( S_j->blas_rmat(),
                                                     S_j->row_is()  - S_j->row_ofs(),
                                                     sub_col_is     - S_j->col_ofs() );
            
                unique_ptr< TDenseMatrix > S_j_sub( new TDenseMatrix( S_j->row_is(), sub_col_is, S_j_BLAS_sub ) );
                                
                //----------------------------------------------------------
                // Compute Temp_sub = M_tilde_ij * S_j_sub
                //----------------------------------------------------------
                if ( ! IS_TYPE( M_tilde_ij, TBlockMatrix ) )
                {
                    multiply( real(1), MATOP_NORM, M_tilde_ij, MATOP_NORM, S_j_sub.get(), real(0), Temp_sub.get(), acc_exact );
                }// if
                else
                {   
                    auto  BM_tilde_ij = cptrcast( M_tilde_ij, TBlockMatrix );
                    
                    Temp_sub->scale( real(0) );
                    
                    mul_block_with_dense_to_dense( MATOP_NORM, BM_tilde_ij, S_j_sub.get(), Temp_sub.get(), _para_parallel.mul_block_with_dense_max_size );
                }// else   
            }// for
        };
    
      
    //--------------------------------------------------
    // Handle small problems separately in order 
    // to avoid unnecessary parallel overhead 
    //--------------------------------------------------
    
    if ( block_cols == 1 || S_j->rows() < _para_parallel.step_1___max_row_size )
    {   
        if ( ! IS_TYPE( M_tilde_ij, TBlockMatrix ) )
        {
            multiply( real(1), MATOP_NORM, M_tilde_ij, MATOP_NORM, S_j, real(0), Temp, acc_exact );
        }// if
        else
        {   
            auto  BM_tilde_ij = cptrcast( M_tilde_ij, TBlockMatrix );
            
            Temp->scale( real(0) );
            
            mul_block_with_dense_to_dense( MATOP_NORM, BM_tilde_ij, S_j, Temp, _para_parallel.mul_block_with_dense_max_size );
        }// else   
    }// if
    else
    {
        if ( _para_parallel.level_of_affinity_partitioner_usage > 0 )
        {
            affinity_partitioner  ap;
            
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(block_cols) ), compute_Temp_sub, ap );
        }// if
        else
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(block_cols) ), compute_Temp_sub );
        
    }// else
}



void
THAMLS::comp_M_red_ij_aux_step_2 ( const TDenseMatrix * S_i,
                                   const TDenseMatrix * Temp,
                                   TDenseMatrix *       M_red_ij ) const
{
    if ( S_i == nullptr || Temp == nullptr || Temp == M_red_ij )
        HERROR( ERR_ARG, "(THAMLS) comp_M_red_ij_aux_step_2", "argument is nullptr" );
    
    
    ////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the matrix       M_red_ij = S_i^{T} * Temp 
    //
    // In order to compute the matrix result in parallel we subdivide 
    // M_red_ij into several subblocks which can be computed separately
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////
    
    
    //-----------------------------------------------------
    // Initialise the block structure of the result
    //-----------------------------------------------------
    const size_t max_size = _para_parallel.step_2___block_partition_size;
        
    size_t block_rows = M_red_ij->row_is().size() / max_size;
    size_t block_cols = M_red_ij->col_is().size() / max_size;
    
    if ( block_rows < 1 )
        block_rows = 1;
        
    if ( block_cols < 1 )
        block_cols = 1;
    
    //----------------------------------------------------------------------
    // Compute M_red_ij block-wise
    //----------------------------------------------------------------------                    
    
    auto  compute_M_red_ij_sub =
        [ S_i, Temp, M_red_ij, & block_rows, & block_cols, & max_size ] ( const tbb::blocked_range2d< uint > &  r )
        {
            for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
            {
                for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                {
                    //-----------------------------------------------------
                    // Determine the corresponding index sets of the result
                    //-----------------------------------------------------
                    const size_t row_first = M_red_ij->row_is().first() + i*max_size;
                    const size_t col_first = M_red_ij->col_is().first() + j*max_size;
                    
                    size_t row_last = M_red_ij->row_is().first() + (i+1)*max_size - 1;
                    size_t col_last = M_red_ij->col_is().first() + (j+1)*max_size - 1;
                    
                    //NOTE: Es scheint hier wichtig zu sein mit 'block_rows' statt 'r.rows().end()' zu 
                    //      arbeiten, denn bei benchmarks hatte das Letzter nicht so gut parallel skaliert
                    if ( i == block_rows - 1 )
                        row_last = M_red_ij->row_is().last();
                    
                    if ( j == block_cols - 1 )
                        col_last = M_red_ij->col_is().last();
                        
                    const TIndexSet sub_row_is( row_first, row_last );
                    const TIndexSet sub_col_is( col_first, col_last );
                    
                    //-----------------------------------------------------
                    // Create reference to submatrix of Temp
                    //-----------------------------------------------------
                    BLAS::Matrix< real >  Temp_BLAS_sub ( Temp->blas_rmat(),
                                                          Temp->row_is() - Temp->row_ofs(),
                                                          sub_col_is     - Temp->col_ofs() );
                
                    unique_ptr< TDenseMatrix > Temp_sub( new TDenseMatrix( Temp->row_is(), sub_col_is, Temp_BLAS_sub ) );
                    
                    
                    //-----------------------------------------------------
                    // Create reference to submatrix of S_i
                    //-----------------------------------------------------
                    BLAS::Matrix< real >  S_i_BLAS_sub ( S_i->blas_rmat(),
                                                         S_i->row_is() - S_i->row_ofs(),
                                                         sub_row_is    - S_i->col_ofs() );
                
                    unique_ptr< TDenseMatrix > S_i_sub( new TDenseMatrix( S_i->row_is(), sub_row_is, S_i_BLAS_sub ) );
                    
                    
                    //-----------------------------------------------------
                    // Create reference to submatrix of M_red_ij
                    //-----------------------------------------------------
                    BLAS::Matrix< real >  M_red_ij_BLAS_sub( M_red_ij->blas_rmat(),
                                                             sub_row_is - M_red_ij->row_ofs(),
                                                             sub_col_is - M_red_ij->col_ofs() );
                
                    unique_ptr< TDenseMatrix > M_red_ij_sub( new TDenseMatrix( sub_row_is, sub_col_is, M_red_ij_BLAS_sub ) );
                    
                    //-----------------------------------------------------
                    // Compute M_red_ij_sub = S_i_sub^{T} * Temp_sub
                    //-----------------------------------------------------
                    multiply( real(1), MATOP_TRANS, S_i_sub.get(), MATOP_NORM, Temp_sub.get(), real(0), M_red_ij_sub.get(), acc_exact );
                }// for
            }// for
        };
    
    
    
    //--------------------------------------------------
    // Handle small problems separately in order 
    // to avoid unnecessary parallel overhead 
    //--------------------------------------------------
    
    if ( (block_rows == 1 && block_cols == 1) || S_i->rows() < _para_parallel.step_2___max_row_size  )
    {   
        multiply( real(1), MATOP_TRANS, S_i, MATOP_NORM, Temp, real(0), M_red_ij, acc_exact );
    }// if
    else
    { 
        if ( _para_parallel.level_of_affinity_partitioner_usage > 0 )
        {
            affinity_partitioner  ap;
            
             tbb::parallel_for ( tbb::blocked_range2d< uint >( uint(0), uint(block_rows), uint(0), uint(block_cols) ), compute_M_red_ij_sub, ap );
        }// if
        else
            tbb::parallel_for ( tbb::blocked_range2d< uint >( uint(0), uint(block_rows), uint(0), uint(block_cols) ), compute_M_red_ij_sub );
        
    }// else
}





//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Routines for computing the matrices of the reduced problem (K_red, M_red)
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




void
THAMLS::comp_M_red_ij_aux ( const TDenseMatrix * S_i,
                            const TMatrix *      M_tilde_ij,
                            const TDenseMatrix * S_j,
                            TDenseMatrix *       M_red_ij )
{
    if ( S_i == nullptr || M_tilde_ij == nullptr || S_j == nullptr || M_red_ij == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_M_red_ij_aux", "argument is nullptr" );
    
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   M_red_ij = S_i^{T} * M_tilde_ij * S_j  
    //
    //////////////////////////////////////////////////////////////////////////////////////////
    
    unique_ptr< TDenseMatrix > Temp( new TDenseMatrix( M_tilde_ij->row_is(), S_j->col_is() ) );
    
    //------------------------------------------------------------------------
    // Step 1: compute  Temp = M_tilde_ij * S_j 
    //------------------------------------------------------------------------
    if ( _para_parallel.M_red_ij_step_1_in_parallel )
    {
        if ( _para_parallel.level_of_affinity_partitioner_usage > 0 )
            comp_M_red_ij_aux_step_1 ( M_tilde_ij, S_j, Temp.get() ); 
        else
            comp_M_red_ij_aux_step_1_task ( M_tilde_ij, S_j, Temp.get() );
    }// if 
    else
        multiply( real(1), MATOP_NORM, M_tilde_ij, MATOP_NORM, S_j, real(0), Temp.get(), acc_exact );
        
    //------------------------------------------------------------------------
    // Step 2: compute   M_red_ij = S_i^{T} * Temp  
    //------------------------------------------------------------------------
    if ( _para_parallel.M_red_ij_step_2_in_parallel )
        comp_M_red_ij_aux_step_2 ( S_i, Temp.get(), M_red_ij );
    else
        multiply( real(1), MATOP_TRANS, S_i, MATOP_NORM, Temp.get(), real(0), M_red_ij, acc_exact );
        
}






void
THAMLS::comp_M_red_ij ( const TMatrix *   M_tilde,
                        const idx_t       i,
                        const idx_t       j,
                        TDenseMatrix *    M_red_ij ) 
{
    if ( M_tilde == nullptr || M_red_ij == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_M_red_ij", "argument is nullptr" );
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Search the smallest submatrix of M_tilde which contains the given row and column indexset
    //
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////
    const TIndexSet row_is = _sep_tree.get_dof_is( i );
    const TIndexSet col_is = _sep_tree.get_dof_is( j );
    
    const TMatrix * M_tilde_ij = search_submatrix( M_tilde, row_is, col_is );
    
    //--------------------------------------------------------------------------
    // Eleminate 'trivial sons'
    //--------------------------------------------------------------------------
    while ( true )
    {
        if ( IS_TYPE( M_tilde_ij, TBlockMatrix ) )
        {
            auto  BM_tilde_ij = cptrcast( M_tilde_ij, TBlockMatrix );
          
            if ( BM_tilde_ij->block_rows() == 1 && BM_tilde_ij->block_cols() == 1 )
                M_tilde_ij = BM_tilde_ij->block(0,0);
            else
                break;
        }// if
        else
            break;
            
    }// while
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    //
    // Compute M_red_ij = S_i^{T} * M_tilde_ij * S_j 
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////    
    
    if ( row_is == M_tilde_ij->row_is() && col_is == M_tilde_ij->col_is() )  
    {
    
        comp_M_red_ij_aux ( _S[i], M_tilde_ij, _S[j], M_red_ij );
        
    }// if 
    else 
    {
        //////////////////////////////////////////////////////////////////////////////////////////
        //
        //
        // If the block indexset row_is x col_is is a real subset of the block index set 
        // associated to the matrix M_tilde_ij then we have to restrict M_tilde_ij further
        //
        //
        //////////////////////////////////////////////////////////////////////////////////////////
    
        //----------------------------------------------------------------------------------------
        // Create restricted version of M_tilde_ij using only copies by reference and NOT copies
        // by value in order to spare memory bandwidth. This is getting important when many cores 
        // are used in a shared memory system and memory bandwidth is becoming a bottleneck of 
        // the computer system architecture and preventing further scalability.
        //----------------------------------------------------------------------------------------        
        const TBlockIndexSet block_is( row_is, col_is );
        
        unique_ptr< TMatrix >  M_tilde_ij_restricted( get_restricted_copy_by_reference( M_tilde_ij, block_is ) );
                
        //////////////////////////////////////////////////////////////////////////////////////////
        //
        // Compute M_red_ij = S_i^{T} * M_tilde_ij_restricted * S_j   
        //
        //////////////////////////////////////////////////////////////////////////////////////////
        comp_M_red_ij_aux ( _S[i], M_tilde_ij_restricted.get(), _S[j], M_red_ij );       
    }// else
}





size_t 
THAMLS::comp_K_and_M_red_parallel ( const TMatrix *  K_tilde,
                                    const TMatrix *  M_tilde,
                                    TDenseMatrix *   K_red,
                                    TDenseMatrix *   M_red )
{
    unique_ptr< subprob_data_t > subprob_data( new subprob_data_t );
    
    subprob_data->first = 0;
    subprob_data->last  = _sep_tree.n_subproblems() - 1;
    
    const size_t n_jobs = comp_K_and_M_red_parallel ( K_tilde, M_tilde, K_red, M_red, subprob_data.get() );
    
    return n_jobs;
}



void
THAMLS::comp_reduced_matrices ( const TMatrix *    K_tilde,
                                const TMatrix *    M_tilde,
                                TDenseMatrix *     K_red,
                                TDenseMatrix *     M_red )  
{
    if ( K_tilde == nullptr || M_tilde == nullptr || K_red == nullptr || M_red == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_reduced_matrices", "argument is nullptr" );
        
        
    ///////////////////////////////////////////////////////////////////////////
    //
    // Construct the reduced eigenvalue problem (K_red, M_red) received by
    // modal truncation of the eigenvalue problem (K, M) (cf. [Bennighof])
    //
    ///////////////////////////////////////////////////////////////////////////

    TIC;
    comp_K_and_M_red_parallel( K_tilde, M_tilde, K_red, M_red );
    TOC;
    
    ///////////////////////////////////////////////////////////////////////////
    //
    // NOTE: Set IDs of reduced matrices. If this is not done it can happen 
    // that an error message is created when matrix algebra routines are 
    // applied to the reduced matrices, e.g., LDL-decomposition of M_red
    //
    ///////////////////////////////////////////////////////////////////////////
    K_red->set_id( K_tilde->id() );
    M_red->set_id( M_tilde->id() );
    
    LOG( to_string("(THAMLS) comp_reduced_matrices : done in %.2fs", toc.seconds() ));
}




//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Eigendecompostion routines for the reduced eigenvalue problem (K_red,M_red)
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



bool 
THAMLS::adjust_MKL_threads_for_eigen_decomp ( const size_t n ) const
{
    bool use_multiple_MKL_threads = false;
    
    const uint used_threads = CFG::nthreads();
    
    if ( (n > _para_parallel.max_size_EVP_for_seqential_MKL) && (used_threads > 1) )
    {
        //--------------------------------------------------------
        // Set the number of threads used in the MKL subroutines
        // to the maximal possible number of threads
        //--------------------------------------------------------

        #if USE_MKL == 1
        mkl_set_num_threads( CFG::nthreads() );
        use_multiple_MKL_threads = true;
        
        LOG( "" );
        LOG( to_string("(THAMLS) adjust_MKL_threads_for_eigen_decomp :"));
        LOGHLINE;
        LOG( to_string("    number of MKL threads was set to maximum") );
        LOG( to_string("    CFG::nthreads()       = %d",CFG::nthreads()) );
        LOG( to_string("    mkl_get_max_threads() = %d",mkl_get_max_threads()) );
        LOGHLINE;
        
        #endif
    }// if
    
    return use_multiple_MKL_threads;
}

void
THAMLS::set_MKL_threads_to_one () const 
{
    #if USE_MKL == 1
    mkl_set_num_threads( 1 );
    
    LOG( "" );
    LOG( to_string("(THAMLS) set_MKL_threads_to_one :"));
    LOGHLINE;
    LOG( to_string("    eigendecomposition has been computed and number of MKL threads is set back to 1") );
    LOG( to_string("    CFG::nthreads()       = %d",CFG::nthreads()) );
    LOG( to_string("    mkl_get_max_threads() = %d",mkl_get_max_threads()) );
    LOGHLINE;
    #endif
}


 
size_t
THAMLS::eigen_decomp_reduced_problem( const TMatrix *   K_red,
                                      const TMatrix *   M_red,
                                      TDenseMatrix  *   D,
                                      TDenseMatrix  *   Z ) 
{
    if ( M_red == nullptr || K_red == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_reduced_problem", "argument is nullptr" );
    
    if ( get_mode_selection() != MODE_SEL_AUTO_H )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_reduced_problem", "not yet supported" ); 
        
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Decide if multiple threads should be used for the MKL routines. If the sequential MKL library 
    // (and not the parallel one) is linked with the program then this statement has basically no effect 
    // on the subsequent computation.
    //
    // This feature allows to use multiple threads for the solution of "large" dense eigenvalue problems
    // which are solved by the corresponding LAPACK solver.
    // 
    // NOTE: K_red and M_red are the reduced matrices of the HAMLS method which are symmetric and where 
    //       only the lower block triangular part of M_red is computed. Correspondingly, M_red is not 
    //       "physically" symmetric and when K_red*Z-M_red*Z*D is computed the result should in genernal 
    //       be not zero since the upper triangular part of the dense matrix M_red is not physically present. 
    //       Correspondingly, do not use the debug option for the LAPACK solver since otherwise an exception 
    //       could be thrown.
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    const bool use_multiple_MKL_threads = adjust_MKL_threads_for_eigen_decomp( K_red->rows() );
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Computed eigendecomposition of the reduced eigenvalue problem 
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    TIC;
    
    size_t n_ev; 
     
    if ( _para_base.n_ev_searched > 0 )
    {
        size_t uindex = _para_base.n_ev_searched;
        
        if ( uindex > K_red->rows() )
            uindex = K_red->rows();
    
        TEigenLapack eigensolver( idx_t(1), idx_t( uindex ) ); 

        // Do no testing since M is not physically symmetric
        eigensolver.set_test_pos_def  ( false );    
        eigensolver.set_test_residuals( false );
        
        n_ev = eigensolver.comp_decomp( K_red, M_red, D, Z );        
    }// if
    else if ( MODE_SEL_BOUND_H == get_mode_selection() && _para_base.n_ev_searched == 0 )
    {
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_reduced_problem", "not supported anymore" );
        
        //-------------------------------------------------------------------------
        // The lower bound is set to 'lower bound = -get_trunc_bound()' because
        // if it is set to 'lower bound = 0' only eigenvalues bigger than zero 
        // are computed. But this would cause some inconsistencies in the later 
        // analysis of the relative errors of eigenvalues (cf. Notizheft 8, S. 33)
        //-------------------------------------------------------------------------
        TEigenLapack eigensolver( -real(1) * get_trunc_bound(), get_trunc_bound() );
        
        // Do no testing since M is not physically symmetric
        eigensolver.set_test_pos_def  ( false );    
        eigensolver.set_test_residuals( false );
                
        n_ev = eigensolver.comp_decomp( K_red, M_red, D, Z );    
    }// else if
    else
    {
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_reduced_problem", "wrong argument/not implemented" );
    }// else
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Adjust number of MKL threads back to 1 if it is necessary
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( use_multiple_MKL_threads )
        set_MKL_threads_to_one();

    TOC;
    LOG( to_string("(THAMLS) eigen_decomp_reduced_problem : done in %.2fs", toc.seconds() ));
    
    return n_ev;
}






//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Routines for improving the eigenvector which are computed by H-AMLS
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


 
size_t
THAMLS::improve_eigensolutions___eigen_decomp_reduced_problem ( const TMatrix *   K_tilde_c,
                                                                const TMatrix *   M_tilde_c,
                                                                TDenseMatrix  *   D_tilde_c,
                                                                TDenseMatrix  *   S_tilde_c ) const 
{
    if ( K_tilde_c == nullptr || M_tilde_c == nullptr || D_tilde_c == nullptr || S_tilde_c == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions___eigen_decomp_reduced_problem", "argument is nullptr" );
    
        
    
    //====================================================================================================
    // Decide if multiple threads should be used for the MKL routines. If the sequential MKL library 
    // (and not the parallel one) is linked with the program then this statement has basically no effect 
    // on the subsequent computation.
    //
    // This feature allows to use multiple threads for the solution of "large" dense eigenvalue problems
    // which are solved by the corresponding LAPACK solver.
    //====================================================================================================
    const bool use_multiple_MKL_threads = adjust_MKL_threads_for_eigen_decomp( K_tilde_c->rows() );
    
        
    //====================================================================================================
    // Compute all eigenpairs of the reduced eigenvalue problem 
    //====================================================================================================
    TEigenLapack eigensolver;

    eigensolver.set_ev_selection( EV_SEL_FULL );
    
    // Apply tests if wanted
    eigensolver.set_test_pos_def  ( _para_base.do_debug );    
    eigensolver.set_test_residuals( _para_base.do_debug );
    
//     eigensolver.set_test_pos_def  ( true );    
//     eigensolver.set_test_residuals( true );
        
    const size_t n_ev_computed = eigensolver.comp_decomp( K_tilde_c, M_tilde_c, D_tilde_c, S_tilde_c );        
    
    //====================================================================================================
    // Adjust number of MKL threads back to 1 if it is necessary
    //====================================================================================================
    if ( use_multiple_MKL_threads )
        set_MKL_threads_to_one();

    return n_ev_computed;
}




void
THAMLS::improve_eigensolutions___comp_reduced_matrices ( const TMatrix *       K_tilde,
                                                         const TMatrix *       M_tilde,
                                                         const TDenseMatrix *  Q_tilde_new,
                                                         TDenseMatrix *        K_tilde_c,
                                                         TDenseMatrix *        M_tilde_c ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr || Q_tilde_new == nullptr || K_tilde_c == nullptr || M_tilde_c == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions___comp_reduced_matrices", "argument is nullptr" );
    
    //=======================================================================================
    // The computation of  
    //
    //      K_tilde_c := (Q_tilde_new)^{T} * K_tilde * Q_tilde_new
    //      M_tilde_c := (Q_tilde_new)^{T} * M_tilde * Q_tilde_new
    //
    // can be performed by the same routine which is used for the computation of M_red_ij
    //=======================================================================================
    const size_t  n_ev = Q_tilde_new->cols();
    const size_t  ofs  = M_tilde->row_ofs();
    
    K_tilde_c->set_size( n_ev, n_ev );
    M_tilde_c->set_size( n_ev, n_ev );
    
    K_tilde_c->set_ofs ( ofs , ofs  );
    M_tilde_c->set_ofs ( ofs , ofs  );
    
    K_tilde_c->scale( real(0) );
    M_tilde_c->scale( real(0) );
    
    
    //TODO NOTE Theoretisch Kann ich K_tilde_c mit alten Zwischenergebnis guenstiger berechnen!
    
    auto  comp_reduced_matrix =
        [ this, Q_tilde_new, K_tilde, M_tilde, K_tilde_c, M_tilde_c ] ( const uint  i )
        {
            if ( i == 0 )
                comp_M_red_ij_aux ( Q_tilde_new, K_tilde , Q_tilde_new, K_tilde_c );
            
            if ( i == 1 )
                comp_M_red_ij_aux ( Q_tilde_new, M_tilde , Q_tilde_new, M_tilde_c );
        };
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        tbb::parallel_for( uint(0), uint(2), comp_reduced_matrix );
    }// if
    else
    {
        for ( uint  i = 0; i < 2; ++i )
            comp_reduced_matrix( i );
    }// else
    
    
    //=======================================================================================
    // Set the computed reduced matrices symmetric (this is for example 
    // a requirement to use the function 'TEigenLapack::comp_decomp')
    //=======================================================================================
    K_tilde_c->set_symmetric();
    M_tilde_c->set_symmetric();
    
    //=======================================================================================
    // NOTE: Set IDs of reduced matrices. If this is not done it can happen 
    // that an error message is created when matrix algebra routines are 
    // applied to the reduced matrices, e.g., LDL-decomposition of M_red
    //=======================================================================================
    K_tilde_c->set_id( K_tilde->id() );
    M_tilde_c->set_id( M_tilde->id() );
}


void
THAMLS::improve_eigensolutions___iterate_eigenvectors ( const TMatrix *       K_tilde,
                                                        const TMatrix *       M_tilde,
                                                        const TDenseMatrix *  S_tilde,
                                                        TDenseMatrix *        Q_tilde_new )
{
    if ( K_tilde == nullptr || M_tilde == nullptr || S_tilde == nullptr || Q_tilde_new == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions___iterate_eigenvectors", "argument is nullptr" );
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   Q_tilde_new: = (K_tilde)^{-1} * M_tilde * S_tilde
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
    const size_t  N    = S_tilde->rows();
    const size_t  n_ev = S_tilde->cols();
    const size_t  ofs  = M_tilde->row_ofs();
    
    Q_tilde_new->set_size( N,    n_ev );
    Q_tilde_new->set_ofs ( ofs , ofs  );
    
    
    
    //===========================================================================================
    // The computation of  
    //
    //         Temp: = M_tilde * S_tilde
    //
    // can be performed by the same subroutine which is used for the computation of M_red_ij
    //===========================================================================================
    unique_ptr< TDenseMatrix > Temp( new TDenseMatrix( M_tilde->row_is(), S_tilde->col_is() ) );
    
    if ( _para_parallel.M_red_ij_step_1_in_parallel )
        comp_M_red_ij_aux_step_1 ( M_tilde, S_tilde, Temp.get() );
    else
        multiply( real(1), MATOP_NORM, M_tilde, MATOP_NORM, S_tilde, real(0), Temp.get(), acc_exact );
    
    
    //===========================================================================================
    // Compute the matrix 
    //
    //         Q_tilde_new: = K_tilde^{-1} * Temp
    //
    //===========================================================================================
    
    ///-------------------------------------------------------------------------------------------------------------
    ///TODO NOTE Ist das auch gut genug parallel implementiert? Sonst mache ich das selber?
    /// Invert diag sollte schnell selber programmiert sein
    ///
    ///TODO NOTE Falls ich das auch rekursive anwenden will, dann muss ich global K_tilde_inv aufstellen
    ///          und dann greife ich auf die entsprechenden Submatrizen zurueck. Dann wird das bei Rekursion nicht
    ///          mehrfach breechnet
    ///
    /// TODO NOTE Kann ich K_tilde_inv direkt bei der Berechnung von der LDL Zerlegung K=L*K_tilde*L bekommen?
    ///-------------------------------------------------------------------------------------------------------------
    unique_ptr< TMatrix >  K_tilde_inv( K_tilde->copy() ); 
    
    const inv_options_t inv_opt;
    
    invert_diag ( K_tilde_inv.get(), acc_exact, inv_opt );
    
    multiply( real(1), MATOP_NORM, K_tilde_inv.get(), MATOP_NORM, Temp.get(), real(0), Q_tilde_new, acc_exact );
    
}


void
THAMLS::improve_eigensolutions ( const TMatrix *   K_tilde,
                                 const TMatrix *   M_tilde,
                                 TDenseMatrix *    D_tilde,
                                 TDenseMatrix *    S_tilde ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr || D_tilde == nullptr || S_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions", "argument is nullptr" );
 
    //===========================================================================================
    // Step 1: Iterate one time, i.e., compute 
    //
    //         Q_tilde_new: = (K_tilde)^{-1} * M_tilde * Q_tilde  with Q_tilde:=S_tilde
    //
    //===========================================================================================
    unique_ptr< TDenseMatrix >  Q_tilde_new( new TDenseMatrix() );
    
    improve_eigensolutions___iterate_eigenvectors( K_tilde, M_tilde, S_tilde, Q_tilde_new.get() );
    
    //===========================================================================================
    // Step 2: Compute reduced matrices
    //
    //         K_tilde_c := (Q_tilde_new)^{T} * K_tilde * Q_tilde_new
    //         M_tilde_c := (Q_tilde_new)^{T} * M_tilde * Q_tilde_new
    //
    //===========================================================================================
    unique_ptr< TDenseMatrix >  K_tilde_c( new TDenseMatrix() );
    unique_ptr< TDenseMatrix >  M_tilde_c( new TDenseMatrix() );  
        
    improve_eigensolutions___comp_reduced_matrices( K_tilde, M_tilde, Q_tilde_new.get(), K_tilde_c.get(), M_tilde_c.get() );
    
    
    //===========================================================================================
    // Step 3: Solve the reduced eigenvalue problem  
    //
    //         K_tilde * S_tilde_c = M_tilde_c * S_tilde_c * D_tilde_c
    //
    //===========================================================================================    
    TDenseMatrix * D_tilde_c = D_tilde;
    
    unique_ptr< TDenseMatrix >  S_tilde_c( new TDenseMatrix );
    
    improve_eigensolutions___eigen_decomp_reduced_problem( K_tilde_c.get(), M_tilde_c.get(), D_tilde_c, S_tilde_c.get() );
    
    
    //===========================================================================================
    // Step 4: Compute new improved eigenvectors 
    //
    //         S_tilde_new := Q_tilde_new * S_tilde_c
    //
    //===========================================================================================
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        // this can be computed in parallel using the following routine
        comp_S_i_times_S_i_red ( Q_tilde_new.get(), S_tilde_c.get(), S_tilde );
    }// if
    else
    {
        multiply( real(1), MATOP_NORM, Q_tilde_new.get(), MATOP_NORM, S_tilde_c.get(), real(0), S_tilde, acc_exact );
    }// else
    
}



void
THAMLS::improve_eigensolutions_sparse___iterate_eigenvectors ( const TMatrix *        K_tilde,
                                                               const TMatrix *        L,
                                                               const TSparseMatrix *  M,
                                                               const TDenseMatrix *   S,
                                                               TDenseMatrix *         MQ,
                                                               TDenseMatrix *         Q_new )
{
    if ( K_tilde == nullptr || L == nullptr || M == nullptr || S == nullptr || Q_new == nullptr)
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse___iterate_eigenvectors", "argument is nullptr" );
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   Q_new: = K^{-1} * M * S 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
    const size_t  N    = S->rows();
    const size_t  n_ev = S->cols();
    const size_t  ofs  = K_tilde->row_ofs();
    
    Q_new->set_size( N,    n_ev );
    Q_new->set_ofs ( ofs , ofs  );
    
    
    //===========================================================================================
    // Compute  Temp: = M * S  where Temp=MQ
    //===========================================================================================
    sparse_matrix_multi ( M, S, MQ, _para_parallel.miscellaneous_parallel  );
    
    unique_ptr< TMatrix >  Temp( MQ->copy() );
    auto  Temp_p = ptrcast( Temp.get(), TDenseMatrix );
    
    //===========================================================================================
    // Compute the matrix 
    //
    //         Q_new: = K^{-1} * Temp = L^{-T}*(K_tilde)^{-1}*L^{-1}  * Temp
    //
    //===========================================================================================
    backtransform_eigenvectors_with_L ( L, MATOP_NORM, Temp_p );
    
    
    ///-------------------------------------------------------------------------------------------------------------
    ///TODO NOTE Ist das auch gut genug parallel implementiert? Sonst mache ich das selber?
    /// Invert diag sollte schnell selber programmiert sein
    ///
    ///TODO NOTE Falls ich das auch rekursive anwenden will, dann muss ich global K_tilde_inv aufstellen
    ///          und dann greife ich auf die entsprechenden Submatrizen zurueck. Dann wird das bei Rekursion nicht
    ///          mehrfach breechnet
    ///
    /// TODO NOTE Kann ich K_tilde_inv direkt bei der Berechnung von der LDL Zerlegung K=L*K_tilde*L bekommen?
    ///-------------------------------------------------------------------------------------------------------------
    unique_ptr< TMatrix >  K_tilde_inv( K_tilde->copy() ); 
    const inv_options_t inv_opt;
    invert_diag ( K_tilde_inv.get(), acc_exact, inv_opt );
    multiply( real(1), MATOP_NORM, K_tilde_inv.get(), MATOP_NORM, Temp_p, real(0), Q_new, acc_exact );
    
    backtransform_eigenvectors_with_L ( L, MATOP_TRANS, Q_new );
}


void
THAMLS::improve_eigensolutions_sparse___iterate_eigenvectors ( const TMatrix *        K_tilde,
                                                               const TMatrix *        L,
                                                               const TSparseMatrix *  M,
                                                               const TDenseMatrix *   S,
                                                               TDenseMatrix *         Q_new )
{
    if ( K_tilde == nullptr || L == nullptr || M == nullptr || S == nullptr || Q_new == nullptr)
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse___iterate_eigenvectors", "argument is nullptr" );
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   Q_new: = K^{-1} * M * S 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
    const size_t  N    = S->rows();
    const size_t  n_ev = S->cols();
    const size_t  ofs  = K_tilde->row_ofs();
    
    Q_new->set_size( N,    n_ev );
    Q_new->set_ofs ( ofs , ofs  );
    
    
    //===========================================================================================
    // Compute  Temp: = M * S  
    //===========================================================================================
    unique_ptr< TDenseMatrix >  Temp( new TDenseMatrix() );
    
    sparse_matrix_multi ( M, S, Temp.get(), _para_parallel.miscellaneous_parallel  );
    
    
    //===========================================================================================
    // Compute the matrix 
    //
    //         Q_new: = K^{-1} * Temp = L^{-T}*(K_tilde)^{-1}*L^{-1}  * Temp
    //
    //===========================================================================================
    backtransform_eigenvectors_with_L ( L, MATOP_NORM, Temp.get() );
    
    ///-------------------------------------------------------------------------------------------------------------
    ///TODO NOTE Ist das auch gut genug parallel implementiert? Sonst mache ich das selber?
    /// Invert diag sollte schnell selber programmiert sein
    ///
    ///TODO NOTE Falls ich das auch rekursive anwenden will, dann muss ich global K_tilde_inv aufstellen
    ///          und dann greife ich auf die entsprechenden Submatrizen zurueck. Dann wird das bei Rekursion nicht
    ///          mehrfach breechnet
    ///
    /// TODO NOTE Kann ich K_tilde_inv direkt bei der Berechnung von der LDL Zerlegung K=L*K_tilde*L bekommen?
    ///-------------------------------------------------------------------------------------------------------------
    unique_ptr< TMatrix >  K_tilde_inv( K_tilde->copy() ); 
    const inv_options_t inv_opt;
    invert_diag ( K_tilde_inv.get(), acc_exact, inv_opt );
    multiply( real(1), MATOP_NORM, K_tilde_inv.get(), MATOP_NORM, Temp.get(), real(0), Q_new, acc_exact );
    
    backtransform_eigenvectors_with_L ( L, MATOP_TRANS, Q_new );
}


void 
THAMLS::improve_eigensolutions_sparse___iterate_eigenvectors_with_precond ( const TSparseMatrix *  K,
                                                                            const TSparseMatrix *  M,
                                                                            const TDenseMatrix *   S,
                                                                            TDenseMatrix *         MQ,
                                                                            TDenseMatrix *         Q_new )
{
    if ( M == nullptr || S == nullptr || Q_new == nullptr)
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse___iterate_eigenvectors_with_precond", "argument is nullptr" );
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   Q_new: = K^{-1} * M * Q_old   with Q_old = S
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
    const size_t  N    = S->rows();
    const size_t  n_ev = S->cols();
    const size_t  ofs  = S->row_ofs();
    
    Q_new->set_size( N,    n_ev );
    Q_new->set_ofs ( ofs , ofs  );
    
    const TDenseMatrix * Q_old = S;
    
    for ( size_t i = 0; i < _para_impro.number_of_iterations; i++ )
    {
        //===========================================================================================
        // Compute  Temp: = M * Q_old  where Temp=MQ
        //===========================================================================================
        sparse_matrix_multi ( M, Q_old, MQ, _para_parallel.miscellaneous_parallel  );
        
        
        //===========================================================================================
        // Compute  Q_new: = K^{-1} * Temp 
        //===========================================================================================
        
        auto K_inv_multi = 
            [ this, MQ, Q_new, K ] ( const tbb::blocked_range< uint > & r )
            {
                for ( auto  j = r.begin(); j != r.end(); ++j )
                {                
                    //-------------------------------------------------------
                    // uses references for the corresponding columns
                    //-------------------------------------------------------
                    const TScalarVector x( MQ->column( j ) );
                    TScalarVector       y( Q_new->column( j ) );
                    
                    //-------------------------------------------------------
                    // compute y = K^{-1} * x  via solving  Ky=x for y
                    //-------------------------------------------------------
                    TSolverInfo     info;
                    
                    // solver with standard setting
                    TStopCriterion  sstop( 100, real(1e-8), real(1e-8), real(1e6) );
                    TAutoSolver     solver( sstop );
                    
                    solver.solve( K, & y , & x, _para_impro.K_preconditioner, &info );
                    
                    if ( !info.has_converged() )
                        LOG( "(THAMLS) improve_eigensolutions_sparse___iterate_eigenvectors_with_precond: WARNING: iterativ solver did not converge" );
                }// for
                
            };
            
        if ( _para_parallel.miscellaneous_parallel ) 
            tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_ev) ), K_inv_multi );
        else 
            K_inv_multi ( tbb::blocked_range< uint >( uint(0), uint(n_ev) ) );
        

        //===========================================================================================
        // Prepare next iteration
        //===========================================================================================        
        Q_old = Q_new;
        //TODO: For the stability of the iteration the iteration matrix Q has to be orthogonalised 
        
    }// for
}



void
THAMLS::improve_eigensolutions_sparse___comp_reduced_matrices ( const TSparseMatrix *  K,
                                                                const TSparseMatrix *  M,
                                                                const TDenseMatrix *   Q_new,
                                                                const TDenseMatrix *   MQ,
                                                                TDenseMatrix *         K_c,
                                                                TDenseMatrix *         M_c ) 
{
    if ( K == nullptr || M == nullptr || Q_new == nullptr || K_c == nullptr || M_c == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse___comp_reduced_matrices", "argument is nullptr" );
    
    //=======================================================================================
    // The computation of  
    //
    //         K_c := (Q_new)^{T} * K * Q_new
    //         M_c := (Q_new)^{T} * M * Q_new
    //
    // can be performed by the same routine which is used for the computation of M_red_ij
    //=======================================================================================
    const size_t  n_ev = Q_new->cols();
    const size_t  ofs  = M->row_ofs();
    
    K_c->set_size( n_ev, n_ev );
    M_c->set_size( n_ev, n_ev );
    
    K_c->set_ofs ( ofs , ofs  );
    M_c->set_ofs ( ofs , ofs  );
    
    K_c->scale( real(0) );
    M_c->scale( real(0) );
    
    auto  comp_reduced_matrix =
        [ this, Q_new, MQ, M, K_c, M_c ] ( const uint  i )
        {
            if ( i == 0 )
            {
                //========================================================================
                // Computation of    K_c := (Q_new)^{T} * K * Q_new
                //
                // Use the identity K*Q_new = M*Q to avoid explicit computaion of K*Q_new
                //========================================================================
                
                //unique_ptr< TDenseMatrix >  Temp( new TDenseMatrix() );
                //sparse_matrix_multi ( K, Q_new, Temp.get(), _para_parallel.miscellaneous_parallel );
                  
                if ( _para_parallel.miscellaneous_parallel )
                    comp_M_red_ij_aux_step_2 ( Q_new, MQ, K_c );
                else
                    multiply( real(1), MATOP_TRANS, Q_new, MATOP_NORM, MQ, real(0), K_c, acc_exact );
            }// if
            
            if ( i == 1 )
            {
                //========================================================================
                // Computation of    M_c := (Q_new)^{T} * M * Q_new
                //========================================================================
                unique_ptr< TDenseMatrix >  Temp( new TDenseMatrix() );
                
                sparse_matrix_multi ( M, Q_new, Temp.get(), _para_parallel.miscellaneous_parallel );
                  
                if ( _para_parallel.miscellaneous_parallel )
                    comp_M_red_ij_aux_step_2 ( Q_new, Temp.get(), M_c );
                else
                    multiply( real(1), MATOP_TRANS, Q_new, MATOP_NORM, Temp.get(), real(0), M_c, acc_exact );
            }// if
                     
        };
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        tbb::parallel_for( uint(0), uint(2), comp_reduced_matrix );
    }// if
    else
    {
        for ( uint  i = 0; i < 2; ++i )
            comp_reduced_matrix( i );
    }// else
    
    
    //=======================================================================================
    // Set the computed reduced matrices symmetric (this is for example 
    // a requirement to use the function 'TEigenLapack::comp_decomp')
    //=======================================================================================
    K_c->set_symmetric();
    M_c->set_symmetric();
    
    //=======================================================================================
    // NOTE: Set IDs of reduced matrices. If this is not done it can happen 
    // that an error message is created when matrix algebra routines are 
    // applied to the reduced matrices, e.g., LDL-decomposition of M_red
    //=======================================================================================
    K_c->set_id( K->id() );
    M_c->set_id( M->id() );
}




void
THAMLS::improve_eigensolutions_sparse___comp_reduced_matrices ( const TSparseMatrix *  K,
                                                                const TSparseMatrix *  M,
                                                                const TDenseMatrix *   Q_new,
                                                                TDenseMatrix *         K_c,
                                                                TDenseMatrix *         M_c ) 
{
    if ( K == nullptr || M == nullptr || Q_new == nullptr || K_c == nullptr || M_c == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse___comp_reduced_matrices", "argument is nullptr" );
    
    //=======================================================================================
    // The computation of  
    //
    //         K_c := (Q_new)^{T} * K * Q_new
    //         M_c := (Q_new)^{T} * M * Q_new
    //
    // can be performed by the same routine which is used for the computation of M_red_ij
    //=======================================================================================
    const size_t  n_ev = Q_new->cols();
    const size_t  ofs  = M->row_ofs();
    
    K_c->set_size( n_ev, n_ev );
    M_c->set_size( n_ev, n_ev );
    
    K_c->set_ofs ( ofs , ofs  );
    M_c->set_ofs ( ofs , ofs  );
    
    K_c->scale( real(0) );
    M_c->scale( real(0) );
    
    auto  comp_reduced_matrix =
        [ this, Q_new, K, M, K_c, M_c ] ( const uint  i )
        {
            if ( i == 0 )
            {
                //========================================================================
                // Computation of    K_c := (Q_new)^{T} * K * Q_new
                //========================================================================
                unique_ptr< TDenseMatrix >  Temp( new TDenseMatrix() );
                
                sparse_matrix_multi ( K, Q_new, Temp.get(), _para_parallel.miscellaneous_parallel );
                  
                if ( _para_parallel.miscellaneous_parallel )
                    comp_M_red_ij_aux_step_2 ( Q_new, Temp.get(), K_c );
                else
                    multiply( real(1), MATOP_TRANS, Q_new, MATOP_NORM, Temp.get(), real(0), K_c, acc_exact );
            }// if
            
            if ( i == 1 )
            {
                //========================================================================
                // Computation of    M_c := (Q_new)^{T} * M * Q_new
                //========================================================================
                unique_ptr< TDenseMatrix >  Temp( new TDenseMatrix() );
                
                sparse_matrix_multi ( M, Q_new, Temp.get(), _para_parallel.miscellaneous_parallel );
                  
                if ( _para_parallel.miscellaneous_parallel )
                    comp_M_red_ij_aux_step_2 ( Q_new, Temp.get(), M_c );
                else
                    multiply( real(1), MATOP_TRANS, Q_new, MATOP_NORM, Temp.get(), real(0), M_c, acc_exact );
            }// if
                
        };
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        tbb::parallel_for( uint(0), uint(2), comp_reduced_matrix );
    }// if
    else
    {
        for ( uint  i = 0; i < 2; ++i )
            comp_reduced_matrix( i );
    }// else
    
    
    //=======================================================================================
    // Set the computed reduced matrices symmetric (this is for example 
    // a requirement to use the function 'TEigenLapack::comp_decomp')
    //=======================================================================================
    K_c->set_symmetric();
    M_c->set_symmetric();
    
    //=======================================================================================
    // NOTE: Set IDs of reduced matrices. If this is not done it can happen 
    // that an error message is created when matrix algebra routines are 
    // applied to the reduced matrices, e.g., LDL-decomposition of M_red
    //=======================================================================================
    K_c->set_id( K->id() );
    M_c->set_id( M->id() );
}




void
THAMLS::improve_eigensolutions_sparse ( const TSparseMatrix *  K,
                                        const TSparseMatrix *  M,
                                        const TMatrix *        K_tilde,
                                        const TMatrix *        L,
                                        TDenseMatrix *         D,
                                        TDenseMatrix *         S ) 
{
    if ( K == nullptr || M == nullptr || K_tilde == nullptr || L == nullptr || D == nullptr || S == nullptr )
        HERROR( ERR_ARG, "(THAMLS) improve_eigensolutions_sparse", "argument is nullptr" );
    
 
    //=========================================================================================================
    // Step 1: Iterate one time, i.e., compute 
    //
    //         Q_new: = K^{-1} * M * Q  with Q:=S
    //
    //=========================================================================================================
    unique_ptr< TDenseMatrix >  Q_new( new TDenseMatrix() );
    unique_ptr< TDenseMatrix >  MQ   ( new TDenseMatrix() );
    
    if ( _para_impro.use_K_inv_as_precomditioner )
    {
        improve_eigensolutions_sparse___iterate_eigenvectors_with_precond( K, M, S, MQ.get(), Q_new.get() );
    }// if
    else
    {
        //old: improve_eigensolutions_sparse___iterate_eigenvectors( K_tilde, L, M, S, MQ.get(), Q_new.get() );
        improve_eigensolutions_sparse___iterate_eigenvectors( K_tilde, L, M, S, Q_new.get() );
    }// if
    
    
    //===========================================================================================
    // Step 2: Compute reduced matrices
    //
    //         K_c := (Q_new)^{T} * K * Q_new
    //         M_c := (Q_new)^{T} * M * Q_new
    //
    //===========================================================================================
    unique_ptr< TDenseMatrix >  K_c( new TDenseMatrix() );
    unique_ptr< TDenseMatrix >  M_c( new TDenseMatrix() );  
        
    //old: improve_eigensolutions_sparse___comp_reduced_matrices( K, M, Q_new.get(), MQ.get(), K_c.get(), M_c.get() );
    improve_eigensolutions_sparse___comp_reduced_matrices( K, M, Q_new.get(), K_c.get(), M_c.get() );
    
    
    //===========================================================================================
    // Step 3: Solve the reduced eigenvalue problem  
    //
    //         K_c * S_c = M_c * S_c * D_c
    //
    //===========================================================================================    
    TDenseMatrix * D_c = D;
    
    unique_ptr< TDenseMatrix >  S_c( new TDenseMatrix );
    
    improve_eigensolutions___eigen_decomp_reduced_problem( K_c.get(), M_c.get(), D_c, S_c.get() );
    
    //===========================================================================================
    // Step 4: Compute new improved eigenvectors 
    //
    //         S_new := Q_new * S_c 
    //
    //===========================================================================================
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        // this can be computed in parallel using the following routine
        comp_S_i_times_S_i_red ( Q_new.get(), S_c.get(), S );
    }// if
    else
    {
        multiply( real(1), MATOP_NORM, Q_new.get(), MATOP_NORM, S_c.get(), real(0), S, acc_exact );
    }// else
    
}

                                                      


//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Back transformation of the eigenvectors
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




void 
THAMLS::transform_eigensolutions ( const TSparseMatrix * K_sparse, 
                                   const TSparseMatrix * M_sparse, 
                                   const TClusterTree *  ct,
                                   const TMatrix *       K_tilde,
                                   const TMatrix *       M_tilde,
                                   const TMatrix *       L,
                                   const TDenseMatrix *  S_red,
                                   TDenseMatrix *        D_approx,
                                   TDenseMatrix *        S_approx )
{    
    if ( K_sparse == nullptr || M_sparse == nullptr || ct == nullptr || K_tilde == nullptr || M_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) transform_eigensolutions", "argument is nullptr" );
    
    if ( L == nullptr || S_red == nullptr || D_approx == nullptr || S_approx == nullptr )
        HERROR( ERR_ARG, "(THAMLS) transform_eigensolutions", "argument is nullptr" );
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Backtransform eigenvectors of the reduced problem contained in S_red by computing 
    //
    // S_approx = L^{-T} * diag[S_1,...,S_m] * S_red
    // 
    // and compute afterthat the corresponding Rayleigh Quotients if wanted
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    TIC;
    
    //--------------------------------------------------------------------------
    // Handle trivial case
    //--------------------------------------------------------------------------
    if ( S_red->cols() == 0 )
    {
        const size_t n   = L->rows();
        const size_t nev = S_red->cols();
        const idx_t  ofs = 0;
        
        S_approx->set_size( n, nev );
        S_approx->set_ofs ( ofs, ofs );
        
        D_approx->set_size( nev, nev );
        D_approx->set_ofs ( ofs, ofs );
        
        return;
    }// if
    
    //--------------------------------------------------------------------------
    // Compute S_tilde = diag[S_1,...,S_m] * S_red
    //--------------------------------------------------------------------------
    backtransform_eigenvectors_with_S_i ( S_red, S_approx );
    
    //--------------------------------------------------------------------------
    // Improve eigensolutions by subspace iteration
    //--------------------------------------------------------------------------    
    if ( _para_impro.do_improving && !_para_impro.use_sparse_version )
    {
        //NOTE: This version is less time efficient than the sparse version!
        improve_eigensolutions ( K_tilde, M_tilde, D_approx, S_approx );
    }// if
    
    //--------------------------------------------------------------------------
    // Compute S_approx = L^{-T} * S_tilde
    //--------------------------------------------------------------------------
    backtransform_eigenvectors_with_L ( L, MATOP_TRANS, S_approx );
    
    TOC;
    LOG( to_string("(THAMLS) transform_eigensolutions : done in %.2fs", toc.seconds() ));
    
    //--------------------------------------------------------------------------
    // Subsequent subspace iteration / computation of Rayleigh Quotients
    //--------------------------------------------------------------------------   
    if ( K_sparse != nullptr && M_sparse != nullptr )
    {
        //--------------------------------------------------------------------------
        // Permute entries in sparse matrices according to internal numbering
        //--------------------------------------------------------------------------    
        auto  K_sparse_copy = K_sparse->copy();
        auto  M_sparse_copy = M_sparse->copy();
        
        auto  K_sparse_internal = ptrcast( K_sparse_copy.get(), TSparseMatrix );
        auto  M_sparse_internal = ptrcast( M_sparse_copy.get(), TSparseMatrix );
        
        TPermutation perm_internal( *(ct->perm_i2e()) );
        
        perm_internal.invert();
        
        K_sparse_internal->permute( perm_internal, perm_internal );
        M_sparse_internal->permute( perm_internal, perm_internal );
                
        if ( _para_impro.do_improving && _para_impro.use_sparse_version )
        {
            //--------------------------------------------------------------------------
            // Improve eigensolutions (in this case Rayleigh Quotients are not needed)
            //--------------------------------------------------------------------------    
            improve_eigensolutions_sparse ( K_sparse_internal, M_sparse_internal, K_tilde, L, D_approx, S_approx );
            
        }// if
        else if ( _para_base.comp_rayleigh_quotients )
        {
            //--------------------------------------------------------------------------
            // Compute Rayleigh quotients if wanted
            //--------------------------------------------------------------------------     
            EIGEN_MISC::comp_rayleigh_quotients( K_sparse_internal, M_sparse_internal, D_approx, S_approx,
                                                _para_base.stable_rayleigh, _para_parallel.miscellaneous_parallel );
             
        }// else if
    }// if
}


void 
THAMLS::backtransform_eigenvectors_with_S_i ( const TDenseMatrix * S_red,
                                              TDenseMatrix *       S_tilde ) 
{
    unique_ptr< subprob_data_t > subprob_data( new subprob_data_t );
    
    subprob_data->first = 0;
    subprob_data->last  = _sep_tree.n_subproblems() - 1;
    
    backtransform_eigenvectors_with_S_i ( S_red, S_tilde, subprob_data.get() );
}



void 
THAMLS::backtransform_eigenvectors_with_S_i ( const TDenseMatrix * S_red,
                                              TDenseMatrix *       S_tilde,
                                              subprob_data_t *     subprob_data ) 
{
    if ( S_red == nullptr || S_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) backtransform_eigenvectors_with_S_i", "argument is nullptr" );
    
    //========================================================================================
    // Initialise S_tilde and adjust offset for subsequent computation
    //========================================================================================
    const size_t n    = get_dofs_SUB ( subprob_data );
    const size_t nev  = S_red->cols();
    const idx_t  ofs  = _subproblem_ofs[subprob_data->first];
    const idx_t  reduced_ofs  = _reduced_subproblem_ofs[subprob_data->first];
    
    if ( get_reduced_dofs_SUB( subprob_data ) != S_red->rows() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) backtransform_eigenvectors_with_S_i", "" ); 
    
    S_tilde->set_size( n,   nev );
    S_tilde->set_ofs ( ofs, reduced_ofs );
    
    if ( nev == 0 )
        return;
                
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Transform the eigenvector 'x' of the reduced problem to an eigenvector approximation
    // 'v' of the eigenvalue problem (K_tilde_sub,M_tilde_sub) via computing v = diag[S_1, ..., S_m] * x
    //
    // The computation of S_tilde: = diag[S_1, ..., S_m] * S_red is performed block-wise.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
            
    auto transform_eigenvector_block = 
        [ this, S_red, S_tilde ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //=================================================================================
                // Adjust the offset of the matrix S_i to ensure that the subsequent 
                // matrix matrix multiplication is consistent
                //=================================================================================
                const idx_t ofs_j         = _subproblem_ofs[j];
                const idx_t reduced_ofs_j = _reduced_subproblem_ofs[j];
                
                _S[j]->set_ofs( ofs_j, reduced_ofs_j );
                
                if ( _S[j]->cols() > 0 )
                {
                    //=================================================================================
                    // Initialise subblock of the eigenvector matrix S_red of the reduced EVP 
                    // (S_red_j is a reference to the corresponding block of S_red)
                    //=================================================================================
                    BLAS::Matrix< real >  S_red_BLAS_restricted( S_red->blas_rmat(),
                                                                _S[j]->col_is() - S_red->row_ofs(),
                                                                S_red->col_is() - S_red->col_ofs() );
            
                    unique_ptr< TDenseMatrix > S_red_j( new TDenseMatrix( _S[j]->col_is(), S_red->col_is(), S_red_BLAS_restricted ) );
                    
                    
                    //=================================================================================
                    // Initialise subblock of the transformed eigenvector matrix S_tilde_j
                    // (S_tilde_j is a reference to the corresponding block of S_tilde)
                    //=================================================================================
                    BLAS::Matrix< real >  S_tilde_BLAS_restricted( S_tilde->blas_rmat(),
                                                                _S[j]->row_is()   - S_tilde->row_ofs(),
                                                                S_tilde->col_is() - S_tilde->col_ofs() );
            
                    unique_ptr< TDenseMatrix > S_tilde_j( new TDenseMatrix( _S[j]->row_is(), S_tilde->col_is(), S_tilde_BLAS_restricted ) );
                    
                    //=================================================================================
                    // Compute   S_tilde_j = S_j * S_red_j
                    //=================================================================================                
                    if ( _para_parallel.miscellaneous_parallel )
                    {
                        comp_S_i_times_S_i_red ( _S[j], S_red_j.get(), S_tilde_j.get() );
                    }// if
                    else
                    {
                        multiply( real(1), MATOP_NORM, _S[j], MATOP_NORM, S_red_j.get(), real(0), S_tilde_j.get(), acc_exact );
                    }//else 
                }// if
            }// for
        };  
    
    if ( _para_parallel.miscellaneous_parallel )
    {
        tbb::parallel_for( tbb::blocked_range< uint >( uint(subprob_data->first), uint(subprob_data->last+1) ), transform_eigenvector_block );
    }// if
    else 
        transform_eigenvector_block ( tbb::blocked_range< uint >( uint(subprob_data->first), uint(subprob_data->last+1) ) );
    
    //========================================================================================
    // Update offset: S_tilde is approximative eigensolution of (K_tilde_sub,M_tilde_sub)
    //========================================================================================
    S_tilde->set_ofs ( ofs, ofs );
}




void
THAMLS::comp_S_i_times_S_i_red ( const TDenseMatrix * S_i,
                                 const TDenseMatrix * S_red_i,
                                 TDenseMatrix *       S_tilde_i ) const
{
    if ( S_i == nullptr || S_red_i == nullptr || S_tilde_i == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_S_i_times_S_i_red", "argument is nullptr" );
    
    
    ////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the matrix       S_tilde_i := S_i * S_red_i
    //
    // In order to compute the matrix result in parallel we subdivide 
    // S_tilde_i into several block rows which can be computed separately
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////
    
    
    //-----------------------------------------------------
    // Initialise the block structure of the result
    //-----------------------------------------------------
    const size_t max_size = _para_parallel.transform_with_S_i___block_partition_size;
        
    size_t block_rows = S_tilde_i->rows() / max_size;
        
    if ( block_rows < 1 )
        block_rows = 1;

    //----------------------------------------------------------------------
    // Compute S_tilde_i blockrow-wise
    //----------------------------------------------------------------------                    
    
    auto  compute_S_tilde_i_sub =
        [ S_i, S_red_i, S_tilde_i, & block_rows, & max_size ] ( const tbb::blocked_range< uint > &  r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //----------------------------------------------------------
                // Determine the row index set of the corresponding subblock
                //----------------------------------------------------------
                const size_t row_first = S_tilde_i->row_is().first() + j*max_size;
                
                size_t row_last = S_tilde_i->row_is().first() + (j+1)*max_size - 1;
                
                if ( j == block_rows - 1 )
                    row_last = S_tilde_i->row_is().last();
                    
                const TIndexSet sub_row_is( row_first, row_last );
                
                //----------------------------------------------------------
                // Create reference to submatrix of S_i
                //----------------------------------------------------------
                BLAS::Matrix< real >  S_i_BLAS_sub ( S_i->blas_rmat(),
                                                     sub_row_is    - S_i->row_ofs(),
                                                     S_i->col_is() - S_i->col_ofs() );
            
                unique_ptr< TDenseMatrix > S_i_sub( new TDenseMatrix( sub_row_is, S_i->col_is(), S_i_BLAS_sub ) );
                
                //----------------------------------------------------------
                // Create reference to submatrix of S_tilde_i
                //----------------------------------------------------------
                BLAS::Matrix< real >  S_tilde_i_BLAS_sub ( S_tilde_i->blas_rmat(),
                                                           sub_row_is          - S_tilde_i->row_ofs(),
                                                           S_tilde_i->col_is() - S_tilde_i->col_ofs() );
            
                unique_ptr< TDenseMatrix > S_tilde_i_sub( new TDenseMatrix( sub_row_is, S_tilde_i->col_is(), S_tilde_i_BLAS_sub ) );
                                
                //-----------------------------------------------------
                // Compute S_tilde_i_sub = S_i_sub * S_red_i
                //-----------------------------------------------------
                multiply( real(1), MATOP_NORM, S_i_sub.get(), MATOP_NORM, S_red_i, real(0), S_tilde_i_sub.get(), acc_exact );
                
            }// for
        };
    
    
    //--------------------------------------------------
    // Handle small problems separately in order 
    // to avoid unnecessary parallel overhead 
    //--------------------------------------------------
    
    if ( block_rows == 1 )
    {   
        multiply( real(1), MATOP_NORM, S_i, MATOP_NORM, S_red_i, real(0), S_tilde_i, acc_exact );
    }// if
    else
    {
        if ( _para_parallel.level_of_affinity_partitioner_usage > 0 )
        {
            affinity_partitioner  ap;
            
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(block_rows) ), compute_S_tilde_i_sub, ap );
        }// if
        else
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(block_rows) ), compute_S_tilde_i_sub );
    }// else
}


void 
THAMLS::backtransform_eigenvectors_with_L ( const TMatrix * L,
                                            const matop_t   op_L,
                                            TDenseMatrix *  S_tilde ) const
{
    if ( L == nullptr || S_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) backtransform_eigenvectors_with_L", "argument is nullptr" );
    
    /////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute   S := L^{-T} * S_tilde
    //
    // Divide S and S_tilde in block columns for parallel computation
    //
    /////////////////////////////////////////////////////////////////////////////////////////
    const solve_option_t  solve_L_opts( block_wise, unit_diag );
    
      
    //-----------------------------------------------------
    // Initialise the block structure of the result
    //-----------------------------------------------------
    size_t max_size;

    if ( _para_parallel.transform_with_L___partition_by_threads )
        max_size = S_tilde->cols() / CFG::nthreads();
    else 
        max_size = _para_parallel.transform_with_L___block_partition_size;
    
    if ( max_size < 15 )
        max_size = 15;  ///good value
        
    size_t block_cols = S_tilde->cols() / max_size;
        
    if ( block_cols < 1 )
        block_cols = 1;

    //----------------------------------------------------------------------
    // Compute S blockcolumn-wise
    //----------------------------------------------------------------------                    
    
    auto  compute_S_sub =
        [ S_tilde, L, & block_cols, & max_size, & solve_L_opts, & op_L ] ( const tbb::blocked_range< uint > &  r )
        {
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //-------------------------------------------------------------
                // Determine the column index set of the corresponding subblock
                //-------------------------------------------------------------
                const size_t col_first = S_tilde->col_is().first() + j*max_size;
                
                size_t col_last = S_tilde->col_is().first() + (j+1)*max_size - 1;
                
                if ( j == block_cols - 1 )
                    col_last = S_tilde->col_is().last();
                    
                const TIndexSet sub_col_is( col_first, col_last );
                
                //----------------------------------------------------------
                // Create reference to submatrix of S_tilde
                //----------------------------------------------------------
                BLAS::Matrix< real >  S_tilde_BLAS_sub ( S_tilde->blas_rmat(),
                                                         S_tilde->row_is() - S_tilde->row_ofs(),
                                                         sub_col_is        - S_tilde->col_ofs() );
            
                unique_ptr< TDenseMatrix > S_tilde_sub( new TDenseMatrix( S_tilde->row_is(), sub_col_is, S_tilde_BLAS_sub ) );
                                                
                //-----------------------------------------------------
                // Compute S_sub = op(L)^{-1} * S_tilde_sub
                //-----------------------------------------------------
                solve_lower_left( op_L, L, nullptr, S_tilde_sub.get(), acc_exact, solve_L_opts );
                
            }// for
        };
    
    
    //--------------------------------------------------
    // Handle small problems separately in order 
    // to avoid unnecessary parallel overhead 
    //--------------------------------------------------
    if ( block_cols == 1 || !_para_parallel.miscellaneous_parallel )
    {   
        solve_lower_left( op_L, L, nullptr, S_tilde, acc_exact, solve_L_opts );
    }// if
    else
    {
        tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(block_cols) ), compute_S_sub );
    }// else
}






//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Analyse routines
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




                          
void 
THAMLS::print_options () const
{
    OUT( "" );
    OUT( "(THAMLS) print_options :" );
    HLINE;
    OUT( to_string("    general options: n_ev_searched     = %d",_para_base.n_ev_searched ) );
    OUT( to_string("                     max_dof_subdomain = %d",_para_base.max_dof_subdomain ) );
    OUT( to_string("                     use_arnoldi       = %d",_para_base.use_arnoldi ) );
    OUT( to_string("                     do_condensing     = %d",_para_base.do_condensing ) );
    OUT( to_string("                     stable_rayleigh   = %d",_para_base.stable_rayleigh ) );
    OUT( to_string("                     do_debug          = %d",_para_base.do_debug ) );
    OUT( to_string("                     coarsen           = %d",_para_base.coarsen ) );
    HLINE;
    OUT( to_string("    subspace iter. : do_improving                = %d",_para_impro.do_improving ) );
    OUT( to_string("                     use_sparse_version          = %d",_para_impro.use_sparse_version ) );
    OUT( to_string("                     use_K_inv_as_precomditioner = %d",_para_impro.use_K_inv_as_precomditioner ) );
    OUT( to_string("                     number_of_iterations        = %d",_para_impro.number_of_iterations ) );
    HLINE;
    OUT( to_string("    parallel:        miscellaneous_parallel           = %d",_para_parallel.miscellaneous_parallel ) );
    OUT( to_string("                     condense_subproblems_in_parallel = %d",_para_parallel.condense_subproblems_in_parallel ) );
    OUT( to_string("                     K_and_M_red_in_parallel          = %d",_para_parallel.K_and_M_red_in_parallel ) );
    OUT( to_string("                     M_red_ij_step_1_in_parallel      = %d",_para_parallel.M_red_ij_step_1_in_parallel ) );
    OUT( to_string("                     M_red_ij_step_2_in_parallel      = %d",_para_parallel.M_red_ij_step_2_in_parallel ) );
    HLINE;
    OUT( to_string("    input/output:    load_problem = %d",_para_base.load_problem ) );
    OUT( to_string("                     save_problem = %d",_para_base.save_problem ) );
    OUT( to_string("                     io_location  = ") +_para_base.io_location );
    OUT( to_string("                     io_prefix    = ") +_para_base.io_prefix );
    HLINE;
    if ( get_mode_selection() == MODE_SEL_BOUND_H )
    {   
        OUT( to_string("    mode selection:  type    = MODE_SEL_BOUND" ) );
        OUT( to_string("                     bound   = %.2f",_para_mode_sel.trunc_bound ) );
        OUT( to_string("                     nmin_ev = %d",_para_mode_sel.nmin_ev ) );
    }// if
    
    if ( get_mode_selection() == MODE_SEL_REL_H )
    {
        OUT( to_string("    mode selection:  type    = MODE_SEL_REL_H" ) );
        OUT( to_string("                     rel     = %.2f",_para_mode_sel.rel ) );
        OUT( to_string("                     nmin_ev = %d",_para_mode_sel.nmin_ev ) );
    }// if
    
    if ( get_mode_selection() == MODE_SEL_ABS_H )
    {
        OUT( to_string("    mode selection:  type    = MODE_SEL_ABS_H" ) );
        OUT( to_string("                     abs     = %.2f",_para_mode_sel.abs ) );
        OUT( to_string("                     nmin_ev = %d",_para_mode_sel.nmin_ev ) );
    }//if
    
    if ( get_mode_selection() == MODE_SEL_AUTO_H )
    {
        OUT( to_string("    mode selection:  type               = MODE_SEL_AUTO" ) );
        OUT( to_string("                     factor subdomain   = %.2f",_para_mode_sel.factor_subdomain ) );
        OUT( to_string("                     factor interface   = %.2f",_para_mode_sel.factor_interface ) );
        OUT( to_string("                     exponent subdomain = %.2f",_para_mode_sel.exponent_subdomain ) );
        OUT( to_string("                     exponent interface = %.2f",_para_mode_sel.exponent_interface ) );
        OUT( to_string("                     nmin_ev            = %d",_para_mode_sel.nmin_ev ) );
    }// if
    
    if ( _para_base.do_condensing )
    {
        OUT( to_string("    recursion:       condensing size    = %d",_para_mode_sel.condensing_size ) );
        OUT( to_string("                     ratio_2_condense   = %.2f",_para_mode_sel.ratio_2_condense ) );
        OUT( to_string("                     factor condense    = %.2f",_para_mode_sel.factor_condense ) );
    }// if
    
    HLINE;
    OUT( to_string("    accuracy:        acc_transform_K ( %.1e, %.1e )", _trunc_acc.transform_K.rel_eps(),_trunc_acc.transform_K.abs_eps() ) );
    OUT( to_string("                     acc_transform_M ( %.1e, %.1e )", _trunc_acc.transform_M.rel_eps(),_trunc_acc.transform_M.abs_eps() ) );
    HLINE;
}    
   

/*
void 
THAMLS::print_matrix_norms ( const TMatrix * K,
                             const TMatrix * M, 
                             const TMatrix * L,
                             const TMatrix * K_tilde,
                             const TMatrix * M_tilde,
                             const bool      print_cond_number ) const
{
    if ( K == nullptr || M == nullptr || L == nullptr || K_tilde == nullptr || M_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) print_matrix_norms", "argument is nullptr" );
    
    TTimer timer( WALL_TIME );
    
    timer.start();
    //////////////////////////////////////////////////////////////////////
    //
    // print norm and condition number
    //
    //////////////////////////////////////////////////////////////////////
    TSpectralNorm  mnorm( 5, 20, real(1e-3) );
    
    const real norm_K = mnorm.norm( K );
    const real norm_M = mnorm.norm( M );
    const real norm_L = mnorm.norm( L );
    const real norm_K_tilde = mnorm.norm( K_tilde );
    const real norm_M_tilde = mnorm.norm( M_tilde );
    
    cout<<endl;
    cout<<endl<<"(THAMLS::print_matrix_norms)";
    cout<<endl;
    cout<<endl<<"\t ---------------------------------------";
    cout<<endl<<"\t |K|_2       = "<<norm_K;
    cout<<endl<<"\t |M|_2       = "<<norm_M;
    cout<<endl<<"\t |L|_2       = "<<norm_L;
    cout<<endl<<"\t |K_tilde|_2 = "<<norm_K_tilde;
    cout<<endl<<"\t |M_tilde|_2 = "<<norm_M_tilde;
    timer.pause();
    cout<<endl; 
    cout<<endl<<"\t --> matrix norms computed in "<<timer;
    cout<<endl<<"\t ---------------------------------------";
    
    //     cout<<endl<<"\t |K|_F       = "<<norm_F( K );
    //     cout<<endl<<"\t |M|_F       = "<<norm_F( M );
    //     cout<<endl<<"\t |K_tilde|_F = "<<norm_F( K_tilde );
    //     cout<<endl<<"\t |M_tilde|_F = "<<norm_F( M_tilde );
    
    if ( print_cond_number )
    {
        timer.start();
        cout<<endl<<"\t cond(K)       = "<<norm_K * mnorm.inv_norm( K );
        cout<<endl<<"\t cond(M)       = "<<norm_M * mnorm.inv_norm( M );
        cout<<endl<<"\t cond(L)       = "<<norm_L * mnorm.inv_norm( L );
        cout<<endl<<"\t cond(K_tilde) = "<<norm_K_tilde * mnorm.inv_norm( K_tilde );
        cout<<endl<<"\t cond(M_tilde) = "<<norm_M_tilde * mnorm.inv_norm( M_tilde );    
        timer.pause();
        cout<<endl; 
        cout<<endl<<"\t --> condition numbers computed in "<<timer;
        cout<<endl<<"\t ---------------------------------------";
    }// if
    
    cout<<endl;
    cout<<endl;
} 


void
THAMLS::print_avg_times ( const TMatrix * Z_approx ) const
{
    const size_t N    = Z_approx->rows();
    const size_t n_ev = Z_approx->cols();

    cout<<endl;
    cout<<endl<<"(THAMLS::print_avg_times)";
    cout<<endl;
    cout<<endl<<"\t - N    = "<<N;
    cout<<endl<<"\t - n_ev = "<<n_ev;
    cout<<endl<<"\t - definition of average time in paper (wall-time is used)";
    

    if ( _timer.T2_amls.cpu.elapsed() > 0 )
        cout<<endl<<"\t - transformation was done 'via amls'";
    
    if ( _timer.T2_ldl.cpu.elapsed() > 0 )
        cout<<endl<<"\t - transformation was done 'via ldl'";
        
    cout.precision(4);
    
    cout<<endl;
    cout<<endl<<"\t =============================";
    cout<<endl<<"\t avg( T4 large interface ) = "<<avg_time( _timer.T4_large_interface, Z_approx )<<"s";
    cout<<endl<<"\t avg( T4 small interface ) = "<<avg_time( _timer.T4_small_interface, Z_approx )<<"s";
    cout<<endl<<"\t avg( T4 subdomain )       = "<<avg_time( _timer.T4_subdomain, Z_approx )<<"s";
    cout<<endl<<"\t -----------------------------";
    cout<<endl<<"\t avg( T6_global )    = "<<avg_time( _timer.T6_global, Z_approx )<<"s";
    cout<<endl<<"\t avg( T7_global )    = "<<avg_time( _timer.T7_global, Z_approx )<<"s";
    cout<<endl<<"\t avg( T8_global )    = "<<avg_time( _timer.T8_global, Z_approx )<<"s";
    cout<<endl<<"\t -----------------------------";
    cout<<endl<<"\t avg( T6_SUB )    = "<<avg_time( _timer.T6_SUB, Z_approx )<<"s";
    cout<<endl<<"\t avg( T7_SUB )    = "<<avg_time( _timer.T7_SUB, Z_approx )<<"s";
    cout<<endl<<"\t avg( T8_SUB )    = "<<avg_time( _timer.T8_SUB, Z_approx )<<"s";
    cout<<endl<<"\t =============================";
    cout<<endl;
        
    
    cout<<endl<<"\t =====================";
    cout<<endl<<"\t avg( T1 )  = "<<avg_time( _timer.T1, Z_approx )<<"s";
    if ( _timer.T2_amls.cpu.elapsed() > 0 )
    {
        cout<<endl<<"\t avg( T2 )  = "<<avg_time( _timer.T2_amls, Z_approx )<<"s";
        cout<<endl<<"\t avg( T3 )  = "<<avg_time( _timer.T3_amls, Z_approx )<<"s";
    }// if
    
    if ( _timer.T2_ldl.cpu.elapsed() > 0 )
    {
        cout<<endl<<"\t avg( T2 )  = "<<avg_time( _timer.T2_ldl, Z_approx )<<"s";
        cout<<endl<<"\t avg( T3 )  = "<<avg_time( _timer.T3_ldl, Z_approx )<<"s";
    }// if
    
    cout<<endl<<"\t avg( T4 )  = "<<avg_time( _timer.T4, Z_approx )<<"s";
    cout<<endl<<"\t avg( T5 )  = "<<avg_time( _timer.T5, Z_approx )<<"s";
    cout<<endl<<"\t avg( T6 )  = "<<avg_time( _timer.T6, Z_approx )<<"s";
    cout<<endl<<"\t avg( T7 )  = "<<avg_time( _timer.T7, Z_approx )<<"s";
    cout<<endl<<"\t avg( T8 )  = "<<avg_time( _timer.T8, Z_approx )<<"s";
    cout<<endl<<"\t avg( TSI)  = "<<avg_time( _timer.TSI, Z_approx )<<"s";
    cout<<endl<<"\t avg( T9 )  = "<<avg_time( _timer.T9, Z_approx )<<"s";
    cout<<endl<<"\t ---------------------";
    cout<<endl<<"\t avg( all ) = "<<avg_time( _timer.all, Z_approx )<<"s";
    cout<<endl<<"\t =====================";
    cout<<endl;
    
    cout<<endl;
    if ( _timer.T2_amls.cpu.elapsed() > 0 )
        cout<<endl<<"\t avg( T2+T3 ) = "<<avg_time( _timer.T2_amls, Z_approx )+avg_time( _timer.T3_amls, Z_approx )<<"s";

    if ( _timer.T2_ldl.cpu.elapsed() > 0 )
        cout<<endl<<"\t avg( T2+T3 ) = "<<avg_time( _timer.T2_ldl, Z_approx )+avg_time( _timer.T3_ldl, Z_approx )<<"s";
}


void
THAMLS::print_output_gnuplot ( const TMatrix * Z_approx ) const
{
    const size_t N     = Z_approx->rows();
    const size_t n_ev  = Z_approx->cols();
    size_t       n_red = 0;
        
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        n_red = n_red + _S[i]->cols();
    }// for

    cout<<endl;
    cout<<endl<<"(THAMLS::print_output_gnuplot)";
    cout<<endl;
    //--------------------------------------------------------------------------------------------------------------------
    // Output for Gnuplot average times
    //--------------------------------------------------------------------------------------------------------------------
    cout<<endl;
    cout<<endl<<"\t ---> Output for Gnuplot: absolut times in seconds (WALL TIME)";
    cout<<endl;
    cout<<endl;
    cout<<"\t column_1:N  2:n_ev  3:(T4_large_interface)  4:(T6_global)  5:(T7_global)";
    cout<<"   6:(T8_global)  7:(T1)  8:(T2_LDL)  9:(T3_LDL)  10:(T4)  11:(T5)";
    cout<<"   12:(T6)  13:(T7)  14:(T8)  15:(TSI)  16:(T9)  17:(Tall)  18:k  19:#cores";
    cout<<"   20:(T_ref)  21: (T9_ref)";
    cout<<endl;
    cout<<endl;
    cout<<"\t "<< N<<"  "<< n_ev<<"  "<< _timer.T4_large_interface.wall;
    cout<<"  "<< _timer.T6_global.wall <<"  "<< _timer.T7_global.wall <<"  "<< _timer.T8_global.wall;
    cout<<"  "<< _timer.T1.wall <<"  "<< _timer.T2_ldl.wall <<"  "<< _timer.T3_ldl.wall;
    cout<<"  "<< _timer.T4.wall <<"  "<< _timer.T5.wall <<"  "<< _timer.T6.wall; 
    cout<<"  "<< _timer.T7.wall <<"  "<< _timer.T8.wall <<"  "<< _timer.TSI.wall;
    cout<<"  "<< _timer.T9.wall <<"  "<< _timer.all.wall;
    cout<<"  "<< n_red<<"  "<< CFG::nthreads();
    cout<<"  "<< _timer.T_reference.wall <<"  "<< _timer.T9_reference.wall;
    cout<<endl;
    cout<<endl;
    //--------------------------------------------------------------------------------------------------------------------
    // Output for Gnuplot average times
    //--------------------------------------------------------------------------------------------------------------------
    cout.precision(4);
    
    cout<<endl;
    cout<<endl<<"\t ---> Output for Gnuplot: average times (WALL TIME):";
    cout<<endl;
    cout<<endl;
    cout<<"\t column_1:N  2:n_ev  3:avg( T4_large_interface )  4:avg( T6_global )  5:avg( T7_global )";
    cout<<"   6:avg( T8_global )  7:avg(T1)  8:avg( T2_LDL )  9:avg( T3_LDL )  10:avg( T4 )  11:avg( T5 )";
    cout<<"   12:avg( T6 )  13:avg( T7 )  14:avg( T8 )  15:avg( TSI )  16:avg(T9)  17:avg( all )  18:k  19:#cores";
    cout<<"   20:avg( T_ref )  21:avg( T9_ref )";
    cout<<endl;
    cout<<endl;
    cout<<"\t "<< N<<"  "<< n_ev<<"  "<< avg_time( _timer.T4_large_interface, Z_approx );
    cout<<"  "<< avg_time( _timer.T6_global, Z_approx ) <<"  "<< avg_time( _timer.T7_global, Z_approx ) <<"  "<< avg_time( _timer.T8_global, Z_approx );
    cout<<"  "<< avg_time( _timer.T1, Z_approx ) <<"  "<< avg_time( _timer.T2_ldl, Z_approx ) <<"  "<< avg_time( _timer.T3_ldl, Z_approx );
    cout<<"  "<< avg_time( _timer.T4, Z_approx ) <<"  "<< avg_time( _timer.T5, Z_approx ) <<"  "<< avg_time( _timer.T6, Z_approx ); 
    cout<<"  "<< avg_time( _timer.T7, Z_approx ) <<"  "<< avg_time( _timer.T8, Z_approx ) <<"  "<< avg_time( _timer.TSI, Z_approx );
    cout<<"  "<< avg_time( _timer.T9, Z_approx ) <<"  "<< avg_time( _timer.all, Z_approx );
    cout<<"  "<< n_red<<"  "<< CFG::nthreads();
    cout<<"  "<< avg_time( _timer.T_reference, Z_approx ) <<"  "<< avg_time( _timer.T9_reference, Z_approx );
    cout<<endl;
    cout<<endl;
}





void 
THAMLS::print_performance ( const bool print_WALL_TIME ) const
{
//     cout<<std::fixed; cout.precision(2);

    if ( print_WALL_TIME )
    {        
        cout<<endl;
        cout<<endl<<"(THAMLS::print_performance) using WALL_TIME";
        cout<<endl;
        cout<<endl<<"\t => task_scheduler_init::default_num_threads() = "<<tbb::task_scheduler_init::default_num_threads();
        cout<<endl<<"\t => CFG::nthreads()                            = "<<CFG::nthreads();
        cout<<endl;
    } 
    else
    {
        cout<<endl;
        cout<<endl<<"(THAMLS::print_performance) using CPU_TIME";
        cout<<endl;
    }// else

    if ( _para_base.load_problem )
    {   
        cout<<endl<<"\t --> NOTE: matrices L, K_tilde, M_tilde have been loaded";
        cout<<endl;
    }// if
    else if ( _para_base.save_problem )
    {
        cout<<endl<<"\t --> NOTE: matrices L, K_tilde, M_tilde have been saved";
        cout<<endl;
    }// else if
        

    cout<<endl<<"\t =================================================================================================";
    cout<<endl<<"\t (T4) eigendecompostion of (K_ii,M_ii)     "<<print_perf_help( _timer.T4,                 _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -------------------------------------------------------------------------------------------------";
    cout<<endl<<"\t -->  subdomains                           "<<print_perf_help( _timer.T4_subdomain,       _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -->  small interfaces                     "<<print_perf_help( _timer.T4_small_interface, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -->  large interfaces                     "<<print_perf_help( _timer.T4_large_interface, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t =================================================================================================";
    cout<<endl<<"\t (T6) global: K_red and M_red computed     "<<print_perf_help( _timer.T6_global, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T7) global: eigendecomp. (K_red,M_red)   "<<print_perf_help( _timer.T7_global, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T8) global: backtransformation           "<<print_perf_help( _timer.T8_global, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -------------------------------------------------------------------------------------------------";
    cout<<endl<<"\t -->  (T6_SUB) K_red and M_red computed    "<<print_perf_help( _timer.T6_SUB, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -->  (T7_SUB) eigendecomp (K_red, M_red)  "<<print_perf_help( _timer.T7_SUB, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -->  (T8_SUB) backtransform eigenvectors  "<<print_perf_help( _timer.T8_SUB, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t =================================================================================================";
    if ( _para_parallel.condense_subproblems_in_parallel )
    {   
        cout<<endl;
        cout<<endl<<"\t --> the above measured times are difficult to interpret if more than 1 thread is used";
        cout<<endl<<"\t --> speedup and efficiency are not really correct since WALL and CPU time have been used for measurement";
        cout<<endl;
    }// if
    cout<<endl;
    cout<<endl;
    cout<<endl<<"\t =================================================================================================";
    cout<<endl<<"\t (T1) construction of H-matrices           "<<print_perf_help( _timer.T1, _timer.all, print_WALL_TIME ); 
    
    
    if ( !_para_base.load_problem )
    {
        if ( _timer.T2_amls.wall.elapsed() > 0 )
        {
            cout<<endl<<"\t (T2) K_tilde computed 'via amls'          "<<print_perf_help( _timer.T2_amls, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t (T3) M_tilde computed 'via amls'          "<<print_perf_help( _timer.T3_amls, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t      -->  comp. of L^{-1}*M in            "<<print_perf_help( _timer.solve_lower_left_amls, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t      -->  comp. of L^{-1}*M*L^{-T} in     "<<print_perf_help( _timer.solve_lower_right_amls, _timer.all, print_WALL_TIME );
        }// if
        
        if ( _timer.T2_ldl.wall.elapsed() > 0 )
        {
            cout<<endl<<"\t (T2) K_tilde computed 'via ldl'           "<<print_perf_help( _timer.T2_ldl, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t (T3) M_tilde computed 'via ldl'           "<<print_perf_help( _timer.T3_ldl, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t      -->  comp. of L^{-1}*M in            "<<print_perf_help( _timer.solve_lower_left_ldl, _timer.all, print_WALL_TIME );
            cout<<endl<<"\t      -->  comp. of L^{-1}*M*L^{-T} in     "<<print_perf_help( _timer.solve_lower_right_ldl, _timer.all, print_WALL_TIME );
        }// if
    }// if
    else
    {
        cout<<endl<<"\t (T2) L,K_tilde,M_tilde have been loaded in "<<print_perf_help( _timer.load_problem, _timer.all, print_WALL_TIME );
        cout<<endl<<"\t (T3)            - \" -                     ";
    }// else
    
    cout<<endl<<"\t (T4) eigendecompostion of (K_ii,M_ii)     "<<print_perf_help( _timer.T4,          _timer.all, print_WALL_TIME ); 
    cout<<endl<<"\t (T5) condensing                           "<<print_perf_help( _timer.T5,          _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T6) K_red and M_red computed             "<<print_perf_help( _timer.T6,          _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T7) eigendecompostion of (K_red,M_red)   "<<print_perf_help( _timer.T7,          _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T8) backtransformation of eigenvectors   "<<print_perf_help( _timer.T8,          _timer.all, print_WALL_TIME );
    cout<<endl<<"\t      -->  transf. with diag[S_1,...,S_m]  "<<print_perf_help( _timer.T8_with_S_i, _timer.all, print_WALL_TIME );
    cout<<endl<<"\t      -->  transf. with L^{-T}             "<<print_perf_help( _timer.T8_with_L,   _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (TSI) additional subspace iteration       "<<print_perf_help( _timer.TSI,         _timer.all, print_WALL_TIME );
    cout<<endl<<"\t (T9) computation of Rayleigh quotients    "<<print_perf_help( _timer.T9,          _timer.all, print_WALL_TIME );
    cout<<endl<<"\t -------------------------------------------------------------------------------------------------";
    cout<<endl<<"\t      total time needed                    "<<print_perf_help( _timer.all,         _timer.all, print_WALL_TIME );
    cout<<endl<<"\t =================================================================================================";
    cout<<endl;
    cout<<endl;
    
    if ( _para_impro.do_improving )
    {
        cout<<endl<<"\t -->  performance of additional subspace iteration for eigenvector improvisation";
        cout<<endl<<"\t      --------------------------------------------------------------------------";
        cout<<endl<<"\t      do_improving                = "<<_para_impro.do_improving;
        cout<<endl<<"\t      use_sparse_version          = "<<_para_impro.use_sparse_version;
        cout<<endl<<"\t      use_K_inv_as_precomditioner = "<<_para_impro.use_K_inv_as_precomditioner;
        cout<<endl<<"\t      number_of_iterations        = "<<_para_impro.number_of_iterations;
        cout<<endl<<"\t      ============================================================================================";
        cout<<endl<<"\t      iteration of the old eigenvectors    "<<print_perf_help( _timer.TSI_iter,           _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t           -->  transform with M           "<<print_perf_help( _timer.TSI_iter_with_M,    _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t           -->  make copy of matrix M*Q    "<<print_perf_help( _timer.TSI_iter_copy_MQ,   _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t           -->  transform with L^{-1}      "<<print_perf_help( _timer.TSI_iter_with_L_inv,_timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t           -->  transform with K^{-1}      "<<print_perf_help( _timer.TSI_iter_with_K_inv,_timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t      computation of reduced matrices      "<<print_perf_help( _timer.TSI_reduce,         _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t      eigendecomposition of reduced EVP    "<<print_perf_help( _timer.TSI_eigen,          _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t      back transform new eigenvectors      "<<print_perf_help( _timer.TSI_transform,      _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t      --------------------------------------------------------------------------------------------";
        cout<<endl<<"\t      total time needed for (TSI)          "<<print_perf_help( _timer.TSI,                _timer.TSI, print_WALL_TIME );
        cout<<endl<<"\t      ============================================================================================";
        cout<<endl;
        cout<<endl;
    }// if
} 

*/

void
THAMLS::print_structural_info () const
{
    unique_ptr< subprob_data_t > subprob_data( new subprob_data_t );
    
    subprob_data->first = 0;
    subprob_data->last  = _sep_tree.n_subproblems() - 1;
    
    print_structural_info_SUB ( subprob_data.get(), false );
}

                     
                             
                             
void
THAMLS::test_problem_transformation ( const TMatrix * K,
                                      const TMatrix * M,
                                      const TMatrix * K_tilde,
                                      const TMatrix * M_tilde,
                                      const TMatrix * L ) 
{    
    //
    // Choose fine accuracy
    //
    const TTruncAcc acc( 1e-8 ); 
    
    ////////////////////////////////////////////////
    //
    // Compute K_tilde_test = L^{-1} * K * L^{-T}
    //
    ////////////////////////////////////////////////
    TMatrix * K_tilde_test = nullptr;
    
    
    transform_M_version_ldl  ( K, K_tilde_test, L );
        
    //
    // Compute errors
    //    
    const real abs_error_K_tilde = diffnorm_2( K_tilde_test , K_tilde );
    const real rel_error_K_tilde = abs_error_K_tilde/norm_2( K_tilde );
    
    
    
    ///////////////////////////////////////////////////////////////
    //
    // Compute K_big_test = L * ( L^{-1} * K * L^{-T} ) * L^T
    //
    ///////////////////////////////////////////////////////////////
    
    unique_ptr< TMatrix >  Temp( EIGEN_MISC::get_nonsym_of_sym( K ) );
    
    unique_ptr< TMatrix >  K_big_test( K->copy() );
    
    multiply( real(1), MATOP_NORM, L, MATOP_NORM, K_tilde_test, real(0), Temp.get(), acc );        
    
    multiply( real(1), MATOP_NORM, Temp.get(), MATOP_TRANS, L, real(0), K_big_test.get(), acc );
        
    //
    // Compute errors
    //    
    const real abs_error_K_big = diffnorm_2( K_big_test.get() , K );
    const real rel_error_K_big = abs_error_K_big/norm_2( K );
    
    
    ////////////////////////////////////////////////
    //
    // Compute M_test = L * M_tilde * L^T
    //
    ////////////////////////////////////////////////
    
    unique_ptr< TMatrix >  Temp_M( EIGEN_MISC::get_nonsym_of_sym( M_tilde ) );
    
    unique_ptr< TMatrix >  M_test( M_tilde->copy() );
    
    multiply( real(1), MATOP_NORM, L, MATOP_NORM, M_tilde, real(0), Temp_M.get(), acc );        
    
    multiply( real(1), MATOP_NORM, Temp_M.get(), MATOP_TRANS, L, real(0), M_test.get(), acc );
        
    //
    // Compute errors
    //    
    const real abs_error_M = diffnorm_2( M_test.get() , M );
    const real rel_error_M = abs_error_M/norm_2( M );
    
    ////////////////////////////////////////////////
    //
    // Compute K_test = L * K_tilde * L^T
    //
    ////////////////////////////////////////////////
    
    ///WARNING: K_tilde is blockdiagonal ==> don't use this H-matrix structure for the temporay result and the result
    unique_ptr< TMatrix >  Temp_K( EIGEN_MISC::get_nonsym_of_sym( K ) );
    
    unique_ptr< TMatrix >  K_test( K->copy() );
    
    multiply( real(1), MATOP_NORM, L, MATOP_NORM, K_tilde, real(0), Temp_K.get(), acc );        
    
    multiply( real(1), MATOP_NORM, Temp_K.get(), MATOP_TRANS, L, real(0), K_test.get(), acc );
        
    //
    // Compute errors
    //    
    const real abs_error_K = diffnorm_2( K_test.get() , K );
    const real rel_error_K = abs_error_K/norm_2( K );
    
    OUT( "" );
    OUT( "(THAMLS) test_problem_transformation :" );
    HLINE;    
    OUT( "    The applied matrix transformation is as follows using classical LDL-decomposition" ); 
    OUT( "    K       = L * K_tilde * L^{T}  (computed by block-diagonalisation routine)" ); 
    OUT( "    M_tilde = L^{-1} * M * L^{-T}  (computed by routine solving block triangular matrix equation)" );
    OUT( "    and the following norms should be zero" );
    HLINE;
    OUT( to_string("    abs_error := |L^{-1}*K*L^{-T} - K_tilde|_2 = %.2e",abs_error_K_tilde) );
    OUT( to_string("    rel_error := abs_error/|K_tilde|_2         = %.2e",rel_error_K_tilde) );
    HLINE;
    OUT( to_string("    abs_error := |L*M_tilde*L^{T} - M|_2       = %.2e",abs_error_M) );
    OUT( to_string("    rel_error := abs_error/|M|_2               = %.2e",rel_error_M) );
    HLINE;
    OUT( to_string("    abs_error := |L*K_tilde*L^{T} - K|_2       = %.2e",abs_error_K) );
    OUT( to_string("    rel_error := abs_error/|K|_2               = %.2e",rel_error_K) );
    HLINE;
    OUT( to_string("    abs_error := |L*( L^{-1}*K*L^{-T} )*L^{T} - K|_2 = %.2e",abs_error_K_big) );
    OUT( to_string("    rel_error := abs_error/|K|_2                     = %.2e",rel_error_K_big) );
    HLINE;
    
    if ( false )
    {
        ////////////////////////////////////////////////
        //
        // Some ouputs
        //       
        ////////////////////////////////////////////////
        if ( true )
        {
//             EIGEN_MISC::print_matrix( K );
//             EIGEN_MISC::print_matrix( M );
//             EIGEN_MISC::print_matrix( L );
//             EIGEN_MISC::print_matrix( K_tilde );
//             EIGEN_MISC::print_matrix( K_tilde_test );
//             EIGEN_MISC::print_matrix( M_tilde );
        }// if
        
        ////////////////////////////////////////////////
        //
        // Compute K_tilde_test = K_tilde - K_tilde_test
        //       
        ////////////////////////////////////////////////
        add ( -real(1), K_tilde, real(1), K_tilde_test, acc );
        
//         cout<<endl<<"==> matrix ouput of 'K_tilde - L^{-1}*K*L^{-T}'";
//         EIGEN_MISC::print_matrix( K_tilde_test );

        
        ////////////////////////////////////////////////
        //
        // Compute M_test = M - M_test
        //       
        ////////////////////////////////////////////////
        add ( -real(1), M, real(1), M_test.get(),acc );
        
//         cout<<endl<<"==> matrix ouput of 'M - L*M_tilde*L^{T}'";
//         EIGEN_MISC::print_matrix( M_test.get() );
        
        ////////////////////////////////////////////////
        //
        // Compute K_test = K - K_test
        //       
        ////////////////////////////////////////////////
        add ( -real(1), K, real(1), K_test.get(),acc );
        
//         cout<<endl<<"==> matrix ouput of 'K - L*K_tilde*L^{T}'";
//         EIGEN_MISC::print_matrix( K_test.get() );
        
    }// if
    
    ////////////////////////////////////////////////
    //
    // clean up
    //
    ////////////////////////////////////////////////
    delete K_tilde_test;
}
                             
                             
                             
              
                             
void
THAMLS::print_summary ( const TMatrix * D_approx ) const
{
    //---------------------------------------------------------
    // Compute size of the reduced problem and original problem
    //---------------------------------------------------------
    size_t n_red = 0;
    size_t n     = 0;
    
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        n_red = n_red + _S[i]->cols();
        n     = n     + _sep_tree.get_dof_is(i).size();
        //NOTE: The impelementation 
        //n     = n     + _S[i]->rows();
        // computes the wrong result when _S[i] has zero columns, since in this case 
        // S[i] would have as well zero rows which would lead to the wrong results.
    }// for
    
    OUT( "" );
    OUT( "(THAMLS) print_summary :" );
    HLINE;
    OUT( to_string("    H-AMLS computed the reduced EVP (K_red,M_red) of the original EVP (K,M)") );
    OUT( to_string("    (D,Z) are eigenpair approximations for (K,M) received from (K_red,M_red)") );
    HLINE;
    OUT( to_string("    dimension of (K,M)         = %d",n) ); 
    OUT( to_string("    dimension of (K_red,M_red) = %d",n_red) );  
    OUT( to_string("    %d of %d possible eigenpair approximations have been computed",D_approx->rows(),n_red) );
    HLINE;
}


void 
THAMLS::print_matrix_memory ( const TSparseMatrix * K_sparse, 
                              const TSparseMatrix * M_sparse, 
                              const TMatrix *       K,
                              const TMatrix *       M,
                              const TMatrix *       K_tilde,
                              const TMatrix *       M_tilde,
                              const TMatrix *       L,
                              const TMatrix *       K_red,
                              const TMatrix *       M_red,
                              const TMatrix *       Z_red,
                              const TMatrix *       D_approx,
                              const TMatrix *       Z_approx ) const
{
    TIC;
    
    ///////////////////////////////////////////////////////////
    //
    // Computation of memory consumption of S = [S_1,...,S_m]
    //
    ///////////////////////////////////////////////////////////
    size_t sum_mem_D = 0;
    size_t sum_mem_S = 0;
    
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        sum_mem_D += _D[i]->byte_size();
        sum_mem_S += _S[i]->byte_size();
    }// for
    
    ///////////////////////////////////////////////////////////
    //
    // Output
    //
    ///////////////////////////////////////////////////////////
    
    OUT( "" );
    OUT( "(THAMLS) print_matrix_memory : " );
//     OUT( "(THAMLS) print_matrix_memory : using binary prefixes --> using decimal prefixes" );
    if ( K_sparse != nullptr && M_sparse != nullptr )
    {
        HLINE;
        OUT( "    size of K_sparse         = "+tot_dof_size( K_sparse ));//+" --> "+mem_in_MB( K_sparse ) );
        OUT( "    size of M_sparse         = "+tot_dof_size( M_sparse ));//+" --> "+mem_in_MB( M_sparse ) );
    }// if
    HLINE;
    OUT( "    size of H-matrix K       = "+tot_dof_size( K ));//+" --> "+mem_in_MB( K ) );
    OUT( "    size of H-matrix M       = "+tot_dof_size( M ));//+" --> "+mem_in_MB( M ) );
    OUT( "    size of H-matrix K_tilde = "+tot_dof_size( K_tilde ));//+" --> "+mem_in_MB( K_tilde ) );
    OUT( "    size of H-matrix M_tilde = "+tot_dof_size( M_tilde ));//+" --> "+mem_in_MB( M_tilde ) );
    OUT( "    size of H-matrix L       = "+tot_dof_size( L ));//+" --> "+mem_in_MB( L ) );
    HLINE;
    OUT( "    size of D_1,...,D_m      = "+Mem::to_string( sum_mem_D ) );
    OUT( "    size of S_1,...,S_m      = "+Mem::to_string( sum_mem_S ) );
    HLINE;
    OUT( "    size of K_red            = "+tot_dof_size( K_red ));//+" --> "+mem_in_MB( K_red ) );
    OUT( "    size of M_red            = "+tot_dof_size( M_red ));//+" --> "+mem_in_MB( M_red ) );
    OUT( "    size of Z_red            = "+tot_dof_size( Z_red ));//+" --> "+mem_in_MB( Z_red ) );
    HLINE;
    OUT( "    size of D_approx         = "+tot_dof_size( D_approx ));//+" --> "+mem_in_MB( D_approx ) );
    OUT( "    size of Z_approx         = "+tot_dof_size( Z_approx ));//+" --> "+mem_in_MB( Z_approx ) );
    LOGHLINE;
    TOC;
    LOG( to_string("    computations done in %.2fs", toc.seconds()) );
    HLINE;
}
    
    


void 
THAMLS::visualize_biggest_interface_problem ( const TMatrix * M,
                                              const string &  filename_M,
                                              const bool      show_full_sym ) const
{
    const TMatrix * M_ii = nullptr;
    
    const size_t i = _sep_tree.n_subproblems() - 1;
    
    get_subproblem_matrix( M, i, M_ii );
    
    visualize_matrix ( M_ii, filename_M, show_full_sym );
}





void 
THAMLS::visualize_matrix ( const TMatrix * M,
                           const string &  filename,
                           const bool      show_full_sym ) const
{
    TPSMatrixVis    mvis;
    
    mvis.entries ( false );
    mvis.rank_col( false );
    mvis.svd     ( false );
    
    const string location = _para_base.io_location;
    const string prefix   = _para_base.io_prefix;
    
    if ( show_full_sym )
    {
        if ( ! M->is_symmetric() ) 
            HERROR( ERR_CONSISTENCY, "(THAMLS) visualize_matrix", "matrix is not symmetric" ); 
    
        unique_ptr< TMatrix >  M_show( EIGEN_MISC::get_nonsym_of_sym( M ) );
        
        mvis.print( M_show.get(), location + prefix + filename );
    }// if
    else
    {
        mvis.print( M, location + prefix + filename );
    }// else
    
    /////////////////////////////////////////////
    //
    // Make notice that matices are visualised
    //
    /////////////////////////////////////////////
    LOG( "" );
    LOG( "(THAMLS) visualize_matrix :" );
    LOGHLINE;
    LOG( to_string("    rows     = ",M->rows()) );
    LOG( "    memory   = "+tot_dof_size( M ) );
    LOG( "    filename = '"+location + prefix + filename+"'" );
    LOGHLINE;
}

void 
THAMLS::visualize_cluster ( const TCluster * cl,
                            const string &   filename ) const
{
    TPSClusterVis            cvis;
    
    const string location = _para_base.io_location;
    const string prefix   = _para_base.io_prefix;
    
    cvis.print( cl, location + prefix + filename );
    
    /////////////////////////////////////////////
    //
    // Make notice that cluster is visualised
    //
    /////////////////////////////////////////////
    LOG( "" );
    LOG( "(THAMLS) visualize_cluster :" );
    LOGHLINE;
    LOG( "    size     = "+cl->size() );
    LOG( "    filename = '"+location + prefix + filename+"'" );
    LOGHLINE;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// THAMLS public
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

size_t 
THAMLS::comp_decomp ( const TClusterTree *   ct,
                      const TMatrix *        K,
                      const TMatrix *        M,
                      TDenseMatrix  *        D,
                      TDenseMatrix  *        Z,
                      const TSparseMatrix *  K_sparse, 
                      const TSparseMatrix *  M_sparse )
{
   
    //----------------------------------------------------------------------------------
    // Consistency checks
    //----------------------------------------------------------------------------------
    if ( ct == nullptr || M == nullptr || K == nullptr || D == nullptr || Z == nullptr )
       HERROR( ERR_ARG, "(THAMLS) comp_decomp", "argument is nullptr" );
       
    if ( !K->is_symmetric() || !M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_decomp", "matrices are not symmetric" ); 
            
    //=========================================================================================================
    //
    // Initialize the auxiliary data structures 
    //
    //=========================================================================================================
    timer_HAMLS_t timer;
    TICC( timer_all );
    
    init_amls_data ( ct );
    
    
    //=========================================================================================================
    //
    // Task (T2) & (T3): Block diagonalise the matrix K and transform the matrix M correspondingly 
    //                   (Note: The block diagonalisation can be changed slightly. Also a LDL 
    //                   diagonalization is possible)
    //
    //=========================================================================================================
    
    TMatrix * L       = nullptr;
    TMatrix * K_tilde = nullptr;
    TMatrix * M_tilde = nullptr;
    
    TICC( timer_T2_T3 );
    if ( _para_base.load_problem )
        load_transformed_problem ( K_tilde, M_tilde, L );
    else
        transform_problem ( K, M, K_tilde, M_tilde, L );   
    TOCC( timer_T2_T3, timer.T2_T3 );
    
    //---------------------------------------------------------
    // Consistency checks
    //---------------------------------------------------------
    if ( !K_tilde->is_symmetric() || !M_tilde->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_decomp", "matrices are not symmetric" ); 
        
    if ( K_tilde->rows() != K->rows() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_decomp", "matrices have different sizes" );
             
    if ( M_tilde->rows() != M->rows() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_decomp", "matrices have different sizes" );     
        
    //--------------------------------------------------------
    // Save transformed problem
    //--------------------------------------------------------
    if ( !_para_base.load_problem && _para_base.save_problem )
        save_transformed_problem ( K_tilde, M_tilde, L );
        
    //=========================================================================================================
    //
    // Task (T4): Compute the partial eigensolutions of the subproblems
    //
    //=========================================================================================================
    
    TICC( timer_T4 );
    comp_partial_eigensolutions( K_tilde, M_tilde );
    TOCC( timer_T4, timer.T4 );

    //=========================================================================================================
    //
    // Condensing: Apply the recursive H-AMLS method by condensing subproblems
    //
    //=========================================================================================================
    if ( _para_base.do_condensing )
    {
        TICC( timer_condensing );
        apply_condensing ( K_tilde, M_tilde );
        TOCC( timer_condensing, timer.condensing );
    }// if
         
    //=========================================================================================================
    //
    // Task (T6): Compute the matrices of the reduced eigenvalue problem (K_red,M_red)
    //
    //=========================================================================================================
    unique_ptr< TDenseMatrix >  K_red( new TDenseMatrix() );
    unique_ptr< TDenseMatrix >  M_red( new TDenseMatrix() ); 
        
    TICC( timer_T6 );
    comp_reduced_matrices( K_tilde, M_tilde, K_red.get(), M_red.get() );
    TOCC( timer_T6, timer.T6 );
    
    if ( _para_base.do_debug && false )
    {
        if ( EIGEN_MISC::has_inf_nan_entry ( K_red.get() ) )
            HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "K_red has bad data" );
            
        if ( EIGEN_MISC::has_inf_nan_entry ( M_red.get() ) )
            HERROR( ERR_CONSISTENCY, "(THAMLS) comp_partial_eigensolutions", "M_red has bad data" );
    }// if
    

    //=========================================================================================================
    //
    // Task (T7): Compute the eigensolution of the reduced problem (K_red,M_red)
    //
    //=========================================================================================================
    unique_ptr< TDenseMatrix >  Z_red( new TDenseMatrix );      
    
    TICC( timer_T7 );
    eigen_decomp_reduced_problem( K_red.get(), M_red.get(), D, Z_red.get() );
    TOCC( timer_T7, timer.T7 );
    
    //=========================================================================================================
    //
    // Task (T8): Transform the eigenvectors of the reduced eigenvalue problem
    //            to eigenvector approximations of the original eigenvalue problem
    //            and compute the Rayleigh quotients associated to these vectors
    //
    //            Apply additional subspace iteration if wanted in order to improve 
    //            the eigenvectors computed by H-AMLS
    //=========================================================================================================
    TICC( timer_T8 );
    transform_eigensolutions ( K_sparse, M_sparse, ct, K_tilde, M_tilde, L, Z_red.get(), D, Z );
    TOCC( timer_T8, timer.T8 );
             
    //--------------------------------------------------------------------------------
    // Set the indexsets of the matrices D and Z received from the eigen decomposition 
    // (The indexsets are defined in this style that the matrix operation K*Z-M*Z*D or
    //  K*Z-Z*D is well defined)
    //--------------------------------------------------------------------------------
    D->set_ofs( K->col_ofs(), K->col_ofs() );
    Z->set_ofs( K->col_ofs(), K->col_ofs() );
    
    TOCC( timer_all, timer.all );
    
    if ( _para_base.print_info )
    {
        print_options();
        print_structural_info();
        print_summary( D );
        
        #if DO_TEST >= 1
        timer.print_performance();
        print_matrix_memory ( K_sparse, M_sparse, K, M, K_tilde, M_tilde, L, K_red.get(), M_red.get(), Z_red.get(), D, Z );
        #endif
    }// if
     
    //=========================================================================================================
    //
    // Clean data and return number of computed eigenpair approximations
    //
    //=========================================================================================================
    delete_amls_data();
    
    delete L;
    delete K_tilde;
    delete M_tilde;
    
    if ( _para_impro.K_preconditioner != nullptr ) { delete _para_impro.K_preconditioner; _para_impro.K_preconditioner = nullptr; }
    if ( _para_impro.K_factorised     != nullptr ) { delete _para_impro.K_factorised;     _para_impro.K_factorised     = nullptr; }
    
    return D->rows();
}







//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!///
//!///
//!///
//!///  Routines for the recursive approach 
//!///
//!///
//!///
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//!////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///NOTE: Diese Datenstruktur(genauso wie auch der Separator Tree) sind an einen ClusterTree bzw. Cluster
///gebunden (da sie Pointer darauf besitzen). Daher aufpassen, dass hier nichts voreilig geloescht wird!
void
THAMLS::get_subprob_data ( const size_t      lvl,
                           TCluster *        cluster,
                           subprob_data_t *  subprob_data ) const
{
    if ( cluster == nullptr || subprob_data == nullptr )
        HERROR( ERR_ARG, "(THAMLS) get_subprob_data", "argument is nullptr" );
    
    //
    // Initialise the 'subprob_data_t' data and make a consistency check
    //    
    bool subproblem_found = false;
    
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        const TIndexSet subproblem_is = _sep_tree.get_dof_is( i );
    
        if ( cluster->is_sub( subproblem_is ) )
        {
            subprob_data->first = i;
            subprob_data->last  = i;
            
            subproblem_found = true;
            
            break;
        }// if
    }//for
    
    if ( !subproblem_found )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subprob_data", "" ); 
    
    //
    // Get all subproblem indices which are associated
    // with the given cluster and make a consistency check
    //
    size_t count_problems = 0;
    
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        const TIndexSet subproblem_is = _sep_tree.get_dof_is( i );
    
        if ( cluster->is_sub( subproblem_is ) )
        {
            if ( i < subprob_data->first )
                subprob_data->first = i;
                
            if ( i > subprob_data->last )
                subprob_data->last = i;
            
            count_problems++;
        }// if
    }//for
    
    if ( count_problems != (subprob_data->last - subprob_data->first + 1) )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subprob_data", "" );
        
    //
    // Assign cluster
    //
    if ( subprob_data->cluster != nullptr )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_subprob_data", "" );
        
    subprob_data->cluster = cluster;
    
    //
    // Assign level
    //
    subprob_data->level = lvl;
}

size_t
THAMLS::get_reduced_dofs_SUB ( const subprob_data_t *  subprob_data ) const
{
    if ( _reduced_subproblem_ofs[0] == -1 )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_reduced_dofs_SUB", "data has not been initialised" );
    
    if ( _reduced_subproblem_ofs.size() == 0 || _reduced_subproblem_ofs.size() != _sep_tree.n_subproblems() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_reduced_dofs_SUB", "" );
        
    const size_t first = subprob_data->first;
    const size_t last  = subprob_data->last;

    const size_t end   = size_t(_reduced_subproblem_ofs[last]) + _S[last]->cols();
    const size_t start = size_t(_reduced_subproblem_ofs[first]);
    
    if ( start > end || _reduced_subproblem_ofs[first] < 0 )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_reduced_dofs_SUB", "" );
    
    const size_t dofs_reduced = end - start;
    
    return dofs_reduced;
}



size_t
THAMLS::get_dofs_SUB ( const subprob_data_t *  subprob_data ) const
{
    if ( _subproblem_ofs[0] == -1 )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_dofs_SUB", "data has not been initialised" );
    
    if ( _subproblem_ofs.size() == 0 || _subproblem_ofs.size() != _sep_tree.n_subproblems() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_dofs_SUB", "" );

    const size_t first = subprob_data->first;
    const size_t last  = subprob_data->last;
    
    const size_t size_of_last_subproblem = _sep_tree.get_dof_is(last).size();

    const size_t end   = size_t(_subproblem_ofs[last]) + size_of_last_subproblem;
    const size_t start = size_t(_subproblem_ofs[first]);
    
    if ( start > end || _subproblem_ofs[first] < 0 )
        HERROR( ERR_CONSISTENCY, "(THAMLS) get_dofs_SUB", "" );
    
    const size_t dofs = end - start;
    
    return dofs;
}






bool
THAMLS::shall_problem_be_condensed ( const size_t            lvl,
                                     const size_t            max_lvl,
                                     const subprob_data_t *  subprob_data ) const
{
    bool problem_should_be_condensed = false;
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Choose among different selection strategies
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////
    if ( true ) ///NOTE: Standard Approach
    {
        //--------------------------------------------------------------------------------------
        // Check if the problem associated to the cluster is so big that it should be 
        // condensed. If this is the case append these subsproblems to the corresponding list
        //--------------------------------------------------------------------------------------
        const int n = get_dofs_SUB ( subprob_data );
            
        //
        // Determine the number of needed eigenpairs to represent the spectral
        // information of the subrproblems contained in 'subprob_data'
        //
        const double exponent = _para_mode_sel.exponent_subdomain;
                
        const size_t k_max = int(_para_mode_sel.ratio_2_condense * _para_mode_sel.factor_condense * Math::pow( real(n) , exponent ) );
        
        const size_t k      = get_reduced_dofs_SUB ( subprob_data );
                    
        bool too_much_spectral_information = false;
        
        if ( k_max < k  )
            too_much_spectral_information = true;
                    
    #if 1
        if ( k >= _para_mode_sel.condensing_size && too_much_spectral_information )
           problem_should_be_condensed = true;   
    #else
        if ( lvl % 3 == 0 && too_much_spectral_information && lvl > 0 )
            problem_should_be_condensed = true;

        // Addtitional criterion
        if ( (max_lvl - lvl) < 3 )
            problem_should_be_condensed = false;
    #endif 
    }// if
            
    
    if ( false )
    {
        
        //--------------------------------------------------------------------------------------
        // Condense all problems every 4 level
        //              <==> 
        // apply H-AMLS recursively every 4 level
        //--------------------------------------------------------------------------------------
        
        if ( lvl % 4 == 0 )
            problem_should_be_condensed = true;
            
        // Addtitional criterion
        if ( (max_lvl - lvl) < 2 )
            problem_should_be_condensed = false;
    }// if
    
    
    if ( false )
    {
        //--------------------------------------------------------------------------------------
        // Condense only on level 1 ---> for Debugging
        //--------------------------------------------------------------------------------------
        
        if ( lvl == 1 )
            problem_should_be_condensed = true;
    }// if
    
    
    

    return problem_should_be_condensed;
}



bool
THAMLS::find_subproblems_which_shall_be_condensed ( list< subprob_data_t * > & subproblems_2_condense ) const
{
    const size_t max_lvl = _root_amls_ct->depth();
    
    //-------------------------------------------------------------
    // if max_lvl is less or equal to 1 no problem can be condensed
    //-------------------------------------------------------------
    if ( max_lvl <= 1 )
        return false;
    
    /////////////////////////////////////////////////////////
    //
    // Look from bottom to the top of the AMLS cluster tree 
    // for subproblems/clusters which should be condensed
    //
    /////////////////////////////////////////////////////////
    bool problem_found = false;
    

    for ( size_t l = max_lvl-1; l >  0; l-- )
    {
        list< TCluster * >  cluster_list_on_lvl;
        
        _root_amls_ct->collect_leaves ( cluster_list_on_lvl, l, 0 );
    
        TCluster * cluster;
        
        while ( !cluster_list_on_lvl.empty() )
        {
            cluster = cluster_list_on_lvl.front();

            cluster_list_on_lvl.pop_front();
                
            //----------------------------------------------------
            // Interface cluster cannot be condensed
            //----------------------------------------------------
            if ( !cluster->is_domain() )
                continue;
                
            //----------------------------------------------------
            // Subproblems without sons don't have to be condensed
            //----------------------------------------------------
            if ( cluster->is_leaf() )
                continue;
                
            //----------------------------------------------------
            // Get the subproblems associated to the cluster
            //----------------------------------------------------
            subprob_data_t * subprob_data = new subprob_data_t;
            
            get_subprob_data( l, cluster, subprob_data );
            
            //----------------------------------------------------
            // If problem shall be condensed append it to the list
            //----------------------------------------------------
            if ( shall_problem_be_condensed( l, max_lvl, subprob_data ) )
            {
                subproblems_2_condense.push_back( subprob_data );
                    
                problem_found = true;
            }//if
        }// while
        
        if ( problem_found )
            break;
    }// for
    
    return problem_found;
}



size_t 
THAMLS::comp_K_and_M_red_parallel ( const TMatrix *         K_tilde,
                                    const TMatrix *         M_tilde,
                                    TDenseMatrix *          K_red_sub,
                                    TDenseMatrix *          M_red_sub,
                                    const subprob_data_t *  subprob_data )  
{
    if ( K_tilde == nullptr || M_tilde == nullptr || K_red_sub == nullptr || M_red_sub == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_K_and_M_red_parallel", "argument is nullptr" );
        
    //////////////////////////////////////////////////////////////////////////////
    //
    // Initialize the reduced matrices 'K_red_sub' and 'M_red_sub'
    //
    //////////////////////////////////////////////////////////////////////////////
    const size_t n_red = get_reduced_dofs_SUB ( subprob_data );
    
    const size_t ofs   = _reduced_subproblem_ofs[ subprob_data->first ];
        
    K_red_sub->set_size( n_red, n_red );
    K_red_sub->set_ofs ( ofs,   ofs   );
    M_red_sub->set_size( n_red, n_red );
    M_red_sub->set_ofs ( ofs,   ofs   );
    
    K_red_sub->scale( real(0) );
    M_red_sub->scale( real(0) );
    
    //////////////////////////////////////////////////////////////////////////////
    //
    // Determine the submatrices K_red_ii and M_red_ij which have to be computed
    //
    //////////////////////////////////////////////////////////////////////////////
    struct job_t
    {
        //-----------------------------------------------------------------------
        // A job consists of an indexpair (index_i,index_j) and a bool value 
        // 'do_compute_K_red'. If 'do_compute_K_red' is true the matrix K_red_ij 
        // has to be computed if 'do_compute_K_red' is false the matrix M_red_ij. 
        //-----------------------------------------------------------------------
        idx_t index_i;
        idx_t index_j;
        bool  do_compute_K_red;
        
        //constructor
        job_t ( const idx_t i, 
                const idx_t j,
                const bool  comp_K_red )
        {
            index_i          = i;
            index_j          = j;
            do_compute_K_red = comp_K_red;
        }
    };
    
    std::list< job_t * >  job_list;
    
    const size_t first       = subprob_data->first;
    const size_t last_plus_1 = subprob_data->last + 1;
     
    for ( size_t i = first; i < last_plus_1; i++ )
    {        
        if ( !is_zero_size(_S[i]) )
        {
            //------------------------------------------------------
            // K_red_ii has to be computed 
            //------------------------------------------------------            
            job_t * job_K_ii = new job_t( idx_t(i), idx_t(i), true );
        
            job_list.push_back( job_K_ii ); 
            
        
            for ( size_t j = first; j <= i; j++ )
            {       
                //------------------------------------------------------
                // Check if M_red_ij has to be computed or if it is zero
                //------------------------------------------------------
                if ( (!_sep_tree.matrix_block_is_zero(j,i)) && (!is_zero_size(_S[j])) )
                {
                    job_t * job_M_ij = new job_t( idx_t(i), idx_t(j), false );
        
                    job_list.push_back( job_M_ij ); 
                }// if
            }// for
        }// if
    }// for
    
    //////////////////////////////////////////////////////////////////////////////
    //
    // Arrange job list in an array 
    //
    //////////////////////////////////////////////////////////////////////////////
    const size_t n_jobs = job_list.size();
    
    vector< job_t * > job_array( n_jobs, nullptr );

    size_t count = 0;
            
    while ( !job_list.empty() )
    {
        job_array[count] = job_list.front();

        job_list.pop_front();
        
        count++;
    }// while 
    
    if ( count != n_jobs  )
        HERROR( ERR_CONSISTENCY, "(THAMLS) comp_K_and_M_red_parallel", "" ); 
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the submatrices M_red_ij of M_red_sub = S_sub^{T} * M_tilde_sub * S_sub where 
    // M_red_sub is symmetric and M_red_ij = S_i^{T} * M_tilde_ij * S_j. Note that only the
    // lower half triangular of M_red_sub is computed because the matrix is symmetric and 
    // only the lower part is needed for the subsequent (Lapack-eigensolver) computations.
    //
    // Compute the submatrices K_red_ij of K_red_sub = S_sub^{T} * K_tilde_sub * S_sub
    // where K_red_sub is symmetric and K_red_ij = S_i^{T} * K_tilde_ij * S_j
    //
    ///////////////////////////////////////////////////////////////////////////////////////////
 
    auto comp_and_embed_K_and_M_red_ij = 
        [ this, K_tilde, K_red_sub, M_tilde, M_red_sub, & job_array ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  k = r.begin(); k != r.end(); ++k )
            {
                //-----------------------------------------------------------------------------
                // Get the block indices of M_red_ij
                //-----------------------------------------------------------------------------
                const idx_t i = job_array[k]->index_i;
                const idx_t j = job_array[k]->index_j;
                
                const bool do_compute_K_red = job_array[k]->do_compute_K_red;
                
                delete job_array[k];
                
                job_array[k] = nullptr;
                
                //-----------------------------------------------------------------------------
                // Consistency check
                //-----------------------------------------------------------------------------
                if ( _sep_tree.matrix_block_is_zero(j,i) || j > i || i < 0 )
                    HERROR( ERR_CONSISTENCY, "(THAMLS) comp_K_and_M_red_parallel", "" );
                
                if ( do_compute_K_red && i!=j )
                    HERROR( ERR_CONSISTENCY, "(THAMLS) comp_K_and_M_red_parallel", "" );
                
                
                if ( do_compute_K_red )
                {
                    //-----------------------------------------------------------------------------
                    // Get a reference to the right submatrix part of K_red_sub corresponding to
                    // K_red_ij, i.e., updating K_red_ij involves an update of K_red_sub as well
                    //-----------------------------------------------------------------------------

                    // Determine the corresponding row and column indexsets
                    const idx_t reduced_ofs_i = _reduced_subproblem_ofs[i];
                            
                    const TIndexSet row_is_ii( reduced_ofs_i, reduced_ofs_i + _S[i]->cols() -1 );
                        
                    BLAS::Matrix< real >  K_red_sub_BLAS_restricted( K_red_sub->blas_rmat(),
                                                                     row_is_ii - K_red_sub->row_ofs(),
                                                                     row_is_ii - K_red_sub->row_ofs() );
            
                    unique_ptr< TDenseMatrix > K_red_ii( new TDenseMatrix( row_is_ii, row_is_ii, K_red_sub_BLAS_restricted ) );
                        
                    //------------------------------------------------------------------------------------------------
                    // Adjust offset of K_red_ii so that the computation of  K_red_ii = S_i^{T} * K_tilde_ii * S_i
                    // can be applied successfully. Note that the offsets of K_red_ii and the corresponding submatrix 
                    // in K_red_sub slightly differ.
                    //------------------------------------------------------------------------------------------------
                    const idx_t ofs_i = _subproblem_ofs[i];
                    
                    K_red_ii->set_ofs( ofs_i, ofs_i );
                
                    //-----------------------------------------------------------------------------------
                    // Compute K_red_ii     (Note: The submatrix part of K_red_sub is updated as well 
                    //                             because K_red_ii is reference to this submatrix part)
                    //-----------------------------------------------------------------------------------
                    const bool S_i_exact_solution = _sep_tree.has_exact_eigensolution( i );
                            
                    if ( S_i_exact_solution )
                        set_diagonal_dense( _D[i], K_red_ii.get() );
                    else 
                        comp_M_red_ij( K_tilde, i, i, K_red_ii.get() );
                
                }// if
                else
                {
                    //--------------------------------------------------------------------------------------------------
                    // Get a reference to the right submatrix part of M_red_sub corresponding to
                    // M_red_ij, i.e., updating M_red_ij involves an update of M_red_sub as well
                    //--------------------------------------------------------------------------------------------------
                        
                    // Determine the corresponding row and column indexsets
                    const idx_t reduced_ofs_i = _reduced_subproblem_ofs[i];
                    const idx_t reduced_ofs_j = _reduced_subproblem_ofs[j];
                            
                    const TIndexSet row_is_ij( reduced_ofs_i, reduced_ofs_i + _S[i]->cols() -1 );
                    const TIndexSet col_is_ij( reduced_ofs_j, reduced_ofs_j + _S[j]->cols() -1 );
                        
                    BLAS::Matrix< real >  M_red_sub_BLAS_restricted( M_red_sub->blas_rmat(),
                                                                    row_is_ij - M_red_sub->row_ofs(),
                                                                    col_is_ij - M_red_sub->col_ofs() );
            
                    unique_ptr< TDenseMatrix > M_red_ij( new TDenseMatrix( row_is_ij, col_is_ij, M_red_sub_BLAS_restricted ) );
            
                    //--------------------------------------------------------------------------------------------------
                    // Adjust offset of M_red_ij so that the computation of  M_red_ij = S_i^{T} * M_tilde_ij * S_j
                    // can be applied successfully. Note that the offsets of M_red_ij and the corresponding submatrix 
                    // in M_red_sub slightly differ.
                    //--------------------------------------------------------------------------------------------------
                    const idx_t ofs_i = _subproblem_ofs[i];
                    const idx_t ofs_j = _subproblem_ofs[j];
                    
                    M_red_ij->set_ofs( ofs_i, ofs_j );
                    
                    //--------------------------------------------------------------------------------------------------
                    // Compute M_red_ij     (Note: The submatrix part of M_red_sub is updated as well 
                    //                             because M_red_ij is reference to this submatrix part)
                    //--------------------------------------------------------------------------------------------------
                    const bool S_i_exact_solution = _sep_tree.has_exact_eigensolution( i );
                    
                    if ( i == j && S_i_exact_solution )
                        set_Id_dense( M_red_ij.get() );
                    else 
                        comp_M_red_ij( M_tilde, i, j, M_red_ij.get() );
                    
                }// if
            }// for
        };
        
    if ( _para_parallel.K_and_M_red_in_parallel )
    {
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(n_jobs) ), comp_and_embed_K_and_M_red_ij );
    }// if
    else
        comp_and_embed_K_and_M_red_ij( tbb::blocked_range< uint >( uint(0), uint(n_jobs) ) );
            
    //=======================================================================================
    // Set the computed reduced matrix symmetric (this is for example 
    // a requirement to use the function 'TEigenLapack::comp_decomp')
    //=======================================================================================
    K_red_sub->set_symmetric();
    M_red_sub->set_symmetric();
    
    return n_jobs;
}



void
THAMLS::comp_reduced_matrices_SUB ( const TMatrix *   K_tilde,
                                    const TMatrix *   M_tilde,
                                    TDenseMatrix *    K_red_sub,
                                    TDenseMatrix *    M_red_sub,
                                    subprob_data_t *  subprob_data )  
{
    if ( K_tilde == nullptr || M_tilde == nullptr || K_red_sub == nullptr || M_red_sub == nullptr )
        HERROR( ERR_ARG, "(THAMLS) comp_reduced_matrices_SUB", "argument is nullptr" );
        
        
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the reduced matrices 'K_red_sub' and 'M_red_sub' of 'K_tilde_sub' and 
    // 'M_tilde_sub' according the subprolblems contained in the data 'subprob_data' 
    // and according to the partial eigensolutions.
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    TIC;
    comp_K_and_M_red_parallel( K_tilde, M_tilde, K_red_sub, M_red_sub, subprob_data );
    TOC;
    ///////////////////////////////////////////////////////////////////////////
    //
    // NOTE: Set IDs of reduced matrices. If this is not done it can happen 
    // that an error message is created when matrix algebra routines are 
    // applied to the reduced matrices, e.g., LDL-decomposition of M_red_sub
    //
    ///////////////////////////////////////////////////////////////////////////
    // K_red_sub->set_id( ??? );
    // M_red_sub->set_id( ??? );
    
    LOG( to_string("(THAMLS) comp_reduced_matrices_SUB : done in %.2fs", toc.seconds() ));
}




 
size_t
THAMLS::eigen_decomp_reduced_SUB( const TMatrix *   K_red_sub,
                                  const TMatrix *   M_red_sub,
                                  TDenseMatrix  *   D,
                                  TDenseMatrix  *   Z,
                                  subprob_data_t *  subprob_data ) const 
{
    if ( M_red_sub == nullptr || K_red_sub == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(THAMLS) eigen_decomp_reduced_SUB", "argument is nullptr" );
    
    if ( get_mode_selection() != MODE_SEL_AUTO_H )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_reduced_SUB", "not yet supported" ); 
        
    if ( get_reduced_dofs_SUB ( subprob_data ) != K_red_sub->rows() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) eigen_decomp_reduced_SUB", "" );
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Decide if multiple threads should be used for the MKL routines. If the sequential MKL library 
    // (and not the parallel one) is linked with the program then this statement has basically no effect 
    // on the subsequent computation.
    //
    // This feature allows to use multiple threads for the solution of "large" dense eigenvalue problems
    // which are solved by the corresponding LAPACK solver.
    // 
    // NOTE: K_red and M_red are the reduced matrices of the HAMLS method which are symmetric and where 
    //       only the lower block triangular part of M_red is computed. Correspondingly, M_red is not 
    //       "physically" symmetric and when K_red*Z-M_red*Z*D is computed the result should in genernal 
    //       be not zero since the upper triangular part of the dense matrix M_red is not physically present. 
    //       Correspondingly, do not use the debug option for the LAPACK solver since otherwise an exception 
    //       could be thrown.
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool use_multiple_MKL_threads = false;
    
    // NOTE: When several condensations are applied in parallel the MKL thread number has to keep equal to 1 
    if ( ! _para_parallel.condense_subproblems_in_parallel )
        use_multiple_MKL_threads = adjust_MKL_threads_for_eigen_decomp( K_red_sub->rows() );
            
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Set the parameters of the eigensolver for the reduced eigenvalue problem of an subproblem
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////
    TIC;
    
    // These are the DOF which are associated to the subproblems
    // in the 'subprob_data' and they typically much bigger than 
    // the size of the reduced problem
    const int n = get_dofs_SUB ( subprob_data );
        
    //
    // Determine the number of wanted eigenpairs
    //
    size_t k = get_number_of_wanted_eigenvectors( CONDENSE_SUBPROBLEM, n );
    
    if ( k > K_red_sub->rows() )
        k = K_red_sub->rows();
    
    TEigenLapack eigensolver( idx_t(1), idx_t(k) ); 
    
    
    // Do no testing since M is not physically symmetric
    eigensolver.set_test_pos_def  ( false );    
    eigensolver.set_test_residuals( false );
    
        
    const size_t  n_ev = eigensolver.comp_decomp( K_red_sub, M_red_sub, D, Z );
    
    TOC;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Adjust number of MKL threads back to 1 if it is necessary
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( use_multiple_MKL_threads )
        set_MKL_threads_to_one();

    LOG( to_string("(THAMLS) eigen_decomp_reduced_SUB : done in %.2fs", toc.seconds() ));
        
    return n_ev;
}





void
THAMLS::compute_rayleigh_quotients_SUB ( const TMatrix *       K_tilde, 
                                         const TMatrix *       M_tilde, 
                                         TDenseMatrix *        D_approx, 
                                         const TDenseMatrix *  Z_approx, 
                                         subprob_data_t *      subprob_data ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr || D_approx == nullptr || Z_approx == nullptr )
        HERROR( ERR_ARG, "(THAMLS) compute_rayleigh_quotients_SUB", "argument is nullptr" );

        TIC;

        
        //////////////////////////////////////////////////////////////
        //
        // Get the submatrices of K_tilde and M_tilde associated 
        // to the subproblems contained in 'subprob_data'
        //
        //////////////////////////////////////////////////////////////
        
        TCluster * cluster = subprob_data->cluster;
        
        const TIndexSet is( cluster->first(), cluster->last() );
        
        const TMatrix * K_tilde_sub = search_submatrix( K_tilde, is, is );
        const TMatrix * M_tilde_sub = search_submatrix( M_tilde, is, is );
        
        /////////////////////////////////////////////////////////////////////////////////////
        //
        // Determine the Rayleigh Quotients of the column vectors contained in 'Z_approx' 
        // according to the general eigenvalue problem (K_tilde_sub,M_tilde_sub)
        //
        /////////////////////////////////////////////////////////////////////////////////////
                
        const size_t n   = Z_approx->rows();
        const size_t nev = Z_approx->cols();        
        
        auto compute_rayleigh = 
            [ K_tilde_sub, M_tilde_sub, Z_approx, D_approx, & is ] ( const tbb::blocked_range< uint > & r )
            {
                for ( auto  j = r.begin(); j != r.end(); ++j )
                {
                    unique_ptr< TScalarVector >  K_v( new TScalarVector( is ) );
                    unique_ptr< TScalarVector >  M_v( new TScalarVector( is ) );
                    
                    //================================================================
                    // Get eigenvector (v is a reference to the column j of Z_approx)
                    //================================================================
                    TScalarVector v( Z_approx->column( j ) );
                    
                    //================================================================
                    // Compute K*v and M*v
                    //================================================================
                    K_tilde_sub->mul_vec( real(1), & v, real(0), K_v.get() );
                    M_tilde_sub->mul_vec( real(1), & v, real(0), M_v.get() );
                    
                    //================================================================
                    // Compute rayleigh = (v^T)*K*v / (v^T)*M*v
                    //================================================================
                    const real rayleigh = re( dot( & v, K_v.get() ) ) / re( dot( & v, M_v.get() ) );
                    
                    //======================================================================
                    // Replace the eigenvalue in D_approx by the computed rayleigh quotient
                    //======================================================================
                    D_approx->set_entry(j,j, rayleigh );
                }// for
            };
        
        
        if ( _para_parallel.miscellaneous_parallel ) 
            tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(nev) ), compute_rayleigh );
        else 
            compute_rayleigh ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
        

        TOC;
        LOG( to_string("(THAMLS) compute_rayleigh_quotients_SUB : done in %.2fs", toc.seconds() ));
}
    
    
void
THAMLS::apply_HAMLS_subproblem( const TMatrix *   K_tilde, 
                                const TMatrix *   M_tilde, 
                                TDenseMatrix *    D_approx, 
                                TDenseMatrix *    Z_approx, 
                                subprob_data_t *  subprob_data ) 
{
    if ( K_tilde == nullptr || M_tilde == nullptr || D_approx == nullptr || Z_approx == nullptr )
        HERROR( ERR_ARG, "(THAMLS) apply_HAMLS_subproblem", "argument is nullptr" );

        TIC;
        
        ////////////////////////////////////////////////////////////////////////////////////
        //
        // Compute the approximative eigensolution of the EVP (K_tilde_sub,M_tilde_sub)
        // associated to the subproblems in 'subprob_data'. This is done by
        // applying H-AMLS to the block diagonalised EVP (K_tilde,M_tilde) 
        // which is restricted to the subproblems contained in 'subprob_data'
        //
        ////////////////////////////////////////////////////////////////////////////////////
        
        //-------------------------------------------------------------------------------------
        // Compute the matrices of the reduced eigenvalue problem associated to the subproblems 
        //-------------------------------------------------------------------------------------
        unique_ptr< TDenseMatrix >  K_red_sub( new TDenseMatrix() );
        unique_ptr< TDenseMatrix >  M_red_sub( new TDenseMatrix() );  
        
        comp_reduced_matrices_SUB ( K_tilde, M_tilde, K_red_sub.get(), M_red_sub.get(), subprob_data );
        
        //-----------------------------------------------------------------------
        // Compute the eigensolution of the reduced problem (K_red_sub,M_red_sub)
        //-----------------------------------------------------------------------
        unique_ptr< TDenseMatrix >  Z_red_sub( new TDenseMatrix );      
    
        eigen_decomp_reduced_SUB ( K_red_sub.get(), M_red_sub.get(), D_approx, Z_red_sub.get(), subprob_data );
                
        //-----------------------------------------------------------------
        // Transform the eigenvectors of the reduced eigenvalue problem 
        // to eigenvector approximations of the original eigenvalue problem
        //-----------------------------------------------------------------
        
        // subprob_data->timer_T8.cont();
        
        backtransform_eigenvectors_with_S_i ( Z_red_sub.get(), Z_approx, subprob_data );

        // subprob_data->timer_T8.pause();

        //-------------------------------------------------------------------------------------------
        // Improve the eigenpair approximations by computing the corresponding Rayleigh Quotients
        // (Note: This step is actual not necessary since D_approx should not used further because in
        //        the matrix K_red is computed when the eigenvectors of the subproblem are computed 
        //        only approximately.
        //-------------------------------------------------------------------------------------------
        if ( false )
            compute_rayleigh_quotients_SUB ( K_tilde, M_tilde, D_approx, Z_approx, subprob_data );
        
        
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // set the offsets of the matrices D_approx and Z_approx 
        // (the indexsets are defined in this style that the matrix operation K*Z-M*Z*D is well defined)
        //
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        TCluster * cluster = subprob_data->cluster;
        
        const TIndexSet is( cluster->first(), cluster->last() );
        
        const TMatrix * K_tilde_sub = search_submatrix( K_tilde, is, is );
                
        D_approx->set_ofs( K_tilde_sub->col_ofs(), K_tilde_sub->col_ofs() );
        Z_approx->set_ofs( K_tilde_sub->col_ofs(), K_tilde_sub->col_ofs() );
        
        TOC;
        LOG( to_string("(THAMLS) apply_HAMLS_subproblem : done in %.2fs", toc.seconds() ));
}



                
void
THAMLS::truncate_amls_clustertree ( const TCluster * cluster_2_find,
                                    TCluster *       new_root_amls_ct ) const
{
    if ( cluster_2_find == nullptr || new_root_amls_ct == nullptr )
        HERROR( ERR_ARG, "(THAMLS) truncate_amls_clustertree", "argument is nullptr" );
        
    list< TCluster * >  cluster_list;
    TCluster *          cluster;
    
    const TIndexSet     is_2_find( cluster_2_find->first(), cluster_2_find->last() );
    
    /////////////////////////////////////////////////////////////////////////////
    //
    // Traverse the cluster tree and delete the sons of the searched cluster
    //
    /////////////////////////////////////////////////////////////////////////////
    
    cluster_list.push_back( new_root_amls_ct ); 
    
    bool cluster_found = false;
    
    while ( !cluster_list.empty() )
    {
        cluster = cluster_list.front();

        cluster_list.pop_front();
        
        const size_t nsons = cluster->nsons();
            
        if ( ( cluster_2_find->first() == cluster->first()) && ( cluster_2_find->last() == cluster->last()) )
        {
            cluster_found = true;
            
            //
            // Delete the sons of the found cluster
            //
            for ( uint i = 0; i < nsons; i++ ) 
                cluster->set_son( i, nullptr );
               
            cluster->clear_sons();
                
            break;
        }// if
        else
        {
            for ( size_t i = 0; i < nsons; i++ )
            {    
                TCluster * cluster_son = cluster->son(i);
                
                if ( cluster_son != nullptr && cluster_son->is_sub( is_2_find ) )
                {
                    cluster_list.push_back( cluster->son(i) );
                        
                    break;
                }// if
            }// for
        }// else
    }// while
    
    if ( !cluster_found )
        HERROR( ERR_CONSISTENCY, "(THAMLS) truncate_amls_clustertree", "" );
}



size_t 
THAMLS::try_2_condense_suproblems ( const TMatrix *  K_tilde,
                                    const TMatrix *  M_tilde )
{
    if ( K_tilde == nullptr || M_tilde == nullptr )
        HERROR( ERR_ARG, "(THAMLS) try_2_condense_suproblems", "argument is nullptr" );
    
        if ( _para_mode_sel.factor_condense < _para_mode_sel.factor_subdomain )
        HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" );
    
    if ( _para_mode_sel.ratio_2_condense < 1 )
        HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" );
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step a) of task (T5) 
    //
    // Find subproblems which should be condensed and save them in an array
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    list< subprob_data_t * >  subproblem_list;
    
    bool subproblem_found = find_subproblems_which_shall_be_condensed( subproblem_list );
    
    if ( !subproblem_found )
        return false;
        
    const size_t number_of_condensations = subproblem_list.size();
    
    vector< subprob_data_t * > subproblems_2_condense( number_of_condensations, nullptr );
    
    size_t count = 0;
            
    while ( !subproblem_list.empty() )
    {
        subproblems_2_condense[count] = subproblem_list.front();

        subproblem_list.pop_front();
        
        count++;
    }// while 
    
    if ( count != number_of_condensations )
        HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" ); 
        
        
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step b) of task (T5) 
    //
    // Compute the approximative eigensolution of the EVP (K_tilde,M_tilde) associated to the subproblems in 'subprob_data' 
    // which should be condensed together. This is done by applying H-AMLS to the block diagonalised EVP (K_tilde,M_tilde) 
    // restricted to the subproblems contained in 'subprob_data'.
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
    auto comp_approx_eigensol = 
        [ this, K_tilde, M_tilde, & subproblems_2_condense  ] ( const tbb::blocked_range< uint > & r )
        {
            for ( auto  k = r.begin(); k != r.end(); ++k )
            {
                subprob_data_t *  subprob_data = subproblems_2_condense[k];
                
                //--------------------------------------------------------------
                // Compute the approximative eigensolution of the EVP associated  
                // to the subproblems in 'subprob_data' using H-AMLS 
                //--------------------------------------------------------------
                TDenseMatrix * D_approx = new TDenseMatrix;
                TDenseMatrix * Z_approx = new TDenseMatrix;
                
                apply_HAMLS_subproblem( K_tilde, M_tilde, D_approx, Z_approx, subprob_data );
                
                //---------------------------------------------------------------
                // Save the obtained approximated eigensolution in 'subprob_data'
                //---------------------------------------------------------------
                if ( subprob_data->D_approx != nullptr || subprob_data->Z_approx != nullptr )
                    HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" ); 
                
                subprob_data->D_approx = D_approx;
                subprob_data->Z_approx = Z_approx;     
            }// for
        };
    
    if ( _para_parallel.condense_subproblems_in_parallel )
        tbb::parallel_for( tbb::blocked_range< uint >( uint(0), uint(number_of_condensations) ), comp_approx_eigensol );
    else            
        comp_approx_eigensol( tbb::blocked_range< uint >( uint(0), uint(number_of_condensations) ) );
    
    //==================================================================================================
    // Update the global counter
    //==================================================================================================
    _para_mode_sel.n_condensed_problems = _para_mode_sel.n_condensed_problems + number_of_condensations;
            
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step c) of task (T5) 
    //
    // Make output if wanted (Note: The previous auxilliary data of AMLS is needed)
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #if DO_TEST >= 2
    for ( size_t k = 0; k < number_of_condensations; k++ )
    {
        subprob_data_t *  subprob_data = subproblems_2_condense[k];
        
        output_SUB( K_tilde, M_tilde, subprob_data,k );
    }// for 
    #endif
    
                
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step d) of task (T5) 
    //
    // Create the new AMLS cluster tree according to the condensed subproblems and the corresponding separator tree 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    TCluster * new_root_amls_ct = _root_amls_ct->copy();
    
    for ( size_t k = 0; k < number_of_condensations; k++ )
    {
        subprob_data_t *  subprob_data = subproblems_2_condense[k];
        
        if ( subprob_data->cluster == nullptr )
            HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" ); 
            
        truncate_amls_clustertree( subprob_data->cluster, new_root_amls_ct );
        
    }// for
        
    TSeparatorTree new_sep_tree = TSeparatorTree( new_root_amls_ct );
        
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step e) of task (T5) 
    //
    // Save the old (unchanged) and the new computed approximated eigensolutions (D_i,S_i) into an array
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
    //==========================================================
    // Initialise the new auxiliary data for saving the 
    // eigensolutions (D_i,S_i) according to the new data
    //==========================================================
    
    const size_t n_new_subproblems = new_sep_tree.n_subproblems();
    
    vector < TDenseMatrix * >  D_new( n_new_subproblems, nullptr );
    vector < TDenseMatrix * >  S_new( n_new_subproblems, nullptr );
        
    //==========================================================
    // Save the new computed approximate eigensolutions of the 
    // condensed subproblems into the new auxilliary data
    //==========================================================
    for ( size_t k = 0; k < number_of_condensations; k++ )
    {
        subprob_data_t *  subprob_data = subproblems_2_condense[k];
                    
        if ( subprob_data->D_approx == nullptr || subprob_data->Z_approx == nullptr || subprob_data->cluster == nullptr )
            HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" ); 
            
        //-----------------------------------------------------------
        // Find right index of the approximate eigensolution and mark
        // in the separator tree that the eigensolution is not exact
        //-----------------------------------------------------------
        const TIndexSet is( (subprob_data->cluster)->first(), (subprob_data->cluster)->last() );
        
        const size_t index = new_sep_tree.get_index( is );
        
        D_new[index] = subprob_data->D_approx;
        S_new[index] = subprob_data->Z_approx;
        
        //------------------------------------------------------
        // Important: Make a note the eigendecomposition is only 
        // approximately and not exact.
        //------------------------------------------------------
        new_sep_tree.set_exact_eigensolution( index, false );
    }// for
    

    //==========================================================
    // Save the old eigensolutions of the subproblems 
    // (which have not been condensed) into the new data
    //==========================================================
    vector< bool > re_used_solution( _sep_tree.n_subproblems(), false );
    
    for ( size_t i = 0; i < n_new_subproblems; i++ )
    {
        //--------------------------------------------------------------------------
        // Check if the new subproblem has already an eigensolution. If this is the 
        // case this problem got an approximate solution in the previous step. If 
        // not then there has to be the corresponding eigensolution in the old data. 
        //--------------------------------------------------------------------------
        if ( D_new[i] == nullptr )
        {
            //-----------------------------------------------------------------------------------
            // Find the corresponding eigensolution in the old data and save it into the new data
            //-----------------------------------------------------------------------------------
            const TIndexSet is = new_sep_tree.get_dof_is( i );
            
            const size_t old_index = _sep_tree.get_index( is );
            
            D_new[i] = _D[old_index];
            S_new[i] = _S[old_index];
            
            // Check if the eigensolution is exact or approximated and save this in the new data
            const bool exact_eigensolution = _sep_tree.has_exact_eigensolution( old_index );
            
            new_sep_tree.set_exact_eigensolution( i, exact_eigensolution );
                
            // Dereference eigensolutions from the old data
            _D[old_index] = nullptr;
            _S[old_index] = nullptr;
            
            // Set local auxialiary value
            re_used_solution[old_index] = true;
        }// if
    }// for 
    
    //==========================================================
    // Delete old eigensolutions of old subproblems in the old 
    // auxiliary data which have been substituted by approximate 
    // eigensolutions of the corresponding condensed problem 
    // because these eigensolutions are not needed anymore
    //==========================================================
    for ( size_t i = 0; i < _sep_tree.n_subproblems(); i++ )
    {
        if ( !re_used_solution[i] )
        {
            if ( _D[i] == nullptr || _S[i] == nullptr )
                HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" );
        
            delete _D[i];
            delete _S[i];
            
            _D[i] = nullptr;
            _S[i] = nullptr;
        }// if
    }// for 
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // step f) of task (T5) 
    // 
    // Set the new auxiliary data of THAMLS
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //==========================================================
    // Replace the old separator tree and the 
    // old amls cluster tree by the new ones 
    //==========================================================
    
    delete _root_amls_ct;
    
    _root_amls_ct = new_root_amls_ct;
    _sep_tree     = new_sep_tree;
    
    //==========================================================
    // Initialise the new auxialiary data 
    //==========================================================
    
    for ( size_t i = 0; i < new_sep_tree.n_subproblems(); i++ )
    {
        if ( _D[i] != nullptr || _S[i] != nullptr )
            HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" );
             
        if ( D_new[i] == nullptr || S_new[i] == nullptr )
            HERROR( ERR_CONSISTENCY, "(THAMLS) try_2_condense_suproblems", "" ); 
        
        _D[i] = D_new[i];
        _S[i] = S_new[i];
    }// for
    
    //==========================================================
    // Set the new offset data for the subsequent computations
    //==========================================================
    set_subproblem_ofs ();
    set_reduced_subproblem_ofs ();  
    
    //==========================================================
    // Update global times with the measured times 
    // in the subproblem data and clean up data
    //==========================================================
    
    for ( size_t k = 0; k < number_of_condensations; k++ )
    {
        subprob_data_t *  subprob_data = subproblems_2_condense[k];
                
        //---------------------------------
        // Clean up data 
        //---------------------------------
        delete subprob_data;
        
        subprob_data = nullptr;
    }// for
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Return number of made condensations
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    return number_of_condensations;
}

void
THAMLS::output_SUB ( const TMatrix *         K_tilde, 
                     const TMatrix *         M_tilde, 
                     const subprob_data_t *  subprob_data,
                     const size_t            problem_number ) const 
{
    /////////////////////////////////////////////////////////////////////////////////
    //
    // Print Output Information
    //
    /////////////////////////////////////////////////////////////////////////////////

    TCluster * cluster = subprob_data->cluster;
    
    const TIndexSet is( cluster->first(), cluster->last() );
    
    const TMatrix * K_tilde_sub   = search_submatrix( K_tilde, is, is );
    const TMatrix * M_tilde_sub   = search_submatrix( M_tilde, is, is );
    
    const TDenseMatrix * D_approx = subprob_data->D_approx;
    const TDenseMatrix * Z_approx = subprob_data->Z_approx;

    LOG( "    ________________________________________________________________________________________________________" );
    LOG( "    |                                                                                                       |" );
    LOG( "    |                    Subproblem number "+to_string("%d",problem_number)+" successfully condensed" );
    LOG( "    |_______________________________________________________________________________________________________|" );
    LOG( "" );
                    
    print_structural_info_SUB ( subprob_data );
    print_summary_SUB         ( D_approx, subprob_data );
    
    if ( false )
    {
        TEigenAnalysis eigen_analysis;
        eigen_analysis.set_verbosity( 3 );
        eigen_analysis.analyse_vector_residual( K_tilde_sub, M_tilde_sub, D_approx, Z_approx );
    }// if
}


void 
THAMLS::apply_condensing ( const TMatrix *  K_tilde,
                           const TMatrix *  M_tilde )
{
    // Print structural information of the original AMLS 
    // cluster before subproblems get condensed
    #if DO_TEST >= 2
    print_structural_info();
    #endif
    
    LOG( "" );
    LOG( "_____________________________________________________________________________________________________________________________" );
    LOG( "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "||||||||||||||||||||||||||||||||||||||||||||||||                       ||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "||||||||||||||||||||||||||||||||||||||||||||||||    Start Condensing   ||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "" );
    
    // Initialise stopping criteria
    size_t number_of_condensations = 1;
    
    while ( number_of_condensations > 0 )
    {
        //-----------------------------------------------------------------------------------------------
        // Try to condense suproblems together and update the separator and the amls tree correspondingly
        //-----------------------------------------------------------------------------------------------
        number_of_condensations = try_2_condense_suproblems ( K_tilde, M_tilde );
        
        LOG( "" );
        LOG( "                |\\ " );
        LOG( "                | \\ " );
        LOG( " _______________|  \\ " );
        LOG( " |                  \\ " );
        LOG( " |                   \\   number of condensations = "+to_string("%d",number_of_condensations) );
        LOG( " |                   /" );
        LOG( " |_______________   / " );
        LOG( "                |  / " );
        LOG( "                | / " );
        LOG( "                |/ " );
        LOG( "" );
            
        #if DO_TEST >= 1
        print_structural_info();
        #endif

    }// while
    
    LOG( "" );
    LOG( "|||||||||||||||||||||||||||||||||||||||||||||||||    End Condensing   |||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "|||||||||||||||||||||||||||||||||||||||||||||||||                     |||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" );
    LOG( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" );
    LOG( "" );
}



void 
THAMLS::print_structural_info_SUB ( const subprob_data_t *  subprob_data,
                                    const bool              is_real_subproblem ) const
{ 
    /////////////////////////////////////////////////////////////////////////////
    //
    // Get the size of the biggest/smallest subdomain/interface suproblems and
    // the size of the biggest/smallest reduced subdomain/interface subproblems
    //
    /////////////////////////////////////////////////////////////////////////////
   
    //----------------------------------------
    //
    // determine initial max values
    //
    //----------------------------------------
    size_t n_subdomain_max     = 0;
    size_t n_interface_max     = 0;
    size_t n_subdomain_red_max = 0;
    size_t n_interface_red_max = 0;
   
    //----------------------------------------
    //
    // determine initial min values
    //
    //----------------------------------------
    size_t n_subdomain_min     = 0;
    size_t n_interface_min     = 0;
    size_t n_subdomain_red_min = 0;
    size_t n_interface_red_min = 0;
    
    for ( size_t i = subprob_data->first; i <= subprob_data->last; i++ )
    {
        if ( !_sep_tree.is_domain( i ) )
        {
            n_interface_min     = _sep_tree.get_dof_is(i).size();
            n_interface_red_min = _S[i]->cols();
            
            break;
        }// 
    }// for
    
    for ( size_t i = subprob_data->first; i <= subprob_data->last; i++ )
    {
        if ( _sep_tree.is_domain( i ) )
        {
            n_subdomain_min     = _sep_tree.get_dof_is(i).size();
            n_subdomain_red_min = _S[i]->cols();
            
            break;
        }// if
    }// for
    
    
    //----------------------------------------
    //
    // determine min/max values
    //
    //----------------------------------------
    size_t n_red      = 0;
    size_t n_temp     = 0;
    size_t n_original = 0;
    
    
    //
    // determine values before reduction
    //
    for ( size_t i = subprob_data->first; i <= subprob_data->last; i++ )
    {
        n_temp = _S[i]->rows();
    
        if ( _sep_tree.is_domain( i ) )
        {
            if ( n_subdomain_max < n_temp )
                n_subdomain_max = n_temp;
                
            if ( n_subdomain_min > n_temp )
                n_subdomain_min = n_temp;
        }// if
        else
        {
            if ( n_interface_max < n_temp )
                n_interface_max = n_temp;  
                
            if ( n_interface_min > n_temp )
                n_interface_min = n_temp;              
        }// else
        
        n_original = n_original + n_temp;
    }// for
    
    //
    // determine values after reduction
    //
    for ( size_t i = subprob_data->first; i <= subprob_data->last; i++ )
    {
        n_temp = _S[i]->cols();
    
        if ( _sep_tree.is_domain( i ) )
        {
            if ( n_subdomain_red_max < n_temp )
                n_subdomain_red_max = n_temp;
                
            if ( n_subdomain_red_min > n_temp )
                n_subdomain_red_min = n_temp;
        }// if
        else
        {
            if ( n_interface_red_max < n_temp )
                n_interface_red_max = n_temp;        
                
            if ( n_interface_red_min > n_temp )
                n_interface_red_min = n_temp;        
        }// else
        
        n_red = n_red + n_temp;
    }// for
    
    //---------------------------------------------------------------------
    //
    // determine number of small and large interface eigenvalue problems
    //
    //---------------------------------------------------------------------
    uint count_subdomains       = 0;
    uint count_small_interfaces = 0;
    uint count_large_interfaces = 0;
    
    for ( size_t i = subprob_data->first; i <= subprob_data->last; i++ )
    {
        if ( _sep_tree.is_domain( i ) )
        {
            count_subdomains++;
        }// if
        else
        {
            if ( is_small_problem( i ) )
                count_small_interfaces++;
            else
                count_large_interfaces++;
        }// else
    }// for
    
    if ( !is_real_subproblem && count_small_interfaces + count_large_interfaces != _sep_tree.n_interface_subproblems() )
        HERROR( ERR_CONSISTENCY, "(THAMLS) print_structural_info_SUB", "" ); 
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //
    // Print information
    //
    //
    /////////////////////////////////////////////////////////////////////////////
    const size_t n_subproblems = subprob_data->last - subprob_data->first + 1;
      
    if ( is_real_subproblem )
    {
        OUT( "" );
        OUT( "(THAMLS) print_structural_info_SUB : " );
        HLINE;
        OUT( to_string("    general:   dimension of (K_tilde_sub,M_tilde_sub) = %d",n_original) );
        OUT( to_string("               dimension of (K_red_sub,M_red_sub)     = %d",n_red) );
        OUT( to_string("               index of first subproblem              = %d",subprob_data->first) );
        OUT( to_string("               index of last  subproblem              = %d",subprob_data->last) );
        OUT( to_string("               depth of separator tree                = %d",_sep_tree.depth()) );
        OUT( to_string("               level of associated cluster            = %d",subprob_data->level) );
    }// if
    else
    {
        OUT( "" );
        OUT( "(THAMLS) print_structural_info : " );
        HLINE;
        OUT( to_string("    general:   dimension of (K,M)           = %d",n_original) );
        OUT( to_string("               dimension of (K_red,M_red)   = %d",n_red) );
        OUT( to_string("               number of condensed problems = %d",_para_mode_sel.n_condensed_problems) );
        OUT( to_string("               depth of separator tree      = %d",_sep_tree.depth()) );
    }// else
    
    HLINE;
    OUT( to_string("    number of: subdomain problems       = %d",count_subdomains) );
    OUT( to_string("               small interface problems = %d",count_small_interfaces) );
    OUT( to_string("               large interface problems = %d",count_large_interfaces) );
    OUT( to_string("               all subproblems          = %d",n_subproblems) );
    HLINE;
    OUT( to_string("    size of largest:  subdomain problem         = %d",n_subdomain_max) );
    OUT( to_string("                      interface problem         = %d",n_interface_max) );
    OUT( to_string("                      reduced subdomain problem = %d",n_subdomain_red_max) );
    OUT( to_string("                      reduced interface problem = %d",n_interface_red_max) );
    HLINE;
    OUT( to_string("    size of smallest: subdomain problem         = %d",n_subdomain_min) );
    OUT( to_string("                      interface problem         = %d",n_interface_min) );
    OUT( to_string("                      reduced subdomain problem = %d",n_subdomain_red_min) );
    OUT( to_string("                      reduced interface problem = %d",n_interface_red_min) );
    HLINE;
}



             
                             
void
THAMLS::print_summary_SUB ( const TMatrix *         D_approx,
                            const subprob_data_t *  subprob_data ) const
{
    const size_t n_red = get_reduced_dofs_SUB( subprob_data );
    const size_t n     = get_dofs_SUB( subprob_data );
    
    const size_t n_subproblems = subprob_data->last - subprob_data->first + 1;
        
    OUT( "" );
    OUT( "(THAMLS) print_summary_SUB :" );
    HLINE;
    OUT( "    H-AMLS computed the reduced EVP (K_red_sub,M_red_sub) of the original EVP (K_tilde_sub,M_tilde_sub)" );
    OUT( "    (D,Z) are eigenpair approximations for (K_tilde_sub,M_tilde_sub) received from (K_red_sub,M_red_sub)" );
    HLINE;
    OUT( to_string("    dimension of (K_tilde_sub,M_tilde_sub) = %d",n) );
    OUT( to_string("    dimension of (K_red_sub,M_red_sub)     = %d",n_red) );  
    OUT( to_string("    %d of %d possible eigenpair approximations have been computed",D_approx->rows(),n_red) );
    HLINE;
    OUT( to_string("    Result: %d subproblems with %d reduced DOF are condensed to one subproblem with %d reduced DOF",n_subproblems,n_red,D_approx->cols()) );
    HLINE;
}



}// namespace
