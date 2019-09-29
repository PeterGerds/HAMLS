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
// File        : TEigenArnoldi.cc
// Description : eigensolver for H-matrices based on Arnoldi method
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "hamls/TEigenArnoldi.hh"

namespace HAMLS
{

using std::unique_ptr;
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


namespace
{


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// local functions for TEigenArnoldi
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

struct timer_arnoldi_t
{
    double all;
    double consistency;
    double transform;
    double arnoldi;
    double backtransform;

    //constructor
    timer_arnoldi_t ()
    {
        all           = 0;
        consistency   = 0;
        transform     = 0;
        arnoldi       = 0;
        backtransform = 0;
    }
    
    void print_performance() 
    {
        OUT( "" );
        OUT( "(TEigenArnoldi) print_performance :" );
        HLINE;
        OUT( to_string("    task_scheduler_init::default_num_threads() = %d",tbb::task_scheduler_init::default_num_threads()) );
        OUT( to_string("    CFG::nthreads()                            = %d",CFG::nthreads()) );
        HLINE;
        OUT( to_string("    all done in %fs",all) );
        HLINE;
        OUT( to_string("    consistency   = %f %",consistency/all*100) );
        OUT( to_string("    transform     = %f %",transform/all*100) );
        OUT( to_string("    arnoldi       = %f %",arnoldi/all*100) );
        OUT( to_string("    backtransform = %f %",backtransform/all*100) );
        HLINE;
    }
};





void
reortho_q_tilde_in_serial ( const vector < TVector * > & q,
                            const idx_t                  first,
                            const idx_t                  last,
                            TVector *                    q_tilde )
{
    /////////////////////////////////////////////////////////////////
    //
    // Re-orthogonalise q_tilde with vectors contained in data 'q'
    //
    /////////////////////////////////////////////////////////////////
    for ( idx_t i = first; i <= last; i++ )
    {
        const real factor = re( dot( q[i], q_tilde ) );
        
        q_tilde->axpy( -factor, q[i] );
    }// for
}
                 


class reortho_q_tilde_cont_t : public tbb::task
{
  private:
  
  public:
    TVector * _q_tilde;
    TVector * _q_tilde_1;
    TVector * _q_tilde_2;
   
    reortho_q_tilde_cont_t( TVector * q_tilde,
                            TVector * q_tilde_1,
                            TVector * q_tilde_2 )
        : _q_tilde  ( q_tilde   )
        , _q_tilde_1( q_tilde_1 )
        , _q_tilde_2( q_tilde_2 )
    {}

    task * execute()
    {
        //-----------------------------------------------
        // q_tilde = -q_tilde_old + q_tilde_1 + q_tilde_2
        //-----------------------------------------------
        _q_tilde->scale( -real(1) );
        _q_tilde->axpy (  real(1), _q_tilde_1 );
        _q_tilde->axpy (  real(1), _q_tilde_2 );
        
        delete _q_tilde_1;
        delete _q_tilde_2;
        
        _q_tilde_1 = nullptr;
        _q_tilde_2 = nullptr;
        
        return nullptr;
    }
};



class reortho_q_tilde_task_t : public tbb::task
{
  private:
   const vector < TVector * > & _q;
   const idx_t                  _first;
   const idx_t                  _last;
   TVector *                    _q_tilde;
   const size_t                 _size_serial_ortho;
   
  public:

    reortho_q_tilde_task_t( const vector < TVector * > & q,
                            const idx_t                  first,
                            const idx_t                  last,
                            TVector *                    q_tilde,
                            const size_t                 size_serial_ortho )
        : _q( q )
        , _first( first )
        , _last( last )
        , _q_tilde( q_tilde )
        , _size_serial_ortho( size_serial_ortho )
    { if ( _first > _last ) HERROR( ERR_CONSISTENCY, "(TEigenArnoldi) reortho_q_tilde_task_t", "" );}

    task * execute()
    {
        if ( _last - _first <= idx_t(_size_serial_ortho) )
        {
            //////////////////////////////////////////////////////////////////
            //
            // If problem is small enough then compute it serially
            //
            //////////////////////////////////////////////////////////////////
            reortho_q_tilde_in_serial ( _q, _first, _last, _q_tilde );
            
            return nullptr;
        }// if
        else
        {
            //////////////////////////////////////////////////////////////////
            //
            // If problem is large then split it into two subproblems
            //
            //////////////////////////////////////////////////////////////////
            
            
            //-------------------------------------------------------------------------------
            // Reorthogonalise q_tilde with first and second half of arnoldi basis separately
            //-------------------------------------------------------------------------------
            const idx_t mid = idx_t( ( _last + _first)/2 );
            
            unique_ptr< TVector >  q_tilde_1  ( _q_tilde->copy() );
            unique_ptr< TVector >  q_tilde_2  ( _q_tilde->copy() );
            
            
            
            
            #if 0
            tbb::task & child1 = * new( allocate_child() ) reortho_q_tilde_task_t( _q, _first,   mid, q_tilde_1.get(), _size_serial_ortho );
            tbb::task & child2 = * new( allocate_child() ) reortho_q_tilde_task_t( _q,  mid+1, _last, q_tilde_2.get(), _size_serial_ortho );
            
            set_ref_count( 3 );
            spawn( child2 );
            spawn_and_wait_for_all( child1 );
            
            //---------------------------------------------
            // q_tilde <== -q_tilde + q_tilde_1 + q_tilde_2
            //---------------------------------------------
            _q_tilde->scale( -real(1) );
            _q_tilde->axpy (  real(1), q_tilde_1.get() );
            _q_tilde->axpy (  real(1), q_tilde_2.get() );
            
            return nullptr;
            #endif
            
            
            
            #if 1
            //----------------------------------------------------------------------------------------------------------------------------------------------
            //NOTE: The pointers of the unique_ptr data types q_tilde_1 and q_tilde_2 have been released since otherwise they may 
            //      be deleted before child1 and child 2 have finished their work. Deletion is done in the continuation task itself.
            //----------------------------------------------------------------------------------------------------------------------------------------------
            reortho_q_tilde_cont_t  & cont = * new ( allocate_continuation() ) reortho_q_tilde_cont_t( _q_tilde, q_tilde_1.release(), q_tilde_2.release() );
            
            tbb::task  & child1 = * new ( cont.allocate_child() ) reortho_q_tilde_task_t( _q, _first,   mid, cont._q_tilde_1, _size_serial_ortho );
            tbb::task  & child2 = * new ( cont.allocate_child() ) reortho_q_tilde_task_t( _q,  mid+1, _last, cont._q_tilde_2, _size_serial_ortho );

            cont.set_ref_count( 2 );
            
            spawn( child2 );
            
            return & child1;
            #endif
            
            
            
        }// else
    }
};




}// namespace



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// TEigenBase
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////



bool 
TEigenBase::solve_trivial_problems ( const TMatrix *  K,
                                      const TMatrix *  M,
                                      TDenseMatrix  *  D,
                                      TDenseMatrix  *  Z ) const
{
    bool trivial_problem_solved = false;

    if ( K->rows() == 0 )
    {
        //////////////////////////////////////////////////////////////////////
        //
        // Handle special case of matrices of size 0x0
        //
        //////////////////////////////////////////////////////////////////////
        D->set_size( 0, 0 );
        Z->set_size( 0, 0 );
        
        D->set_ofs( K->col_ofs(), K->col_ofs() );
        Z->set_ofs( K->col_ofs(), K->col_ofs() );
        
        trivial_problem_solved = true;
    
    }// if
    else if ( _parameter_base.n_ev_searched == 0 )
    {
        //////////////////////////////////////////////////////////////////////
        //
        // Handle special case that number of searched eigenpairs is zero
        //
        //////////////////////////////////////////////////////////////////////
        D->set_size( 0, 0 );
        Z->set_size( K->rows(), 0 );
        
        D->set_ofs( K->col_ofs(), K->col_ofs() );
        Z->set_ofs( K->col_ofs(), K->col_ofs() );
        
        trivial_problem_solved = true;
        
    }// else if
    else if ( K->rows() == 1 )
    {
        //////////////////////////////////////////////////////////////////////
        //
        // Handle special case of matrices of size 1x1
        //
        ////////////////////////////////////////////////////////////////////// 
        D->set_size( 1, 1 );
        Z->set_size( 1, 1 );
        
        //------------------------------------------------
        //
        // Solve K*x = lambda*M*x with x^{T} * M * x = 1
        //
        //------------------------------------------------
        const real entry_M = M->entry(0,0);
        const real entry_K = K->entry(0,0);
        
        if ( ! ( entry_M > 0 ) )
            HERROR( ERR_CONSISTENCY, "(TEigenBase) solve_trivial_problems", "M is not pos. def." );
            
        const real lambda      = entry_K / entry_M;
        const real eigenvector = real(1) / Math::sqrt( entry_M );
        
        Z->set_entry( 0,0, eigenvector );
        D->set_entry( 0,0, lambda );
        
        trivial_problem_solved = true;
        
    }// else if
    
    return trivial_problem_solved;
}

 
void
TEigenBase::check_input_consistency ( const TMatrix *  K,
                                       const TMatrix *  M,
                                       TDenseMatrix  *  D,
                                       TDenseMatrix  *  Z ) 
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Do some consistency checks
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( M == nullptr || K == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenBase) check_input_consistency", "argument is nullptr" );
        
    if ( K->rows() != K->cols() || M->rows() != M->cols() )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) check_input_consistency", "matrices are not quadratic" );
        
    if ( K->rows() != M->rows() )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) check_input_consistency", "matrices have different size" );
        
    if ( !K->is_symmetric() || !M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) check_input_consistency", "matrices are not symmetric" );
        
        
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Verify that the matrix 'M' is positive definit. 
    //
    // - this property of M is needed, compare with [Templates for solution of Algebraic Eigenvalue Problems]
    // - this property of M is needed in mode 1 and 2 of the ARPACK routine 'dsaupd' (see manueal ARPACK)
    // - this property of M is needed for the used LAPACK eigensolvers '_sygv' and '_sygx'
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_base.test_pos_def )
    {
        if ( M->rows() == 0 )
            HERROR( ERR_CONSISTENCY, "(TEigenBase) check_input_consistency", "M has zero size" ); 
    
        if ( ! EIGEN_MISC::is_pos_def( M ) )
            HERROR( ERR_CONSISTENCY, "(TEigenBase) check_input_consistency", "M is not positive definit" ); 
    }// if
}

  



void
TEigenBase::comp_shifted_matrix ( const TMatrix * K, 
                                   const TMatrix * M, 
                                   const real      shift, 
                                   TMatrix * &     K_minus_shift_M,
                                   const real      eps_rel_exact ) const
{
    if ( K == nullptr || M == nullptr )
       HERROR( ERR_ARG, "(TEigenBase) comp_shifted_matrix", "argument is nullptr" );
       
    if ( K_minus_shift_M != nullptr )
       HERROR( ERR_ARG, "(TEigenBase) comp_shifted_matrix", "argument is NOT nullptr" );
       
    /////////////////////////////////////////////////////////
    //   
    // Compute the matrix   K_minus_shift_M = K-shift*M 
    //
    /////////////////////////////////////////////////////////
    TIC;
    
    if ( shift == real(0) )
        K_minus_shift_M = K->copy().release();
    else
    {
        K_minus_shift_M = K->copy().release();
        
        const TTruncAcc acc( eps_rel_exact );
        
        add( -shift, M, real(1), K_minus_shift_M, acc ); 
    }// else
    
    TOC;
    LOG( to_string("(TEigenBase) comp_shifted_matrix : done in %.2fs", toc.seconds() ));
}




void
TEigenBase::factorise ( const TMatrix *  K_minus_shift_M, 
                         TMatrix * &      L,
                         const real       eps_rel_exact ) 
{
    if ( K_minus_shift_M == nullptr )
       HERROR( ERR_ARG, "(TEigenBase) factorise", "argument is nullptr" );
       
    if ( L != nullptr )
       HERROR( ERR_ARG, "(TEigenBase) factorise", "argument is NOT nullptr" );
       
    if ( !K_minus_shift_M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) factorise", "matrix is not symmetric" ); 
        
    //////////////////////////////////////////////////////////////////////
    //
    // Test if K_minus_shift_M is positive definite if wanted
    //    
    //////////////////////////////////////////////////////////////////////
    if ( _parameter_base.test_pos_def )
    {
        if ( K_minus_shift_M->rows() == 0 )
            HERROR( ERR_CONSISTENCY, "(TEigenBase) factorise", "matrix has zero size" ); 
    
        if ( ! EIGEN_MISC::is_pos_def( K_minus_shift_M ) )
            HERROR( ERR_CONSISTENCY, "(TEigenBase) factorise", "matrix is not positive definit" ); 
    }// if
        
    //////////////////////////////////////////////////////////////////////
    //
    // Compute the Cholesky factorisation  K_minus_shift_M = L*L^{T} 
    //    
    //////////////////////////////////////////////////////////////////////
    TIC;
    
    L = K_minus_shift_M->copy().release();
    
    //--------------------------------------------------------------------
    // Only pointwise is possible since we have the Cholesky factorisation
    //--------------------------------------------------------------------
    fac_options_t  fac_opts;
    
    fac_opts.eval = point_wise;
    
    const TTruncAcc acc( eps_rel_exact );
    
    LL::factorise( L, acc, fac_opts );
    
    TOC;
    LOG( to_string("(TEigenBase) factorise : done in %.2fs", toc.seconds() ));
}


 
void
TEigenBase::comp_precond ( const TMatrix *      K_minus_shift_M,
                            TMatrix * &          K_minus_shift_M_factorised,
                            TLinearOperator * &  K_minus_shift_M_precond,
                            const real           eps_rel_precond_start ) 
{
    if ( K_minus_shift_M == nullptr )
       HERROR( ERR_ARG, "(TEigenBase) comp_precond", "argument is nullptr" );
       
    if ( K_minus_shift_M_factorised != nullptr || K_minus_shift_M_precond != nullptr)
       HERROR( ERR_ARG, "(TEigenBase) factorise", "argument is NOT nullptr" );
       
    if ( !K_minus_shift_M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) comp_precond", "matrices are not symmetric" );  

    
    //////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute a good approximate inverse K_minus_shift_M_precond of the  
    // matrix K_minus_shift_M which is later used as a preconditioner
    //
    //
    //////////////////////////////////////////////////////////////////////////////
    
    fac_options_t  fac_opt;
    
    //----------------------------------------------------------------------------
    //block_wise is more stable than point_wise (cf. online documentation HLIBpro)
    //----------------------------------------------------------------------------
    fac_opt.eval = block_wise; 
            
    const matform_t  matform = K_minus_shift_M->form();
    
    //--------------------------------------------------------------------------------------------------------------
    // Start with initial approximation accuracy for computing a good preconditioner and adapt this accuracy
    // depending on the norm ||K_minus_shift_M * K_minus_shift_M_precond - Id|| until a good preconditioner is found
    //--------------------------------------------------------------------------------------------------------------
    real   eps_rel_precond = eps_rel_precond_start;
    size_t number_of_tries = _parameter_EVP_transformation.precond_max_try;
    size_t count_try       = 0;
    
    TIC;
            
    while ( true )
    {   
    
        //////////////////////////////////////////////////////////////////////////////
        //
        // Compute a factorisation (used for inversion) with the predefined accuracy
        //
        //////////////////////////////////////////////////////////////////////////////
                
        TTruncAcc  acc_inverse ( eps_rel_precond );
        
        unique_ptr< TMatrix > Temp_factorised ( K_minus_shift_M->copy() );
        
        LDL::factorise( Temp_factorised.get(), acc_inverse, fac_opt );
    
        ///-----------------------------------------------------------------------------------------
        /// WARNING: Die Matrix 'Temp_factorised' (repraesentiert die Faktorisierung von K) 
        /// ist eigentlich nicht mehr symmetrisch. An dieser Stelle wird 'Temp_factorised' 
        /// aber auf symmetrisch gesetzt, um zu erzwingen dass der lineare Operator 'Temp_precond' 
        /// selbstadjungiert ist. Die Symmetrie Eigenschaft wird aber gleich wieder zurueckgenommen, 
        /// sobald 'Temp_factorised' erstellt wurde. (Vgl. Heft 9 S. 2)
        ///-----------------------------------------------------------------------------------------
        Temp_factorised->set_symmetric();
        
        unique_ptr< TLinearOperator >  Temp_precond( LDL::inv_matrix( Temp_factorised.get(), matform, fac_opt ) );
        
        ///---------------------------------------------------
        /// WARNING: set factorisation nonsymmetric back again
        ///---------------------------------------------------
        Temp_factorised->set_nonsym();
        
        //////////////////////////////////////////////////////////////////////////////
        //
        // Check if the computed inverse is a good preconditioner
        //
        //////////////////////////////////////////////////////////////////////////////
        
        //---------------------------------------------
        // set arbitrary error threshold smaller than 1
        //---------------------------------------------
        const real error_threshold = real(1e-2);
        
        TSpectralNorm norm;
        
        const real precond_error = norm.inv_approx( K_minus_shift_M, Temp_precond.get() );
    
        count_try++;
    
        if ( precond_error < error_threshold )
        {                                 
            //////////////////////////////////////////////////////////////////////////
            //
            // If preconditioner is good enough set ouput and print information
            //
            //////////////////////////////////////////////////////////////////////////
            TOC;
            
            K_minus_shift_M_factorised = Temp_factorised.release();
            K_minus_shift_M_precond    = Temp_precond.release();
                        
            _parameter_EVP_transformation.eps_rel_precond_final = eps_rel_precond;
            _parameter_EVP_transformation.precond_error         = precond_error;
            _parameter_EVP_transformation.precond_count_try     = count_try;
            
            LOG( "" );
            LOG( to_string("(TEigenBase) comp_precond : done in %.2fs", toc.seconds() ));
            LOGHLINE;
            LOG( to_string("    got good inverse approx. of (K-shift*M) using relative accuracy %.2e",eps_rel_precond) );
            LOG( to_string("    approximation error of preconditioner is %.2e",precond_error) );
            LOG( to_string("    try number %d of %d was successful",count_try,number_of_tries) );
            LOGHLINE;
            
            break;
        }// if
        else if ( count_try < number_of_tries )
        {
            LOG( "" );
            LOG( to_string("(TEigenBase) comp_precond :"));
            LOGHLINE;
            LOG( to_string("    got bad inverse approx. of (K-shift*M) using relative accuracy %.2e",eps_rel_precond) );
            LOG( to_string("    approximation error of preconditioner is %.2e",precond_error) );
            LOG( to_string("    try number %d of %d was NOT successful",count_try,number_of_tries) );
            LOGHLINE;
            
            //////////////////////////////////////////////////////////////////////////
            //        
            // Initialise next try
            //
            //////////////////////////////////////////////////////////////////////////
            
            eps_rel_precond = eps_rel_precond / real(10);
            
        }// else if 
        else
        {
            TOC; 
            
            LOG( to_string("(TEigenBase) comp_precond : done in %.2fs and no good preconditioner was found ", toc.seconds() ));
            
            HERROR( ERR_CONSISTENCY, "(TEigenBase) comp_precond", "no good preconditioner found" );    
        }// else        
    }// while
    
}






void
TEigenBase::transform_problem ( const TMatrix * K, 
                                 const TMatrix * M )
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Transform implicitely the generalized EVP (K,M) into an equavilent standard EVP
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    //---------------------------------------------
    // Set the internal matrix pointers of K and M
    //---------------------------------------------
    _K = K;
    _M = M;
    
    const real shift = _parameter_EVP_transformation.shift;
    
    if ( _parameter_EVP_transformation.transform_symmetric )
    {
        //==============================================================================================
        //
        //                      K*x = lambda*M*x  <==>  M_tilde*y = mu*y  with 
        //                    
        //    mu = 1/(lambda-shift), M_tilde = L^{-1}*M*L^{-T}, y = L^{T}*x, (K-shift*M) = L*L^{T}
        //
        //==============================================================================================
        
        if ( shift == 0 )
        {
            //---------------------------------------------
            // Compute the Cholesky factorisation K = L*L^T 
            //---------------------------------------------
            factorise ( K, _L, _parameter_EVP_transformation.eps_rel_exact );
        
        }// if 
        else
        {
            //-----------------------------------------------------
            // Compute the matrix K-shift*M 
            //-----------------------------------------------------
            comp_shifted_matrix ( K, M, shift, _K_minus_shift_M, _parameter_EVP_transformation.eps_rel_exact );
        
            //-----------------------------------------------------
            // Compute the Cholesky factorisation K-shift*M = L*L^T 
            //-----------------------------------------------------
            factorise ( _K_minus_shift_M, _L, _parameter_EVP_transformation.eps_rel_exact );
            
        }// else
    }// if
    else
    {
        //==============================================================================================
        //
        //                         K*x = lambda*M*x  <==>  C*x = mu*x  with 
        //                    
        //                     mu = 1/(lambda-shift), C = (K-shift*M)^{-1} * M
        //
        //==============================================================================================
        
        if ( shift == real(0) )
        {
            //------------------------------------------------------------------------
            // Compute preconditioner for K, i.e., compute an approximate inverse of K 
            //------------------------------------------------------------------------
            comp_precond ( K, _K_minus_shift_M_factorised, _K_minus_shift_M_precond, 
                                _parameter_EVP_transformation.eps_rel_precond_start );
            
        }// if 
        else
        {
            //------------------------------------------------------------------------
            // Compute the matrix K-shift*M 
            //------------------------------------------------------------------------
            comp_shifted_matrix ( K, M, shift, _K_minus_shift_M, _parameter_EVP_transformation.eps_rel_exact );
            
            //------------------------------------------------------------------------
            // Compute preconditioner for K, i.e., compute an approximate inverse of K 
            //------------------------------------------------------------------------
            
            comp_precond ( _K_minus_shift_M, _K_minus_shift_M_factorised, _K_minus_shift_M_precond, 
                                _parameter_EVP_transformation.eps_rel_precond_start );
            
        }// else
    }// else
}
  




void 
TEigenBase::backtransform_eigenvectors_sym ( const TMatrix * L,
                                              TDenseMatrix *  Z_standard ) const
{
    if ( Z_standard == nullptr || L == nullptr )
       HERROR( ERR_ARG, "(TEigenBase) backtransform_eigenvectors_sym", "argument is nullptr" );
       
    if ( ! _parameter_EVP_transformation.transform_symmetric )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) backtransform_eigenvectors_sym", "" );
    
        
    /////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Transform the eigenvector 'y' of the symmetric standard eigenvalue problem back to an 
    // eigenvector 'x' of the original general eigenvalue problem by computing 'x = L^{-T} * y'
    //
    /////////////////////////////////////////////////////////////////////////////////////////////
    const solve_option_t  solve_L_opts( point_wise, general_diag );
    
    auto transform_eigenvector = 
        [ L, Z_standard, & solve_L_opts ] ( const tbb::blocked_range< uint > & r )
        {        
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                //===================================================
                // Initialise the eigenvector 'y' of the standard EVP                                  
                // (y is a reference to the column j of Z_stanadard)
                //===================================================
                TScalarVector  y( Z_standard->column( j ) );
                        
                //==============================================
                // Compute the vector x = L^{-T} * y via solving 
                // the lower triangular system L^{T} * x = y
                //==============================================
                solve_lower( MATOP_TRANS, L, & y, solve_L_opts );
                
                //===================================================================================
                // The eigenvector 'y' has been transformed to the eigenvector 'x' of the original 
                // generalized EVP. Since 'y' is a reference to the corresponding column of 
                // 'Z_standard' this matrix contains the eigenvectors of the original generalized EVP
                //===================================================================================
            }// for
        };
    
    const size_t  n_ev = Z_standard->cols();
    
    bool  do_parallel = true;
    
    if ( CFG::nthreads() == 1 )
        do_parallel = false;
    
    if ( do_parallel ) 
        tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(n_ev) ), transform_eigenvector );
    else
        transform_eigenvector ( tbb::blocked_range< uint >( uint(0), uint(n_ev) ) );
}







void
TEigenBase::iterate_sym ( TVector * v ) const
{       
    if ( _M == nullptr || _L == nullptr || v == nullptr )
        HERROR( ERR_ARG, "(TEigenBase) iterate_sym", "argument is nullptr" );
        
    if ( ! _parameter_EVP_transformation.transform_symmetric )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) iterate_sym", "" ); 
    
    //////////////////////////////////////////////////////////////////////////
    //
    // Compute   v <== L^{-1} * M * L^{-T} * v    where the 
    //
    // Cholesky factor L is given by (K-shift*M) = L*L^{T}
    //
    //////////////////////////////////////////////////////////////////////////
    const solve_option_t  solve_L_opts( point_wise, general_diag );
    
    //----------------------------------------------------------------------
    // Compute         v <== L^{-T} * v          (This is done by computing 
    //
    // y = L^{-T} * v via solving the lower triangular system L^{T} * y = v)
    //----------------------------------------------------------------------
    solve_lower( MATOP_TRANS, _L, v, solve_L_opts );
            
    //-----------------------------------------------
    // Compute the vector   v <== M * v
    //-----------------------------------------------
    unique_ptr< TVector >  v_temp( v->copy() );
    
    _M->mul_vec( real(1), v_temp.get(), real(0), v );
    
    //-------------------------------------------------------------------
    // Compute         v <== L^{-1} * v       (This is done by computing 
    //
    // y = L^{-T} * v via solving the lower triangular system L * y = v)
    //-------------------------------------------------------------------
    solve_lower( MATOP_NORM, _L, v, solve_L_opts );
}



    
void
TEigenBase::iterate_nonsym ( TVector * v ) const 
{       
    if ( _K_minus_shift_M_precond == nullptr ||  _M == nullptr || v == nullptr )
        HERROR( ERR_ARG, "(TEigenBase) iterate_nonsym", "argument is nullptr" );

    if ( _parameter_EVP_transformation.transform_symmetric )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) iterate_nonsym", "" ); 
        
    //////////////////////////////////////////////////////////////////////////
    //
    // Compute   v <== (K-shift*M)^{-1} * M * v   
    //
    //////////////////////////////////////////////////////////////////////////
        
    //-----------------------------------------------
    // Compute the vector     v <== M * v
    //-----------------------------------------------
    unique_ptr< TVector >  v_temp( v->copy() );
    
    _M->mul_vec( real(1), v, real(0), v_temp.get() );
        
    //------------------------------------------------------------------
    // Compute   v <== (K-shift*M)^{-1} * v   (This is done by computing 
    //
    // y = (K-shift*M)^{-1} * v via solving (K-shift*M) * y = v for y)
    //------------------------------------------------------------------
    TSolverInfo     info;
    // TStopCriterion  sstop( 100, real(1e-8), real(1e-8), real(1e6) );
    TStopCriterion  sstop( 150, real(1e-10), real(1e-10), real(1e6) );
    TAutoSolver     solver( sstop );
            
    if ( _parameter_EVP_transformation.shift == real(0) )
        solver.solve( _K,               v, v_temp.get(), _K_minus_shift_M_precond, &info );
    else
        solver.solve( _K_minus_shift_M, v, v_temp.get(), _K_minus_shift_M_precond, &info );
        
    if ( !info.has_converged() )
    {   
        LOG( "(TEigenBase) iterate_nonsym : WARNING! Iteration did not converge" );
    }// if
}


void
TEigenBase::iterate_nonsym_partial ( TVector * v ) const
{       
    if ( _K_minus_shift_M_precond == nullptr || v == nullptr )
        HERROR( ERR_ARG, "(TEigenBase) iterate_nonsym_partial", "argument is nullptr" );
        
    if ( _parameter_EVP_transformation.transform_symmetric )
        HERROR( ERR_CONSISTENCY, "(TEigenBase) iterate_nonsym", "" ); 

    ///////////////////////////////////////////////////////////////////////
    //
    // Compute  v <== (K-shift*M)^{-1} * v   (This is done by computing 
    //
    // y = (K-shift*M)^{-1} * v via solving (K-shift*M) * y = v for y)
    //
    ///////////////////////////////////////////////////////////////////////
    unique_ptr< TVector >  v_temp( v->copy() );
    
    TSolverInfo     info;
    // TStopCriterion  sstop( 100, real(1e-8), real(1e-8), real(1e6) );
    TStopCriterion  sstop( 150, real(1e-10), real(1e-10), real(1e6) );
    TAutoSolver     solver( sstop );
            
    if ( _parameter_EVP_transformation.shift == real(0) )
        solver.solve( _K,               v, v_temp.get(), _K_minus_shift_M_precond, &info );
    else
        solver.solve( _K_minus_shift_M, v, v_temp.get(), _K_minus_shift_M_precond, &info );
    
        
    if ( !info.has_converged() )
    {   
        LOG( "(TEigenBase) iterate_nonsym_partial : WARNING! Iteration did not converge" );
    }// if
}






//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// TEigenArnoldi
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////




void
TEigenArnoldi::debug_update_Q_and_A_hat ( TDenseMatrix * A_hat, 
                                           TDenseMatrix * Q ) 
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // 
    // Orhtogonality of Q is very important for the eigenpair approximation. This debug routine 
    // checks the orthogonality of Q, reorthogonalize Q and update A_hat by computing A_hat = Q^{T}*A*Q
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    
    //---------------------------------------
    // Check orhtogonality of Q
    //---------------------------------------
    EIGEN_MISC::check_ortho( Q );
    
    //---------------------------------------
    // Reorthogonalize Q
    //---------------------------------------
    const size_t n = Q->rows();
    const size_t m = Q->cols();
        
    BLAS::Matrix< real >  Q_blas( n, m );
        
    for ( size_t j = 0; j < Q->cols(); j++ )
    {
        for ( size_t i = 0; i < Q->rows(); i++ )
        {
            const real entry = Q->entry(i,j);
            
            Q_blas(i,j)= entry;
        }// for
    }// for
        
    BLAS::Matrix< real >  R( m, m );
    
    qr( Q_blas, R );
        
    for ( size_t j = 0; j < Q->cols(); j++ )
    {
        for ( size_t i = 0; i < Q->rows(); i++ )
        {
            const real entry = Q_blas(i,j);
            
            Q->set_entry(i,j,entry);
        }// for
    }// for
    
    //---------------------------------------
    // Check orhtogonality of Q again
    //---------------------------------------
    EIGEN_MISC::check_ortho( Q );
    
    
    //////////////////////////////////////////////////////////////////////////////////
    //
    // Compute A_hat = Q^{T} * A * Q
    //
    //////////////////////////////////////////////////////////////////////////////////
    
    //
    // Compute A * Q
    // 
    unique_ptr< TDenseMatrix >  A_Q( new TDenseMatrix( n, m ) );
    
    A_Q->set_ofs ( _K->col_ofs(), _K->col_ofs() );
            
    for ( size_t j = 0; j < m; j++ )
    {
        TScalarVector  A_Q_j( A_Q->column( j ) );    
        TScalarVector  Q_j  ( Q->column  ( j ) );    
    
        iterate_arnoldi( & Q_j, & A_Q_j );
    }// for
    
    //
    // Compute Q^{T} * (A * Q)
    //
    unique_ptr< TMatrix >  A_hat_copy( A_hat->copy() );
        
    const TTruncAcc acc( real(0) );
        
    multiply( real(1), MATOP_TRANS, Q, MATOP_NORM, A_Q.get(), real(0), A_hat, acc );
        
    LOG( to_string("(TEigenArnoldi) debug_update_Q_and_A_hat : relative diffnorm = %.2e",diffnorm_2( A_hat_copy.get() , A_hat )) );
}




void
TEigenArnoldi::iterate_arnoldi ( const TVector * v, 
                                  TVector *       y ) 
{       
    if ( ! get_transform_symmetric() )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) iterate_arnoldi", "not yet implemented" );

    /////////////////////////////////////////////
    //
    // Compute y = L^{-1} * M * L^{-T} * v  
    //
    /////////////////////////////////////////////
    
    y->assign( real(1), v );
    
    iterate_sym( y );
    
    _parameter_arnoldi.count_iterate++;
}


void            
TEigenArnoldi::comp_arnoldi_basis_ortho_and_update ( const vector < TVector * > & q,
                                                      const size_t                 m,
                                                      TVector *                    q_tilde,
                                                      TDenseMatrix *               A_hat_temp ) const
{

    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply full re-orthogonalisation (Modified Gram-Schmidt)
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_arnoldi.re_ortho_strategy == FULL_ORTHO )
    {
        
        for ( size_t i = 0; i <= m; i++ )
        {
            const real factor = re( dot( q[i], q_tilde ) );
            
            q_tilde->axpy( -factor, q[i] );
            
            // Since the problem is symmetric the matrix 'A_hat' should be tridiagonal
            // and correspondingly only these values are set. Furthermore the subsequent
            // used eigensolver supports right now only symmetric problems
            if ( i == m || i == m-1 )
                A_hat_temp->set_entry( i, m, factor );
            
        }// for
        
        return;
    }// if


    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply full re-orthogonalisation in parallel
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    
    if ( _parameter_arnoldi.re_ortho_strategy == FULL_ORTHO_PARALLEL )
    {
    
        //----------------------------------------------------------------
        // Re-orthogonalize with all previous basis vectors i with i < m-1
        //----------------------------------------------------------------
        if ( m-1 > 0 )
        {
            if ( m-1 <= _parameter_arnoldi.size_serial_ortho || q_tilde->size() < 500 )
            {
                reortho_q_tilde_in_serial ( q, idx_t(0), idx_t(m-2), q_tilde );
            }// if
            else
            {
                tbb::task & root = * new( tbb::task::allocate_root() ) reortho_q_tilde_task_t( q, idx_t(0), idx_t(m-2), q_tilde, _parameter_arnoldi.size_serial_ortho );

                tbb::task::spawn_root_and_wait( root );
            }// else
        }// if
    
    
        //----------------------------------------------------------------------------------------
        // Orthogonalise with the two last-computed basis vectors and update A_hat correspondingly
        //----------------------------------------------------------------------------------------
        for ( size_t i = m-1; i <= m; i++ )
        {
            const real factor = re( dot( q[i], q_tilde ) );
            
            q_tilde->axpy( -factor, q[i] );
            
            // Since the problem is symmetric the matrix 'A_hat' should be tridiagonal
            // and correspondingly only these values are set. Furthermore the subsequent
            // used eigensolver supports right now only symmetric problems
            A_hat_temp->set_entry( i, m, factor );
            
        }// for
        
        return;
    }// if
    
    
    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply selective re-orthogonalisation, cf. [Templates for Solution of Algebraic EVPs]
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_arnoldi.re_ortho_strategy == SELECTIVE_ORTHO )
    {
        HERROR( ERR_ARG, "(TEigenArnoldi) comp_arnoldi_basis_ortho_and_update", "not implemented" );
        
        return;
    }// if


    HERROR( ERR_CONSISTENCY, "(TEigenArnoldi) comp_arnoldi_basis_ortho_and_update", "instable re-orthogonalisation strategy selected" );
    
    
    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply partial orthogonalisation
    //
    // (Benchmarks have shown that this doesn't work at all and the resulting basis vectors
    // are not orthogonal leading to bad respectibely meaningless eigenpair approximations)
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_arnoldi.re_ortho_strategy == PARTIAL_ORTHO )
    {
        for ( size_t i = m-1; i <= m; i++ )
        {
            const real factor = re( dot( q[i], q_tilde ) );
            
            q_tilde->axpy( -factor, q[i] );
            
            A_hat_temp->set_entry( i, m, factor );
            
            // NOTE: Don't use this strategy
            // Since the problem is symmetric the matrix 'A_hat' should be tridiagonal
            // and correspondingly only these values are set and in theory orthogonalisation
            // is only necessary with these vectors. However, since computation
            // is not exact the theoretical orthogonality if the new iteration vector
            // with the "first" one gets lost and this computed A_hat is not exactly equal to
            //  Q^{T} * A * Q. Especially is the matrix  Q^{T} * A * Q is not any more really 
            // tridiagonal and even not upper Hessenberg since the matrix Q corresponds not 
            // exactly to the Arnoldi basis since the orthogonalisation issue described above.
        }// for
        
        return;
    }// if
    

    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply mix between full and partial re-orthogonalisation
    //
    // (Benchmarks have shown that this doesn't work at all and the resulting basis vectors
    // are not orthogonal leading to bad respectibely meaningless eigenpair approximations)
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_arnoldi.re_ortho_strategy == MIXED_ORTHO )
    {
        if ( m%2 == 0 )
        {
            for ( size_t i = 0; i <= m; i++ )
            {
                const real factor = re( dot( q[i], q_tilde ) );
                
                q_tilde->axpy( -factor, q[i] );
                
                if ( i == m || i == m-1 )
                    A_hat_temp->set_entry( i, m, factor );
            }// for
        }// if
        else
        {
            for ( size_t i = m-1; i <= m; i++ )
            {
                const real factor = re( dot( q[i], q_tilde ) );
                
                q_tilde->axpy( -factor, q[i] );
                
                A_hat_temp->set_entry( i, m, factor );
            }// for
        }// else
        
        return;
    }// if
    
    
    ////////////////////////////////////////////////////////////////////////////////////////
    //
    // For Debugging:
    //
    // Apply full re-orthogonalisation (not stable original Gram-Schmidt version)
    //
    ////////////////////////////////////////////////////////////////////////////////////////
    if ( false )
    {
        unique_ptr< TVector >  q_tilde_old ( q_tilde->copy() );
        
        for ( size_t i = 0; i <= m; i++ )
        {
            const real factor = re( dot( q[i], q_tilde_old.get() ) );
            
            q_tilde->axpy( -factor, q[i] );
            
            // Since the problem is symmetric the matrix 'A_hat' should be tridiagonal
            // and correspondingly only these values are set. Furthermore the subsequent
            // used eigensolver supports right now only symmetric problems
            if ( i == m || i == m-1 )
                A_hat_temp->set_entry( i, m, factor );
            
        }// for
        
        return;
    }// if
}



real
TEigenArnoldi::comp_arnoldi_basis ( const size_t     m_max,
                                     const TVector *  v, 
                                     TDenseMatrix *   A_hat, 
                                     TDenseMatrix *   Q, 
                                     const real       rel_error_bound )
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //  
    //
    // Determine the Arnoldi basis of the vector 'v' and the matrix A:= L^{-1}*M*L^{-T}.
    // The Arnoldi basis is contained in 'Q' and the projected EVP in 'A_hat:=Q^{T}*A*Q'. 
    // The parameter 'rel_error_bound' controls the relative accuracy of the associated subspace
    //
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    
    if ( m_max < 1 ) 
        HERROR( ERR_CONSISTENCY, "(TEigenArnoldi) comp_arnoldi_basis", "" );

    //-------------------------------------------
    // Initialise array keeping the Arnoldi basis
    //-------------------------------------------
    vector < TVector * >  q( m_max, nullptr );
    
    //-------------------------------------------
    // Initialise auxiliary data 
    //-------------------------------------------
    unique_ptr< TDenseMatrix >  A_hat_temp( new TDenseMatrix( m_max, m_max ) );
    unique_ptr< TVector >       q_tilde   ( _K->col_vector() );
    
    ////////////////////////////////////////////////////////////////////////////////
    //
    // Initialise the start values of the Arnoldi method
    //
    ////////////////////////////////////////////////////////////////////////////////
    A_hat_temp->scale( real(0) );    
    
    //------------------------------
    // Determine q_tilde 
    //------------------------------
    q[0] = v->copy().release();
    q[0]->scale( real(1)/q[0]->norm2() );
    
    iterate_arnoldi( q[0], q_tilde.get() );
    
    //----------------------------
    // Determine gamma 
    //----------------------------
    real gamma = q_tilde->norm2();
    
    //------------------------------
    // Update A_hat_temp and q_tilde 
    //------------------------------
    real entry = re( dot( q[0], q_tilde.get() ) );
    A_hat_temp->set_entry( 0, 0, entry );
    
    q_tilde->axpy( -A_hat_temp->entry(0,0), q[0] );
    
    // Beachte: Auf Uebungsblatt 7 zu [Numerik von EWP] ist 'm' um 1 groesser
    size_t m = 0;
    
    real rel_error = q_tilde->norm2() / gamma;
    
    ////////////////////////////////////////////////////////////////////////////////////
    //
    // Apply the Arnoldi method
    //
    ////////////////////////////////////////////////////////////////////////////////////
    if ( m_max > 1 )
    {
        A_hat_temp->set_entry( 1, 0, q_tilde->norm2() );
    
        while ( A_hat_temp->entry(m+1,m)  >= rel_error_bound * gamma )
        {   
            //------------------
            // Determine q_(m+1)
            //------------------
            q[m+1] = q_tilde->copy().release();
            q[m+1]->scale( real(1)/ A_hat_temp->entry(m+1,m) );
            
            //------------------
            // Increase m
            //------------------
            m = m + 1;
    
            //------------------
            // Update q_tilde 
            //------------------
            iterate_arnoldi( q[m], q_tilde.get() );
            
            //------------------
            // Determine gamma
            //------------------
            gamma = q_tilde->norm2();
            
            //------------------------------------------------------------------------
            // Orthogonalise q_tilde with existing Arnoldi basis and update A_hat_temp
            //------------------------------------------------------------------------
            comp_arnoldi_basis_ortho_and_update ( q, m, q_tilde.get(), A_hat_temp.get() );
             
            //-------------------------------------------------------------
            // Compute relatve "error" for the return value of the function
            //-------------------------------------------------------------
            rel_error = q_tilde->norm2() / gamma;
            
            if ( m >= m_max-1 ) 
                break;
            
            //------------------------------------------------
            // Update A_hat_temp
            //------------------------------------------------
            A_hat_temp->set_entry( m+1, m , q_tilde->norm2() );
        }// while
    }// if
    
    ////////////////////////////////////////////////////////////////////////////////////
    // 
    // Copy results into output data
    //
    ////////////////////////////////////////////////////////////////////////////////////
    
    //-----------------------------
    // Update size of Arnoldi basis
    //-----------------------------
    m = m + 1;
    
    _parameter_arnoldi.basis_size = m;
            
    //------------------------------------------
    // Set matrix Q containing the Arnoldi basis
    //------------------------------------------
    const size_t n = _K->rows(); 
    
    Q->set_size( n, m );
    Q->set_ofs ( _K->col_ofs(), _K->col_ofs() );
    
    for ( size_t j = 0; j < m; j++ )
    {
        for ( size_t i = 0; i < n; i++ )
        {
            entry = q[j]->entry(i);
            
            Q->set_entry( i, j, entry );
        }// for
        
        delete q[j];
            
        q[j] = nullptr;
    }// for
    
    //--------------------------------------------------
    // Determine reduced Eigenvalue problem matrix A_hat
    //--------------------------------------------------
    A_hat->set_size( m, m );
    A_hat->set_ofs ( _K->col_ofs(), _K->col_ofs() );
    
    for ( size_t j = 0; j < m; j++ )
    {
        for ( size_t i = 0; i < m; i++ )
        {
            entry = A_hat_temp->entry(i,j);
            
            A_hat->set_entry( i, j, entry );
        }// for
    }// for
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Adjust "A_hat" for suseqent computation and return relative "error" of last iteration
    //
    // (In theory "A_hat" should be symmetric and tridiagonal, however, depending on the accuracy of the 
    // compuation this is not always the case. The symmetry/tridiagonality of the computed matrix is 
    // increased if in the orhthogonalisation process the new iterate is orthogonalised against all 
    // previous computed basis vectors. For the subsequent eigendecomposition of "A_hat" the matrix is
    // nevertheless set to be symmetric, because TEigenLapack supports only symmetric matrices right now.)
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    A_hat->set_symmetric();
    
    A_hat->set_ofs( _K->col_ofs(), _K->col_ofs() );
    
    return rel_error;
}





size_t
TEigenArnoldi::comp_decomp_arnoldi ( const size_t  n_ev_searched, 
                                      TMatrix * &   D_standard,
                                      TMatrix * &   Z_standard ) 
{    
    if ( _parameter_arnoldi.rel_error_bound > 1e-7 )
            HERROR( ERR_CONSISTENCY, "(TEigenArnoldi) comp_decomp_arnoldi", "eps is too large, convergency could be destroyed" );
            
    if ( ! get_transform_symmetric() )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) comp_decomp_arnoldi", "not yet implemented" );
                 
    if ( n_ev_searched == 0 )
        return n_ev_searched;
        
    size_t n_ev = n_ev_searched;
    
    if ( n_ev > _K->rows() )
        n_ev = _K->rows();
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the n_ev largest eigenpairs of the matrix A:= L^{-1}*M*L^{-T}
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
     
    //---------------------------------------------------
    // Create random starting vector v
    //---------------------------------------------------
    unique_ptr< TVector >  v( _K->col_vector() ); 
    
    v->fill_rand( _parameter_arnoldi.seed );
    v->scale    ( real(1)/v->norm2() );
    
    _parameter_arnoldi.seed++;
    
    //---------------------------------------------------
    // Determine maximum size of Arnoldi-Basis 
    //---------------------------------------------------
    size_t m_max = _parameter_arnoldi.factor_basis_size * n_ev;
    
    if ( m_max < _parameter_arnoldi.min_basis_size ) 
        m_max = _parameter_arnoldi.min_basis_size;
    
    if ( m_max > _K->rows() )
        m_max = _K->rows();
    
    //---------------------------------------------------
    // Compute Arnoldi basis / Krylov space
    //---------------------------------------------------
    unique_ptr< TDenseMatrix > A_hat( new TDenseMatrix );
    unique_ptr< TDenseMatrix > Q    ( new TDenseMatrix );
    
    const real rel_error = comp_arnoldi_basis ( m_max, v.get(), A_hat.get(), Q.get(), _parameter_arnoldi.rel_error_bound );
    
    //---------------------------------------------------
    // Compute eigenvalue decomposition of A_hat
    //---------------------------------------------------
    TEigenLapack eigensolver;
    
    eigensolver.set_test_residuals( false );
    eigensolver.set_test_pos_def  ( false );
    
    unique_ptr< TDenseMatrix >  D_hat( new TDenseMatrix );
    unique_ptr< TDenseMatrix >  Z_hat( new TDenseMatrix );
    
    const size_t n_ev_computed = eigensolver.comp_decomp ( A_hat.get(), D_hat.get(), Z_hat.get() );
    
    LOG( to_string("(TEigenArnoldi) comp_decomp_arnoldi : rel_error = %2.e",rel_error) );

    //EIGEN_MISC::check_ortho( Q.get() );
    //TEigenAnalysis eigen_analysis;
    //eigen_analysis.analyse_solution( A_hat.get(), D_standard, Z_hat.get() );
    
    //---------------------------------------------------
    // Compute Ritz-Vectors Z_standard = Q*Z_hat
    //---------------------------------------------------
    unique_ptr< TMatrix > Z_ritz( Q->mul_right( real(1), Z_hat.get(), MATOP_NORM, MATOP_NORM ) );
        
    //-----------------------------------------------------------------------------------
    // Set the offsets of the matrices D and Z (The indexsets are defined in this
    // style that the matrix operation  K*Z-M*Z*D respectively well defined)
    //-----------------------------------------------------------------------------------
    D_standard = D_hat.release();
    Z_standard = Z_ritz.release();
    
    D_standard->set_ofs( _K->col_ofs(), _K->col_ofs() );
    Z_standard->set_ofs( _K->col_ofs(), _K->col_ofs() );
    
    return n_ev_computed;
}


                          
void 
TEigenArnoldi::print_options () const
{
    OUT( "" );
    OUT( "(TEigenArnoldi) print_options :" );
    HLINE;
    OUT( to_string("    basic options:      n_ev_searched  = %d",_parameter_base.n_ev_searched ) );
    OUT( to_string("                        test_pos_def   = %d",_parameter_base.test_pos_def ) );
    OUT( to_string("                        test_residuals = %d",_parameter_base.test_residuals ) );
    HLINE;
    OUT( to_string("    EVP transformation: transform_symmetric = %d",_parameter_EVP_transformation.transform_symmetric ) );
    OUT( to_string("                        shift               = %.2f",_parameter_EVP_transformation.shift ) );
    OUT( to_string("                        eps_rel_exact       = %.2e",_parameter_EVP_transformation.eps_rel_exact ) );
    HLINE;
    OUT( to_string("    Arnoldi parameter:  rel_error_bound   = %.2e",_parameter_arnoldi.rel_error_bound ) );
    OUT( to_string("                        factor_basis_size = %d",_parameter_arnoldi.factor_basis_size ) );
    OUT( to_string("                        min_basis_size    = %d",_parameter_arnoldi.min_basis_size ) );
    OUT( to_string("                        size_serial_ortho = %d",_parameter_arnoldi.size_serial_ortho ) );
    
    
    if ( _parameter_arnoldi.re_ortho_strategy == FULL_ORTHO_PARALLEL )
        OUT( to_string("                        re_ortho_strategy = FULL_ORTHO_PARALLEL") );
    else if ( _parameter_arnoldi.re_ortho_strategy == FULL_ORTHO )
        OUT( to_string("                        re_ortho_strategy = FULL_ORTHO") );
    else
        OUT( to_string("                        re_ortho_strategy = WARNING: wrong strategy selected") );
    HLINE;
}    
   

                             
void
TEigenArnoldi::print_summary ( const TMatrix * D ) const
{
    OUT( "" );
    OUT( "(TEigenArnoldi) print_summary :" );
    HLINE;
    OUT( to_string("    eigenpairs of EVP (K,M) closest to the shift %.2f have been computed",get_shift() ) );
    HLINE;
    OUT( to_string("    dimension of (K,M)    = %d",_M->rows() ) );
    OUT( to_string("    n_ev searched         = %d",_parameter_base.n_ev_searched ) );
    OUT( to_string("    n_ev computed         = %d",D->rows() ) );
    OUT( to_string("    number of iterations  = %d",_parameter_arnoldi.count_iterate ) );
    OUT( to_string("    size of Arnoldi basis = %d",_parameter_arnoldi.basis_size ) );
    HLINE;    
    if ( !get_transform_symmetric() )
    {
        OUT( to_string("    eps_rel_precond_final = %2.e",_parameter_EVP_transformation.eps_rel_precond_final ) );
        OUT( to_string("    precond_error         = %2.e",_parameter_EVP_transformation.precond_error ) );
        OUT( to_string("    precond_count_try     = %d",_parameter_EVP_transformation.precond_count_try ) );
        HLINE;
    }// if
}


 
void 
TEigenArnoldi::backtransform_eigensolutions_arnoldi ( const TMatrix *  L,
                                                       const TMatrix *  D_standard,
                                                       const TMatrix *  Z_standard,
                                                       TDenseMatrix *   D,
                                                       TDenseMatrix *   Z ) const
{
    if ( L == nullptr || D == nullptr || D_standard == nullptr || Z == nullptr || Z_standard == nullptr )
       HERROR( ERR_ARG, "(TEigenArnoldi) backtransform_eigensolutions_arnoldi", "argument is nullptr" );
       
    if ( ! get_transform_symmetric() )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) backtransform_eigensolutions_arnoldi", "not yet implemented" );
        
    if ( get_shift() != real(0) )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) backtransform_eigensolutions_arnoldi", "not yet implemented" );   
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Transform the eigensolutions of the standard problem back to eigenpairs fo the original
    // general eigenvalue problem. If 'mu' is an eigenvalue of the standard problem then 
    // 'lambda = 1/mu' is an eigenvalue of the original general eigenvalue problem. 
    // If 'y' is an eigenvector of the symmetric standard eigenvalue problem then 
    // 'x' is an eigenvector of the original general eigenvalue problem with 'x = L^{-T} * y'
    //
    // Note that the eigenpairs of the standard problem contained in the data (D_standard,Z_standard) 
    // are ordered from the smallest to the largest eigenvalue. 
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    //-----------------------------------------------------------------------------------
    // Transform only the n_ev largest eigenpairs of (D_standard,Z_standard)
    //-----------------------------------------------------------------------------------
    const size_t n = _K->rows();
    const size_t n_ev_computed = D_standard->cols();
    size_t       n_ev = _parameter_base.n_ev_searched;
    
    if ( n_ev > n_ev_computed )
        n_ev = n_ev_computed;
    
    D->set_size( n_ev, n_ev );
    Z->set_size( n   , n_ev );
    
    D->scale( real(0) );
    Z->scale( real(0) );
    
    //-----------------------------------------------------------------------------------
    // Transformation of the eigenvalues 
    //-----------------------------------------------------------------------------------
    for ( size_t j = 0; j < n_ev; j++ )
    {
        const size_t index = n_ev_computed-1-j;
        const real   mu    = D_standard->entry( index, index );
            
        if ( ! ( mu > 0 )  )
            HERROR( ERR_CONSISTENCY, "(TEigenArnoldi) backtransform_eigensolutions_arnoldi", "" );
            
        const real lambda = real(1)/mu;
        
        D->set_entry( j, j, lambda );
    }// for
    
    //-----------------------------------------------------------------------------------
    // Transformation of the eigenvectors
    //-----------------------------------------------------------------------------------    
    auto reorder_columns = 
    [ Z_standard, Z, n, n_ev_computed ] ( const tbb::blocked_range< uint > & r )
    {        
        for ( auto  j = r.begin(); j != r.end(); ++j )
        {
            const size_t index = n_ev_computed-1-j;
    
            for ( size_t i = 0; i < n; i++ )
            {
                const real entry = Z_standard->entry( i, index );
                
                Z->set_entry( i,j, entry );
            }// for 
        }// for
    };
        
    bool  do_parallel = true;
    
    if ( CFG::nthreads() == 1 || n < 1000 )
        do_parallel = false;
    
    if ( do_parallel ) 
        tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(n_ev) ), reorder_columns );
    else
        reorder_columns ( tbb::blocked_range< uint >( uint(0), uint(n_ev) ) );
    
    //-----------------------------------------------------------------------------------
    // Set the offsets of the matrices D and Z (The indexsets are defined in this
    // style that the matrix operation  K*Z-M*Z*D respectively well defined)
    //-----------------------------------------------------------------------------------
    D->set_ofs( _K->col_ofs(), _K->col_ofs() );
    Z->set_ofs( _K->col_ofs(), _K->col_ofs() );
    
    //-----------------------------------------------------------------------------------
    // backtransform eigenvector by computing 'x = L^{-T} * y'
    //-----------------------------------------------------------------------------------
    backtransform_eigenvectors_sym ( L, Z );
}





size_t
TEigenArnoldi::comp_decomp ( const TMatrix *  K,
                              const TMatrix *  M,
                              TDenseMatrix  *  D,
                              TDenseMatrix  *  Z ) 
{   
    
    timer_arnoldi_t timer;
    
    //======================================================================
    //
    // Do some consistency checks
    //
    //======================================================================
    TICC( timer_all );
    TICC( timer_consistency );
    check_input_consistency( K, M, D, Z );
    TOCC( timer_consistency, timer.consistency );
    
    if ( ! get_transform_symmetric() )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) comp_decomp", "not yet implemented" );
        
    if ( get_shift() != real(0) )
        HERROR( ERR_NOT_IMPL, "(TEigenArnoldi) comp_decomp", "not yet implemented" );
        
        
    //======================================================================
    //
    // Handle tivial cases of the solution of the EVP K*x = lambda*M*x
    //
    //======================================================================
    const bool trivial_problem_solved = solve_trivial_problems ( K, M, D, Z );
    
    if ( trivial_problem_solved )
    {
        if ( _parameter_base.print_info )
        {
            print_options();
            print_summary( D );
        }// if
        
        return Z->cols();
    }// if
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Transform the generalized EVP K*x = lambda*M*x to the standard EVP A*y = mu*y 
    // with mu = 1/lambda, A = L^{-1} *M* L^{-T}, y = L^{T}*x and K = L*L^{T}
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////
    TICC( timer_transform );
    transform_problem ( K, M );
    TOCC( timer_transform, timer.transform );
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute the largest eigenpairs of the standard EVP A*y = mu*y with the Arnoldi Mehtod
    //
    //    
    //////////////////////////////////////////////////////////////////////////////////////////
        
    TMatrix * Z_standard = nullptr;
    TMatrix * D_standard = nullptr;
    
    TICC( timer_arnoldi );
    const size_t n_ev_computed = comp_decomp_arnoldi ( _parameter_base.n_ev_searched, D_standard, Z_standard );
    TOCC( timer_arnoldi, timer.arnoldi );
    
    //--------------------
    // Check orthogonality
    //--------------------
    if ( false )
        EIGEN_MISC::check_ortho( Z_standard );
    
    //////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Transform the eigensolution of the standard eigenvalue problem back
    // to the eigensolution of the original generalized eigenvalue problem
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////
    
    TICC( timer_backtransform );
    backtransform_eigensolutions_arnoldi ( _L, D_standard, Z_standard, D, Z );
    TOCC( timer_backtransform, timer.backtransform );
    TOCC( timer_all, timer.all );
    
    delete D_standard;
    delete Z_standard;
    
    //======================================================================
    //
    // Output
    //        
    //======================================================================
    if ( _parameter_base.print_info )
    {
        print_options();
        print_summary( D );
        
        #if DO_TEST >= 1
        timer.print_performance();
        #endif
    }// if
    
    
    //======================================================================
    //
    // Test results if wanted
    //
    //======================================================================
    if ( _parameter_base.test_residuals )
    {
        //----------------------------------------------
        // Check errors
        //----------------------------------------------
        TEigenAnalysis  eigen_analysis;
            
        eigen_analysis.set_verbosity( 0 );
        
        const bool large_errors_detected = eigen_analysis.analyse_vector_residual ( K, M, D, Z );
        
        if ( large_errors_detected )
        {
//             eigen_analysis.set_verbosity( 3 );
//         
//             eigen_analysis.analyse_vector_residual ( K, M, D, Z );            
        
            HERROR( ERR_NCONVERGED, "(TEigenArnoldi) comp_decomp", "large errors detected" );
        }// if
    }// if
    

    //======================================================================
    //
    // return number of computed eigenpairs
    //
    //======================================================================
    return D->cols();
}





}// namespace HAMLS
