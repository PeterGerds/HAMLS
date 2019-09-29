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
// File        : TEigenArpack.cc
// Description : eigensolver for H-matrices based on ARPACK library 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "hamls/TEigenArpack.hh"

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
// local functions 
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

struct timer_arpack_t
{
    double all;
    double consistency;
    double transform;
    double iterate;
    double saupd;
    double seupd;
    double backtransform;

    //constructor
    timer_arpack_t ()
    {
        all           = 0;
        consistency   = 0;
        transform     = 0;
        iterate       = 0;
        saupd         = 0;
        seupd         = 0;
        backtransform = 0;
    }
    
    void print_performance() 
    {
        OUT( "" );
        OUT( "(TEigenArpack) print_performance :" );
        HLINE;
        OUT( to_string("    task_scheduler_init::default_num_threads() = %d",tbb::task_scheduler_init::default_num_threads()) );
        OUT( to_string("    CFG::nthreads()                            = %d",CFG::nthreads()) );
        HLINE;
        OUT( to_string("    all done in %fs",all) );
        HLINE;
        OUT( to_string("    consistency   = %f %",consistency/all*100) );
        OUT( to_string("    transform     = %f %",transform/all*100) );
        OUT( to_string("    iterate       = %f %",iterate/all*100) );
        OUT( to_string("    saupd         = %f %",saupd/all*100) );
        OUT( to_string("    seupd         = %f %",seupd/all*100) );
        OUT( to_string("    backtransform = %f %",backtransform/all*100) );
        HLINE;
    }
};


void
copy_vector_2_array ( const TVector * y, 
                      real *          out, 
                      const size_t    n )
{
    if ( out == nullptr || y == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) copy_vector_2_array", "argument is nullptr" );
        
    if ( y->size() != n )
        HERROR( ERR_CONSISTENCY, "(TEigenArpack) copy_vector_2_array", "" );
        
    //------------------------------------------------
    // Copy information of vector 'y' into array 'out'
    //------------------------------------------------
    for ( size_t i = 0; i < n; i++ )
    {
        const real entry = y->entry(i);
        
        out[i] = entry;
    }// for
}


void
out_equal_in ( const real * in, 
               real *       out, 
               const size_t n )
{
    if ( out == nullptr || in == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) out_equal_in", "argument is nullptr" );
            
    for ( size_t i = 0; i < n; i++ )
        out[i] = in[i];
}


}// namespace



TScalarVector * 
TEigenArpack::get_vector_from_array ( const real * in,
                                            const size_t n ) const
{
    if ( in == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) get_vector_from_array", "argument is nullptr" );
        
    if ( _M->rows() != n )
        HERROR( ERR_CONSISTENCY, "(TEigenArpack) get_vector_from_array", "" );
    
    //--------------------------------------------------------------------------------------------
    //TODO: Routine noch optimieren indem real array direkt an Konstruktor vom TScalarVector
    // uebergeben wird. Dazu muss aber erstmal das array in einen BLAS::Vector transformiert werden.
    // Das sollte insgesamt effizienter sein. Mehrkosten sind aber erstmal unerheblich bezueglich 
    // Gesamtkomplexitaet
    //--------------------------------------------------------------------------------------------
    
    TScalarVector * v = new TScalarVector( _M->cols(), _M->col_ofs(), _M->is_complex() );
    
    for ( size_t i = 0; i < n; i++ )
    {
        const real entry = in[i];
        
        v->set_entry( i, entry );
    }// for
    
    return v;
}





void 
TEigenArpack::apply_B ( const real * in, 
                         real *       out,
                         const size_t n ) const
{
    if ( in == nullptr || out == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) apply_B", "argument is nullptr" );
        
    if ( get_transform_symmetric() )
    {     
        ////////////////////////////////////////////////////////////
        //
        // out = in
        //    
        ////////////////////////////////////////////////////////////
        out_equal_in ( in, out, n );
    }// if
    else
    {
        ////////////////////////////////////////////////////////////
        //
        // out = M * in
        //    
        ////////////////////////////////////////////////////////////
        
        //-------------------------------
        // Transform arrays into TVectors    
        //-------------------------------
        unique_ptr< TVector >  v( get_vector_from_array( in,  n ) );
        unique_ptr< TVector >  y( get_vector_from_array( out, n ) );
        
        //-------------------------------
        // y = M * v
        //-------------------------------
        _M->mul_vec( real(1), v.get(), real(0), y.get() );
        
        //-----------------------------
        // Transform TVector into array
        //-----------------------------
        copy_vector_2_array ( y.get(), out, n );    
    }// else
}
        

void 
TEigenArpack::apply_OP ( const real * in, 
                          real *       out,
                          const size_t n ) const
{
    if ( in == nullptr || out == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) apply_OP", "argument is nullptr" );
        
    //-----------------------------
    // Transform array into TVector
    //-----------------------------
    unique_ptr< TVector >  v( get_vector_from_array( in,  n ) );
            
    if ( get_transform_symmetric() )
    {
        ///////////////////////////////////////////////////////////////////////////////////
        //
        // out = M_tilde * in   with M_tilde = L^{-1}*M*L^{-T} and (K-shift*M) = L*L^{T}
        //
        ///////////////////////////////////////////////////////////////////////////////////
        iterate_sym ( v.get() );
    }// if
    else
    {
        ///////////////////////////////////////////////////////////////////////////////////
        //
        // out = C * in   with C = (K-shift*M)^{-1} * M 
        //
        ///////////////////////////////////////////////////////////////////////////////////
        iterate_nonsym ( v.get() );
    }// else
    
    //-----------------------------
    // Transform TVector into array
    //-----------------------------
    copy_vector_2_array ( v.get(), out, n );
}




void 
TEigenArpack::apply_OP_partial ( const real * in, 
                                  real *       out,
                                  const size_t n ) const
{
    if ( in == nullptr || out == nullptr )
        HERROR( ERR_ARG, "(TEigenArpack) apply_OP_partial", "argument is nullptr" );
        
    //-----------------------------
    // Transform array into TVector
    //-----------------------------
    unique_ptr< TVector >  v( get_vector_from_array( in,  n ) );
            
    if ( get_transform_symmetric() )
    {
        HERROR( ERR_CONSISTENCY, "(TEigenArpack) apply_OP_partial", "" );
    }// if
    else
    {
        ///////////////////////////////////////////////////////////////////////////////////
        //
        // out = C * in   with C = (K-shift*M)^{-1}
        //
        ///////////////////////////////////////////////////////////////////////////////////
        iterate_nonsym_partial ( v.get() );
    }// else
    
    //-----------------------------
    // Transform TVector into array
    //-----------------------------
    copy_vector_2_array ( v.get(), out, n );
}


                          
void 
TEigenArpack::print_options () const
{
    OUT( "" );
    OUT( "(TEigenArpack) print_options :" );
    HLINE;
    OUT( to_string("    basic options:      n_ev_searched  = %d",_parameter_base.n_ev_searched ) );
    OUT( to_string("                        test_pos_def   = %d",_parameter_base.test_pos_def ) );
    OUT( to_string("                        test_residuals = %d",_parameter_base.test_residuals ) );
    HLINE;
    OUT( to_string("    EVP transformation: transform_symmetric   = %d",_parameter_EVP_transformation.transform_symmetric ) );
    OUT( to_string("                        shift                 = %.2f",_parameter_EVP_transformation.shift ) );
    OUT( to_string("                        eps_rel_exact         = %.2e",_parameter_EVP_transformation.eps_rel_exact ) );
    OUT( to_string("                        precond_max_try       = %.d",_parameter_EVP_transformation.precond_max_try ) );
    OUT( to_string("                        eps_rel_precond_start = %.2e",_parameter_EVP_transformation.eps_rel_precond_start ) );
    HLINE;
    OUT( to_string("    Arpack parameter:   stopping_tol          =  %.2e",_parameter_arpack.stopping_tol) );
    HLINE;
}    
   

                             
void
TEigenArpack::print_summary ( const TMatrix * D ) const
{
    OUT( "" );
    OUT( "(TEigenArpack) print_summary :" );
    HLINE;
    OUT( to_string("    eigenpairs of EVP (K,M) closest to the shift %.2f have been computed",get_shift() ) );
    HLINE;
    OUT( to_string("    dimension of (K,M)        = %d",_M->rows() ) );
    OUT( to_string("    n_ev searched             = %d",_parameter_base.n_ev_searched ) );
    OUT( to_string("    n_ev computed             = %d",_parameter_arpack.iparam_5 ) );
    OUT( to_string("    Arnoldi Update iterations = %d",_parameter_arpack.iparam_3 ) );
    OUT( to_string("    number of OP*x operations = %d",_parameter_arpack.iparam_9 ) );
    OUT( to_string("    number of B*x operations  = %d",_parameter_arpack.iparam_10 ) );
    OUT( to_string("    number of re-orthogonal.  = %d",_parameter_arpack.iparam_11 ) );
    HLINE;    
    if ( !get_transform_symmetric() )
    {
        OUT( to_string("    eps_rel_precond_final = %2.e",_parameter_EVP_transformation.eps_rel_precond_final ) );
        OUT( to_string("    precond_error         = %2.e",_parameter_EVP_transformation.precond_error ) );
        OUT( to_string("    precond_count_try     = %d",_parameter_EVP_transformation.precond_count_try ) );
        HLINE;
    }// if
}




size_t 
TEigenArpack::comp_decomp ( const TMatrix *      K,
                             const TMatrix *      M,
                             TDenseMatrix  *      D,
                             TDenseMatrix  *      Z )
{
    timer_arpack_t timer;
    
    //======================================================================
    //
    // Do some consistency checks
    //
    //======================================================================
    TICC( timer_all );
    TICC( timer_consistency );
    check_input_consistency( K, M, D, Z );
    TOCC( timer_consistency, timer.consistency );

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

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Adjust parameters of ARPACK 
    //
    // (for clarity the same notation is used as in the ARPACK documentation)
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////


    //-----------------------------------------------------------------------
    // Set status parameter to 0 always on the first call (input/ouput value)
    //-----------------------------------------------------------------------
    int ido = 0;
    
    //------------------------------------------------------------------------
    // Specify the type of EVP ( 'I' for standard and 'G' for generalized EVP)
    //------------------------------------------------------------------------
    char bmat[2];
     
    if ( get_transform_symmetric() )
        bmat[0] = 'I';
    else
        bmat[0] = 'G';
    
    //---------------------
    // Problem size
    //---------------------
    const int n = K->rows();
    
    //--------------------------------------------------------
    // Specify which which of the Ritz values of OP to compute
    // (user options: "LA", "SA", "SM", "LM", "BE")
    //--------------------------------------------------------
    const char which[3] = "LM";  
    
    //-----------------------------------------------------------
    // Number of eigenvalues of OP to be computed. 0 < nev < n
    //-----------------------------------------------------------
    const int nev = std::min(int(_parameter_base.n_ev_searched), n-1);
    
    if ( nev == 0 )
        HERROR( ERR_CONSISTENCY, "(TEigenArpack) comp_decomp", "" );
    
    //--------------------------------------------------------------
    // Stopping criteria: the relative accuracy of the Ritz value
    // 'lambda[i]' is considered as acceptable if 
    // 
    //      bound[i] <= tol * |lambda[i]|
    //
    // If 'tol<=0' than in ARPACK the tolerance is set automatically 
    // to machine precission (i.e., intern is LAPACK used to compute 
    // this machine precission)
    //--------------------------------------------------------------
    const real tol = _parameter_arpack.stopping_tol;
    
    //------------------------------
    // residual vector (input/ouput)
    //------------------------------
    real * resid = new real[n];
    
    //----------------------------------------------------------
    // Number of Lanczos vectors which are used in iterations. 
    // It has to hold ncv > nev and ncv >= 2*nev is recommended
    //----------------------------------------------------------
    const int ncv = std::min(std::max(2*nev, 20), n);
        
    //---------------------------------------------------------------
    // The ncv columns of v contain the Lanczos basis vectors (Ouput)
    //---------------------------------------------------------------
    real * v = new real[n*ncv];
    ///NOTE: Here is the problem that range of the type 'integer' is to small to represenent n*ncv correctly
    //       when n*ncv is very large. Instead 'long' (size_t) is needed, however, ARPACK routines are 
    //       implemented using 'int'.
    
    //-----------------------
    // leading dimension of v
    //-----------------------
    const int ldv = n;
    
    //=============================================================================
    //
    // Parameter array of ARPACK (input/ouput)
    //
    //=============================================================================
    int * iparam= new int[11];
    
    //-----------------------------------------------------------------------------
    // 1 means ARPACK applies exact shifts, 0 means shifts are provided by the user
    //-----------------------------------------------------------------------------
    iparam[0] = 1; 
    
    // has something to do with postprocess -> TODO
    //iparam[1] = 1; 
    
    //------------------------------------------------------------
    // Input: Maximum number of Arnoldi update iterations allowed
    // Ouput: Actual number of Arnoldi update iterations taken
    //------------------------------------------------------------
    iparam[2] = std::max(  300, int( std::ceil(2*n/std::max(ncv,1)) )  ); 
    
    // iparam[3] is blocksize to be used in the recurrence (the code works currently only for blocksize=1 -> TODO)

    // iparam[4] is number of converged "Ritz" values (i.e. values which satisfy the convergency criterion)
    
    // iparam[5] not longer referenced (implicit restarting is always used)
    
    //---------------------------------------------------------------------------------
    // On input this parameter determines what type of eigenvalue problem is solved
    // (mode 1 is standard, mode 2 is generalized and mode 3 is "shift-and-invert" EVP)
    //---------------------------------------------------------------------------------
    if ( get_transform_symmetric() )
        iparam[6] = 1;
    else 
        iparam[6] = 3;
    
    // iparam[7] number of shifts necessary in reverse communication
    
    // iparam[8], iparam[9], iparam[10] are number of OP*x operations, B*x operations and number of re-othogonalisation
    
    
    //=====================================================================
    // Pointer to mark the starting locations in the workd and workl arrays 
    // for matrices/vectors used by the Lanczos iteration (ouput)
    //=====================================================================
    int * ipntr = new int[11]; 
    
    //------------------------------
    //  working space
    //------------------------------
    real * workd     = new real[3*n];    // reverse communication
    
    const int lworkl = ncv*(ncv+8);      // must be at least of this size
    real * workl     = new real[lworkl]; // ouput/workspace
    
    //=====================================================================================
    // Input:  If info = 0 a randomly initial residual vector is used, if info != 0 resid 
    //         contains the initial residual vector, possibly from a previous run
    // Output: Error flag
    //=====================================================================================
    int info = 0; 
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Compute factorisation or preconditionier and initialise needed matrices
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    TICC( timer_transform );
    transform_problem ( K, M );
    TOCC( timer_transform, timer.transform );
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Start with ARPACK iteration via reverse communication interface
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    while (ido != 99)
    {
        
        TICC( timer_saupd );
        ARPACK::saupd< real >( ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, 
                                    iparam, ipntr, workd, workl, lworkl, info);
        TOCC( timer_saupd, timer.saupd );
        
        //------------------------------------------------------
        // Note that FORTRAN arrays start with index 1 and not 0
        //------------------------------------------------------
        TICC( timer_iterate );
        
        if ( ido == -1 )
        {
            real *in  = workd + ipntr[0] - 1;
            real *out = workd + ipntr[1] - 1;
            
            apply_OP( in, out, n );
        }// if
        else if ( ido == 1 )
        {
            real *in         = workd + ipntr[0] - 1;
            real *out        = workd + ipntr[1] - 1;
            real *in_partial = workd + ipntr[2] - 1;
            
            if ( get_transform_symmetric() )
                apply_OP( in, out, n );
            else
                apply_OP_partial( in_partial, out, n );
            
        }// else if
        else if ( ido == 2 )
        {
            real *in  = workd + ipntr[0] - 1;
            real *out = workd + ipntr[1] - 1;
        
            apply_B( in, out, n );
            
        }// else if
        
        TOCC( timer_iterate, timer.iterate );
    }// while
  
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Analyse the error flag of the ARPACK iteration
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
  
    _parameter_arpack.iparam_3   = iparam[2];
    _parameter_arpack.iparam_5   = iparam[4];
    _parameter_arpack.iparam_9   = iparam[8];
    _parameter_arpack.iparam_10  = iparam[9];
    _parameter_arpack.iparam_11  = iparam[10];
    _parameter_arpack.info_saupd = info;
    
    if ( info != 0 )
    {
        LOG( "" );
        LOG( "(TEigenArpack) comp_decomp :" );
        LOGHLINE;
        LOG( to_string("    error occured in ARPACK function saupd") );
        LOG( to_string("    info = %d",info) );
        LOGHLINE;
        
        if ( info < 0 )
            HERROR( ERR_CONSISTENCY, "(TEigenArpack) comp_decomp", "error occured in dsaupd" );
    }// if
        
        
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Postprocessing of previous ARPACK iteration (extract eigenvectors)
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
        
    //=====================================================================================
    // Adjust parameters of postprocessing
    //=====================================================================================
     
    // Specify if Ritz vectors should be computed 
    //(if 0 compute only Ritz values, if 1 compute also Ritz vectors)
    int rvec = 1;

    // Specify how many Ritz vectors are wanted ( "A" means "All", 
    // "S" means only these specified in the array selected) 
    char howmny[2] = "A"; 

    // The array specifies the Ritz vectors to be computed if "S" in howmny is selected 
    // if howmny is "A" this array is used as a workspace (input/workspace)
    int * select = new int[ncv];

    // Array containing the final Ritz values in ascending order
    // Note: these are the Ritz values associated to the matrix L^{-1} * M * L^{-T}
    real * d = new real[nev];
    
    // Ritz vectors are contained in the Arnoldi basis array computed via the routine 'dsaupd'
    
    // Shift (not referenced using mode=1,2)
    const real sigma = get_shift();
    
    
    //=====================================================================================
    // Apply postprocessing
    //=====================================================================================
    TICC( timer_seupd );
    
    ARPACK::seupd< real >( rvec, howmny, select, d, v, ldv, sigma, bmat, n, which, nev, tol, 
                          resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info );
                          
    TOCC( timer_seupd, timer.seupd );
                                
    //=====================================================================================
    // Analyse error flag of postprocessing
    //=====================================================================================

    _parameter_arpack.info_seupd = info;
    
    if ( info != 0 )
    {
        LOG( "" );
        LOG( "(TEigenArpack) comp_decomp :" );
        LOGHLINE;
        LOG( to_string("    error occured in ARPACK function seupd") );
        LOG( to_string("    info = %d",info) );
        LOGHLINE;
        
        HERROR( ERR_CONSISTENCY, "(TEigenArpack) comp_decomp", "error occured in dseupd" );
    }// if

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Copy Ritz values/vectors into output data with respect to transformation
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    TICC( timer_backtransform );

    //=====================================================================================
    // transform eigenvalues to original problem
    //=====================================================================================
    D->set_size( size_t(nev), size_t(nev) );
    D->set_ofs ( _M->col_ofs(), _M->col_ofs() );
    D->scale( real(0) );
    
    for ( int j = 0; j < nev; j++ )
    {
        real lambda_general;
         
        if ( get_transform_symmetric() )
        {
            lambda_general = real(1)/d[j] + _parameter_EVP_transformation.shift;
            
            D->set_entry( nev-1-j, nev-1-j, lambda_general );
        }// if 
        else
        {
            lambda_general = d[j];
            
            D->set_entry( j, j, lambda_general );
        }// else
    }// for
    
    
    //=====================================================================================
    // transform eigenvectors to original problem
    //=====================================================================================
    Z->set_size( size_t(n) , size_t(nev) );
    Z->set_ofs ( _M->col_ofs(), _M->col_ofs() );
    
    if ( get_transform_symmetric() )
    {
        //-----------------------------------------
        // If symmetric transformation was used
        //-----------------------------------------
        auto reorder_columns = 
        [ v, Z, n, nev ] ( const tbb::blocked_range< uint > & r )
        {        
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                for ( int i = 0; i < n; i++ )
                {
                    const real entry = v[ j*n + i ];
                
                    Z->set_entry( i, nev-1-j, entry );
                }// for
            }// for
        };
            
        bool  do_parallel = true;

        if ( CFG::nthreads() == 1 || n < 1000 )
            do_parallel = false;

        if ( do_parallel ) 
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(nev) ), reorder_columns );
        else
            reorder_columns ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
            
        backtransform_eigenvectors_sym ( _L, Z );    
    }// if
    else
    {
        //-----------------------------------------
        // If non-symmetric transformation was used
        //-----------------------------------------
        auto reorder_columns = 
        [ v, Z, n, nev ] ( const tbb::blocked_range< uint > & r )
        {        
            for ( auto  j = r.begin(); j != r.end(); ++j )
            {
                for ( int i = 0; i < n; i++ )
                {
                    const real entry = v[ j*n + i ];
                
                    Z->set_entry( i, j, entry );    
                }// for
            }// for
        };
            
        bool  do_parallel = true;

        if ( CFG::nthreads() == 1 || n < 1000 )
            do_parallel = false;

        if ( do_parallel ) 
            tbb::parallel_for ( tbb::blocked_range< uint >( uint(0), uint(nev) ), reorder_columns );
        else
            reorder_columns ( tbb::blocked_range< uint >( uint(0), uint(nev) ) );
    }// else
        
    TOCC( timer_backtransform, timer.backtransform );
    TOCC( timer_all, timer.all );
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Output
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    if ( _parameter_base.print_info )
    {
        print_options();
        print_summary( D );
        
        #if DO_TEST >= 1
        timer.print_performance();
        #endif
    }// if
    

    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Clean up memory
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    delete [] resid;
    delete [] v;
    delete [] iparam;
    delete [] ipntr;
    delete [] workd;
    delete [] workl;
    delete [] select;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Test results if wanted
    //
    ///////////////////////////////////////////////////////////////////////////////////////    
    if ( _parameter_base.test_residuals )
    {
        TEigenAnalysis  eigen_analysis;
            
        eigen_analysis.set_verbosity( 0 );
        
        const bool large_errors_detected = eigen_analysis.analyse_vector_residual ( K, M, D, Z );
        
        if ( large_errors_detected )
        {
//             eigen_analysis.set_verbosity( 3 );
//         
//             eigen_analysis.analyse_vector_residual ( K, M, D, Z );            
        
            HERROR( ERR_NCONVERGED, "(TEigenArpack) comp_decomp", "large errors detected" );
        }// if
    }// if
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // return number of computed eigenpairs
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    return nev;
}




}// namespace HAMLS
