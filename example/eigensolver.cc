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
// File        : eigensolver.cc
// Description : example for H-AMLS methopd applied to an elliptic PDE eigenvalue problem
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include <iostream>
#include <hlib.hh>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "hamls/TEigenArnoldi.hh"
#include "hamls/TEigenArpack.hh"
#include "hamls/THAMLS.hh"

using namespace std;
using namespace HLIB;
using namespace boost::filesystem;
using namespace boost::program_options;
using HLIB::uint;
using boost::format;

using std::cout;
using std::endl;
using std::unique_ptr;
using std::string;

using  real_t    = HLIB::real;
using  complex_t = HLIB::complex;

using namespace HAMLS;

namespace
{


template < typename T >
string
mem_per_dof ( T && A )
{
    const size_t  mem  = A->byte_size();
    const size_t  pdof = size_t( double(mem) / double(A->rows()) );

    return Mem::to_string( mem ) + " (" + Mem::to_string( pdof ) + "/dof)";
}


//
// Print some general informations of the given sparse matrix 
//
TSparseMatrix *
import_matrix ( const std::string fmatrix )
{
    cout<<endl;
    cout<<"  importing sparse matrix from file \'"<<fmatrix<<"\'"<<endl;
        
    THLibMatrixIO   matrix_io;
       
    unique_ptr< TMatrix >   K( matrix_io.read( fmatrix ) );
    
    if ( K.get() == nullptr )
    {
        cout<<endl<<"  warning: no matrix found" << endl;
        exit( 1 );
    }// if

    if ( ! IS_TYPE( K.get(), TSparseMatrix ) )
    {
        // try to convert to sparse
        cout<<endl<<"  warning: matrix is of type "<<K->typestr()<<" --> converting to sparse format";
        
        auto  T = to_sparse( K.get() );

        if ( T.get() != nullptr )
            K = std::move( T );
        else
            exit( 1 );
    }// if

    TSparseMatrix * S = ptrcast( K.release(), TSparseMatrix );
    
    if ( S->rows() != S->cols() )
    {
        cout<<endl<<"  warning: matrix not quadratic" << endl;
        exit( 1 );
    }// if
    
    if ( S->test_symmetry() )
        S->set_form( symmetric );
    
    cout << "  matrix has dimension " << S->rows() << " x " << S->cols() << endl;
    cout << "    no of non-zeroes    = " << S->n_non_zero() << " ("
            << format( "%.2f" ) % ( 1000.0 * double(S->n_non_zero()) / Math::square(double(S->rows())) ) << "‰"
            << ", avg=" << S->avg_entries_per_row()
            << ", max=" << S->max_entries_per_row() 
            << ")" << endl;
    cout << "    matrix is             " << ( S->is_complex() ? "complex" : "real" ) << " valued" << endl;
    cout << "    format              = ";
    if      ( S->is_unsymmetric() ) cout << "unsymmetric" << endl;
    else if ( S->is_symmetric()   ) cout << "symmetric" << endl;
    else if ( S->is_hermitian()   ) cout << "hermitian" << endl;
    cout << "  size of sparse matrix = " << mem_per_dof( S ) << endl;
    cout << "  |S|_F                 = " << format( "%.6e" ) % norm_F( S ) << endl;
    
    return S;
}
 




}// namespace anonymous





//
//
// main function
//
//
int
main ( int argc, char ** argv )
{
    //=====================================================================
    // program options
    //=====================================================================
    
    //---------------------------------------------------------
    //       general options
    //---------------------------------------------------------
    int     input_hlib_verbosity = 1;
    real_t  input_eta            = real_t(50);
    uint    input_nmin           = 0;
    size_t  input_n_ev          = 5;
    int     input_nthreads      = 0;
    string  input_data_path     = "/home/user/";
    bool    input_print_solver_info = false;
    //---------------------------------------------------------
    //       basic HAMLS options
    //---------------------------------------------------------    
    real_t  input_rel_accuracy_transform = real_t(1e-8);
    uint    input_max_dof_subdomain      = 1000;
    //---------------------------------------------------------
    //       options for mode selection options applied in HAMLS
    //---------------------------------------------------------
    real_t   input_c_subdomain   = real_t(1.5);
    real_t   input_c_interface   = real_t(1.0);
    real_t   input_c_condense    = real_t(6.0);
    real_t   input_e_subdomain   = real_t(1)/real_t(3.0);
    real_t   input_e_interface   = real_t(1)/real_t(2.0);    
    //---------------------------------------------------------
    //       options for classical iterativ eigensolver
    //---------------------------------------------------------
    real_t   input_shift              = real_t(0);
    
    //=====================================================================
    // define command line options
    //=====================================================================
    
    // define command line options
    options_description             all_opts;
    options_description             vis_opts( "usage: eigensolver [options] data_path\n  where options include" );
    options_description             general_opts( "General options", 0 );
    options_description             HAMLS_opts( "Basic options of H-AMLS", 0 );
    options_description             modalTrunc_opts( "Options for the modal trunction applied in H-AMLS", 0 );
    options_description             arnoldi_opts( "Options for the classical iterativ eigensolver", 0 );
    options_description             hid_opts( "Hidden options", 0 );
    positional_options_description  pos_opts;
    variables_map                   vm;

    // general options
    general_opts.add_options()
        ( "help,h",                             ": print this help text" )
        ( "hlib_verbosity",    value<int>(),    ": HLIBpro verbosity level" )
        ( "eta",               value<real_t>(), ": set eta for admissible condition" )
        ( "nmin",              value<uint>(),   ": set minimal cluster size" )
        ( "n_ev",              value<size_t>(), ": set the number of sought eigenpairs" )
        ( "threads",           value<int>(),    ": number of threads used for parallel computation (if set to 0 all available threads are used)" )
        ( "print_solver_info",                  ": print basic information regarding eigensolver execution to logfile" );
    // basic options of HAMLS 
    HAMLS_opts.add_options()
        ( "rel_accuracy_transform", value<real_t>(), ": relative accurcy of H-matrix arithmetic used for the problem transformation in H-AMLS" )
        ( "max_dof_subdomain",  value<uint>(),       ": maximal degrees of freedom used for the subdomain problems applied in the matrix partitioning" );
    // options for mode selection applied in HAMLS
    modalTrunc_opts.add_options()
        ( "c_subdomain",  value<real_t>(),      ": select k_i smallest eigenpairs of subdomain problem with size N_i where k_i := c_subdomain (N_i)^(e_subdomain)" )
        ( "e_subdomain",value<real_t>(),        ": select k_i smallest eigenpairs of subdomain problem with size N_i where k_i := c_subdomain (N_i)^(e_subdomain)" )
        ( "c_interface",  value<real_t>(),      ": select k_i smallest eigenpairs of interface problem with size N_i where k_i := c_interface (N_i)^(e_interface)" )
        ( "e_interface",value<real_t>(),        ": select k_i smallest eigenpairs of interface problem with size N_i where k_i := c_interface (N_i)^(e_interface)" )
        ( "c_condense",   value<real_t>(),      ": select k_i smallest eigenpairs of condensed subdomain problem with size N_i where k_i := c_condense (N_i)^(e_subdomain)" );
    // parameter for classical iterativ eigensolver
    arnoldi_opts.add_options()
        ( "shift",             value<real_t>(), ": set shift displacement in get_shift" );
        
    hid_opts.add_options()
        ( "data_path",      value<string>(), ": path to the folder that must contain the following input files of this program\n"
                                             "      file 'K' contains the data for the sparse matrix K\n"
                                             "      file 'M' contains the data for the sparse matrix M\n"
                                             "      file 'coord' contains the data for the associated coordinate information of the problem\n");

    // options for command line parsing
    vis_opts.add(general_opts).add(HAMLS_opts).add(modalTrunc_opts).add(arnoldi_opts);
    all_opts.add( vis_opts ).add( hid_opts );
    
    // all "non-option" arguments should be "--data_path" arguments
    pos_opts.add( "data_path", -1 );


    //=====================================================================
    // parse command line options
    //=====================================================================
    try
    {
        store( command_line_parser( argc, argv ).options( all_opts ).positional( pos_opts ).run(), vm );
        notify( vm );
    }// try
    catch ( unknown_option &  e )
    {
        cout << e.what() << ", try \"-h\"" << endl;
        exit( 1 );
    }// catch

    //=====================================================================
    // eval command line options
    //=====================================================================
    if ( vm.count( "help") )
    {
        cout << vis_opts << endl;
        cout << "usage: eigensolver [options] data_path "<< endl;
        cout << "  where \'data_path\' is the path to the folder that must contain the following input files for this program (use corresponding HLIBpro format for each file):\n"
                "      file 'K' contains the data for the sparse matrix K\n"
                "      file 'M' contains the data for the sparse matrix M\n"
                "      file 'coord' contains the data for the associated coordinate information of the problem\n";
        
        exit( 1 );
    }// if
    
    if ( vm.count( "data_path" ) )
        input_data_path = vm["data_path"].as<string>();
    else
    {
        cout << "usage: eigensolver [options] data_path "<< endl;
        exit( 1 );
    }// if
    
    //---------------------------------------------------------
    //       general parameters
    //---------------------------------------------------------
    if ( vm.count( "hlib_verbosity" ) ) input_hlib_verbosity = vm["hlib_verbosity"].as<int>();
    if ( vm.count( "eta"            ) ) input_eta      = vm["eta"].as<real_t>(); 
    if ( vm.count( "nmin"           ) ) input_nmin     = vm["nmin"].as<uint>();
    if ( vm.count( "n_ev"           ) ) input_n_ev     = vm["n_ev"].as<size_t>(); 
    if ( vm.count( "threads"        ) ) input_nthreads = vm["threads"].as<int>();
    if ( vm.count( "print_solver_info" ) ) input_print_solver_info = true;
    //---------------------------------------------------------
    //       basic HAMLS parameters
    //---------------------------------------------------------
    if ( vm.count( "rel_accuracy_transform" ) ) input_rel_accuracy_transform = vm["rel_accuracy_transform"].as<real_t>();
    if ( vm.count( "max_dof_subdomain"      ) ) input_max_dof_subdomain      = vm["max_dof_subdomain"].as<uint>();
    //---------------------------------------------------------
    //       mode selection parameters of HAMLS
    //---------------------------------------------------------
    if ( vm.count( "c_subdomain"  ) ) input_c_subdomain   = vm["c_subdomain"].as<real_t>();
    if ( vm.count( "c_interface"  ) ) input_c_interface   = vm["c_interface"].as<real_t>();
    if ( vm.count( "c_condense"   ) ) input_c_condense    = vm["c_condense"].as<real_t>();
    if ( vm.count( "e_subdomain") ) input_e_subdomain = vm["e_subdomain"].as<real_t>();
    if ( vm.count( "e_interface") ) input_e_interface = vm["e_interface"].as<real_t>();
    //---------------------------------------------------------
    //       parameter for classical iterativ eigensolver
    //---------------------------------------------------------
    if ( vm.count( "shift"      ) ) input_shift = vm["shift"].as<real_t>();
    
     
    try
    {
        //----------------------------------------------------------------------------
        // init HLIBpro and set general HLIBpro options
        //----------------------------------------------------------------------------
                
        INIT();
        
        if ( input_nthreads != 0 )
            CFG::set_nthreads( input_nthreads );
        
        cout<<endl<<"# "<<CFG::nthreads()<<" threads are used for parallel computation"<<endl;
        
        if ( CFG::nthreads() > 1 )
            CFG::Arith::use_dag = true;
        else
            CFG::Arith::use_dag = false;

        CFG::set_verbosity( input_hlib_verbosity );
        
        //////////////////////////////////////////////////////////////////////////////
        //
        // Load sparse matrices K and M, and the associated coordinate information
        //
        //////////////////////////////////////////////////////////////////////////////
        cout<<endl<<"# Load Sparse Matrix K";
        unique_ptr< TSparseMatrix >  K( import_matrix( input_data_path + "input_K" ) );
        
        cout<<endl<<"# Load Sparse Matrix M";
        unique_ptr< TSparseMatrix >  M( import_matrix( input_data_path + "input_M" ) );
        
        cout<<endl<<"# Load Coordinate Data From File \'"<<input_data_path + "input_coord"<<"\'"<<endl;
        TAutoCoordIO    coord_io;
        unique_ptr< TCoordinate >  coord( coord_io.read ( input_data_path + "input_coord") );
        
            
        //////////////////////////////////////////////////////////////////////////////
        //
        // Construct H-matrix representations K_h and M_h of the sparse matrices K and M 
        //
        //////////////////////////////////////////////////////////////////////////////

        //----------------------------------------------------------------------------
        // 1) Apply nested dissection for the creation of the cluster tree
        //----------------------------------------------------------------------------
        uint  nmin;
    
        if ( input_nmin == 0 )
            nmin = max( size_t(40), min( M->avg_entries_per_row(), size_t(100) ) );
        else
            nmin = uint(input_nmin);
        
        // partitioning of clusters is based on an adaptive bisection
        TGeomBSPPartStrat  part_strat( adaptive_split_axis );
        
        //NOTE: Choose this special sparse matrix for the construction of cluster tree
        //that has everywhere there non-zero entries where K or M has a non-zero entry,
        //i.e., choose for example |K|+|M|
        TBSPNDCTBuilder  ct_builder( M.get(), & part_strat, nmin );
        
        unique_ptr< TClusterTree > ct( ct_builder.build( coord.get() ) );   

        //----------------------------------------------------------------------------
        // 2) Create corresponding block cluster tree
        //----------------------------------------------------------------------------  
        TStdGeomAdmCond  adm_cond( input_eta );
        
        TBCBuilder                       bct_builder;
        unique_ptr< TBlockClusterTree >  bct( bct_builder.build( ct.get(), ct.get(), & adm_cond ) );
        
        //----------------------------------------------------------------------------
        // 3) Construct corresponding H-matrix representations
        //----------------------------------------------------------------------------  
        TSparseMBuilder  h_builder_K( K.get(), ct->perm_i2e(), ct->perm_e2i() ); 
        TSparseMBuilder  h_builder_M( M.get(), ct->perm_i2e(), ct->perm_e2i() );
        
        TTruncAcc  acc_exact( real_t(0) );
        
        unique_ptr< TMatrix >  K_h( h_builder_K.build( bct.get(), K->form(), acc_exact ) );
        unique_ptr< TMatrix >  M_h( h_builder_M.build( bct.get(), M->form(), acc_exact ) );

  
        ////////////////////////////////////////////////////////////////////////////////////////
        //
        // Compute the eigendecomposition (D,Z) of the eigenvalue problem (K,M), i.e.,
        // solve the problem K*Z = M*Z*D where D is a diagonal matrix containing the 
        // eigenvalues and Z is the matrix containing the associated eigenvectors.
        //
        ////////////////////////////////////////////////////////////////////////////////////////

        //--------------------------------------------------------------------------------------
        // Solve eigenvalue problem using the ARPACK eigensolver
        //
        // - this solver is fast and very robust 
        // - this solver computes numerically exact eigenpairs
        // - this solver is based on the well-proven and well-tested ARPACK library
        // - this solver is recommended only when a small portion of the eigenpairs is sought
        // - this solver supports shifts
        // - this solver can be performed with and without a symmentric problem transformation
        // - this solver has been parallelized as well, however, only one problem can be solved 
        //   by this solver concurrently, since the ARPACK library is not threadsafe
        //--------------------------------------------------------------------------------------
        cout<<endl<<"# Solve Eigenvalue Problem (K,M) via ARPACK and H-Matrix Arithmetic"<<endl;
        // initialise the output data
        unique_ptr< TDenseMatrix >  D_arpack ( new TDenseMatrix() );
        unique_ptr< TDenseMatrix >  Z_arpack ( new TDenseMatrix() );
        
        // adjust the ARPACK eigensolver 
        TEigenArpack arpack_solver;    
        arpack_solver.set_shift         ( input_shift );
        arpack_solver.set_print_info    ( input_print_solver_info);
        arpack_solver.set_n_ev_searched ( input_n_ev  );
            
        // solve the problem
        arpack_solver.comp_decomp( K_h.get(), M_h.get(), D_arpack.get(), Z_arpack.get() );
        
        // analyse the computed eigensolution
        TEigenAnalysis  eigen_analysis;
        eigen_analysis.set_verbosity( 4 );
        eigen_analysis.analyse_vector_residual ( K.get(), M.get(), D_arpack.get(), Z_arpack.get(), ct.get() );
        
        //--------------------------------------------------------------------------------------
        // Solve eigenvalue problem using the Arnoldi eigensolver
        //
        // - this solver is fast  
        // - this solver computes numerically exact eigenpairs
        // - this solver is a basic implementation of the Arnoldi solver and uses a symmetric 
        //   problem transformation, i.e., the generalized eigenvalue problem (K,M) is transformed 
        //   to a standard eigenvaluue 
        // - this solver is recommended only when a small portion of the eigenpairs is sought
        // - this solver supports shifts
        // - this solver has been parallelized several problem can be solved by this solver concurrently
        //--------------------------------------------------------------------------------------
        cout<<endl<<"# Solve Eigenvalue Problem (K,M) via Arnoli Method and H-Matrix Arithmetic"<<endl;
        // initialise the output data
        unique_ptr< TDenseMatrix >  D_arnoldi ( new TDenseMatrix() );
        unique_ptr< TDenseMatrix >  Z_arnoldi ( new TDenseMatrix() );
        
        // adjust the Arnoldi eigensolver
        TEigenArnoldi arnoldi_solver;
        arnoldi_solver.set_shift              ( input_shift );
        arnoldi_solver.set_print_info         ( input_print_solver_info);
        arnoldi_solver.set_n_ev_searched      ( input_n_ev  );
        
        // solve the problem
        arnoldi_solver.comp_decomp( K_h.get(), M_h.get(), D_arnoldi.get(), Z_arnoldi.get() );
        
        // analyse the computed eigensolution
        eigen_analysis.analyse_vector_residual ( K.get(), M.get(), D_arnoldi.get(), Z_arnoldi.get(), ct.get() );
        
        //--------------------------------------------------------------------------------------
        // Solve eigenvalue problem using the H-AMLS method
        //
        // - this solver is very fast especially when many eigenpairs are sought
        // - this solver computes only approximative eigenpairs, however, the approximation 
        //   error can be controlled such that the error is of the order of the discretisation error
        // - this solver is highly recommended when a large portion of the eigenpairs is sought
        //   and when numerical exact eigenpairs are not needed, e.g., when one is actually  
        //   interested in the eigensolutions of an underlying continuous problem
        //--------------------------------------------------------------------------------------
        cout<<endl<<"# Solve Eigenvalue Problem (K,M) via H-AMLS Method"<<endl;
        // initialise the output data
        unique_ptr< TDenseMatrix >  D_hamls   ( new TDenseMatrix() );
        unique_ptr< TDenseMatrix >  Z_hamls   ( new TDenseMatrix() );
        
        // adjust basic options of the H-AMLS eigensolver
        THAMLS hamls_solver;
        hamls_solver.set_n_ev_searched    ( input_n_ev );
        hamls_solver.set_max_dof_subdomain( input_max_dof_subdomain );
        hamls_solver.set_do_improving    ( true   );
        hamls_solver.set_do_condensing  ( true   );
        hamls_solver.set_print_info       ( input_print_solver_info );
        
        // set the accurcy of the H-matrix arithmetic used in H-AMLS
        TTruncAcc acc_transform( input_rel_accuracy_transform, real_t(0) );
        hamls_solver.set_acc_transform_K( acc_transform );
        hamls_solver.set_acc_transform_M( acc_transform );
        
        // set options for mode selection applied in HAMLS 
        hamls_solver.set_factor_subdomain  ( input_c_subdomain  );
        hamls_solver.set_factor_interface  ( input_c_interface  );
        hamls_solver.set_factor_condense   ( input_c_condense  );
        hamls_solver.set_exponent_interface( input_e_interface );
        hamls_solver.set_exponent_subdomain( input_e_subdomain );

        // solve the problem
        hamls_solver.comp_decomp ( ct.get(), K_h.get(), M_h.get(), D_hamls.get(), Z_hamls.get(), K.get(), M.get() );

        // analyse the computed eigensolution
        eigen_analysis.analyse_vector_residual ( K.get(), M.get(), D_hamls.get(), Z_hamls.get(), ct.get() );
        
        DONE();
    }// try
    catch ( Error & e )
    {
        cout << e.to_string() << endl;
    }// catch
    catch ( std::exception & e )
    {
        std::cout << e.what() << std::endl;
    }// catch
    
    return 0;
}
