#ifndef __HAMLS_THAMLS_HH
#define __HAMLS_THAMLS_HH

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
// File        : THAMLS.hh
// Description : class for the HAMLS algortihm
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.


#include "cluster/TCluster.hh"

#include "matrix/TMatrix.hh"
#include "matrix/TDenseMatrix.hh"
#include "matrix/TBlockMatrix.hh"

#include "hamls/TEigenLapack.hh"
#include "hamls/TSeparatorTree.hh"

namespace HAMLS
{

using namespace HLIB;

////////////////////////////////////////////////////////////////////////////////////////////
//
// Mode selection strategies in THAMLS
//
////////////////////////////////////////////////////////////////////////////////////////////
enum mode_selection_t
{ 
    MODE_SEL_REL_H   = 1,  // select a relative amount of eigenvalues in all subproblems
    MODE_SEL_ABS_H   = 2,  // select an absolut amount of eigenvalues in all subproblems           
    MODE_SEL_BOUND_H = 3,  // select all eigenvalues until a given bound in all subproblems
    MODE_SEL_AUTO_H  = 4,  // select eigenvalues depending on the DOF the subproblem
};  


//!
//! \ingroup  HAMLS_Module
//! \class    THAMLS
//! \brief    class for the AMLS algortihm using H-Algebra
//!
class THAMLS
{

    
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The eigensolver THAMLS solves the generalized eigenvalue problem K*Z = M*Z*D with (Z^T)*M*Z = Id
// approximately using a recursive approach of the so-called "Automated Multi-Level Substructuring" combined
// with the hierarchical matrices. The matrices K and M have to be symmetric, and M has to be positive
// definite. D is a diagonal matrix containing the approximated eigenvalues and the matrix Z is 
// containing the corresponding eigenvector approximations. 
//
//     
// For more details check the HLIBpro documentation and see the following publications: 
//     
// P. Gerds: Solving an Elliptic PDE Eigenvalue Problem via Automated Multi-Level Substructuring and  
// Hierarchical Matrices, Ph.D. thesis, RWTH Aachen University (2017), doi:10.18154/RWTH-2017-10520
// 
// P. Gerds, L. Grasedyck: Solving an Elliptic PDE Eigenvalue Problem via Automated Multi-Level Substructuring 
// and Hierarchical Matrices, Computing and Visualization in Science: Volume 16, Issue 6 (2015), Page 283-302, 
// doi:10.1007/s00791-015-0239-x 
// 
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////// 

    
private:

    //!
    //! \struct  parameter_new_t
    //! \brief   Datatype to summarize parameter for mode selection
    //!
    struct parameter_base_t
    {      
        ///////////////////////////////////////////////////////////
        //
        // basic parameter for THAMLS
        //
        ///////////////////////////////////////////////////////////         
                
        //! maximal DOFs of the subdomain subproblems
        size_t  max_dof_subdomain;
        
        //! if true then print summarised information of program execution to the logfile
        bool  print_info;
        
        //! number of searched eigenpair approximations (if set to zero, than all eigenvalues of the reduced problem will be computed)
        size_t  n_ev_searched;
        
        //! if true several consistency tests are applied
        bool    do_debug;
        
        //! if true the Arnoldi method (own impelementation) is used for the solution of large sized interface EVPs
        //! if false the ARPACK solver (if available) is used for the solution of large sized interface EVPs 
        //! (Note that since ARPACK library is not threadsafe large sized interface eigenvalue problems cannot be 
        //! solved in parallel when ARPACK is used)
        bool    use_arnoldi;
                
        //! if true "corsening" is used when the matrix K is factorised for the computation of K_tilde
        bool    coarsen;
                
        //! if true apply condensing, i.e., the recursive H-AMLS is applied
        bool    do_condensing;
        
        //! if true Rayleigh quotients of the computed eigenvector approximaitons are used as eigenvalue approximations
        bool    comp_rayleigh_quotients;
                
        //! if true then the stable dot product is used when the Rayleigh quotient (x^T*K*x) / (x^T*M*x) is computed
        //! (Benchmark for the Laplace eigenvalue problem on [0,1]^3 showed that using this feature does NOT improve 
        //! the computed eigenvalue approximations, i.e., no difference could be observed in the approximation quality.
        //! Furthermore, Rayleigh quotients are only applied when the feature 'do_improving' is deactivated)
        bool    stable_rayleigh;
        
        ///////////////////////////////////////////////////////////
        // 
        // input/output parameter
        //
        ///////////////////////////////////////////////////////////
        
        //! location where the matrices should be loaded/saved
        std::string     io_location;
        
        //! prefix of the several input/output files
        std::string     io_prefix;
        
        //! if true the matrices of the transformed problem K_tilde, M_tilde and L are saved to/
        //! loaded from the folder 'io_location' where each matrix-file gets the prefix 'io_prefix'
        bool            save_problem;
        bool            load_problem;
        
        ///////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_base_t ( )
        {
            //----------------------------------
            // set the default general parameter
            //----------------------------------
            max_dof_subdomain  = 1000;   
            print_info         = false;
            n_ev_searched      = 40;
            comp_rayleigh_quotients = true;
            stable_rayleigh    = false;  
            use_arnoldi        = true;
            do_debug           = false;
            do_condensing      = true;
            coarsen            = false;
            
            //---------------------------------------
            // set the default input/output parameter
            //---------------------------------------
            load_problem   = false;
            save_problem   = false;
            io_prefix      = "";
            io_location    = "/home/user/output/";
        }
    };
    
    
    //!
    //! \struct  parameter_mode_selection_t
    //! \brief   Datatype to summarize parameter for mode selection
    //!
    struct parameter_mode_selection_t
    {      
        ///////////////////////////////////////////////////////////
        //
        // basic parameter for mode selection
        //
        ///////////////////////////////////////////////////////////
        
        //! the type of mode selection strategy that is applied for the subproblems
        mode_selection_t  mode_selection;
        
        //! lower bound of the modes which will be truncated in the subproblems
        real   trunc_bound;
        
        //! relative amount of the modes which should be selected in the subproblems
        real   rel;
        
        //! absolut amount of modes which should be selected in the subproblems 
        size_t abs;
        
        //! minimal amount of modes which should be selected in the subproblems 
        size_t nmin_ev;
        
        ///////////////////////////////////////////////////////////
        //
        // parameter automatic mode selection
        //
        ///////////////////////////////////////////////////////////
        
        //! ====================================================================================
        //!
        //! If "automatic mode selection" is chosen the arbitrarely values "factor" and 
        //! "exponent" determine the used spectral information of the subproblems. The 
        //! number of used eigenpairs is "factor * DOF^{exponent}"
        //!
        //! ====================================================================================
        
        //! factor and exponent for subdomain subproblems
        real  factor_subdomain;
        real  exponent_subdomain;
        
        //! factor and exponent for interface suproblems
        real  factor_interface;
        real  exponent_interface;
        
        ///////////////////////////////////////////////////////////
        //
        // parameter for recursive H-AMLS
        //
        ///////////////////////////////////////////////////////////
        
        //! ====================================================================================
        //!
        //! If "automatic mode selection" and "condensing" is activated the following  
        //! values are used to determine the used spectral information of large subdomain problems
        //! which are planed to be condensed.
        //!
        //! ====================================================================================
        
        
        //!  when the ratio between reduced DOFs before condensing and after condensing exceeds this value
        // then the corresponding subproblems are condensed.
        real    ratio_2_condense;
        
        //! this value works the same as "factor_subdomain" but is used for problems that get condensed. 
        //! this value should be higher than "factor_subdomain"
        real    factor_condense;
        
        //! minimal number of reduced DOFs of subproblems that can still be condensed together
        size_t  condensing_size;
        
        //! counting the number of condensed problems
        size_t  n_condensed_problems;

        ///////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_mode_selection_t ( )
        {            
            //-----------------------------------------
            // set default parameter for mode selection
            //-----------------------------------------
            mode_selection = MODE_SEL_BOUND_H;
            trunc_bound    = real(0);
            rel            = real(0);
            abs            = 0; 
            nmin_ev        = 0; 
            
            //---------------------------------------------------
            // set default parameter for automatic mode selection
            //---------------------------------------------------
            factor_subdomain     = real(1.5); 
            factor_interface     = real(1.0); 
            exponent_subdomain   = real(1.0)/real(3.0);
            exponent_interface   = real(1.0)/real(2.0);
            
            //-------------------------------------------------------------
            // set default parameter for mode selection in recursive H-AMLS
            //-------------------------------------------------------------
            condensing_size      = 500;     
            ratio_2_condense  = real(2); 
            factor_condense      = real(6);
            n_condensed_problems = 0;
        }
    };
    
    //!
    //! \struct  parameter_impro_t
    //! \brief   Datatype to summarize parameter for subsequent subspace iteration which improves eigenpair approximations
    //!
    struct parameter_impro_t
    {      
        /////////////////////////////////////////////////////////
        //
        // parameter for subseqent subspace iteration
        //
        /////////////////////////////////////////////////////////
        
        //! if true apply a subsequent subspace iteration to improve the eigenpairs obtained by H-AMLS
        bool  do_improving;
        
        //! if true the sparse version of the subspace iteration is used
        bool  use_sparse_version; 
        
        //! number of applied iterations
        size_t  number_of_iterations;
        
        //! if true then during the subspace iteration the representation of K^{-1} is applied as an exact precondtioner
        //! (when true then possibly a little bit better eigenpair approximations are obtained in the end but more expensive)
        bool  use_K_inv_as_precomditioner;
        
        //! pointer to the factorised matriix of K and the corresponding preconditionier 
        //! (only used when 'use_K_inv_as_precomditioner' is true)
        TMatrix *          K_factorised;
        TLinearOperator *  K_preconditioner;
        
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_impro_t ( )
        {
            //--------------------------------------------------
            // set default parameter for subspace iteration
            //--------------------------------------------------
            do_improving                = true;
            use_sparse_version          = true;
            use_K_inv_as_precomditioner = false;
            number_of_iterations        = 1;
            K_factorised                = nullptr;
            K_preconditioner            = nullptr;
        }

    };
    
    //!
    //! \struct  parameter_parallel_t
    //! \brief   Datatype to summarize parameter for parallelisation
    //!
    struct parameter_parallel_t
    {      
        /////////////////////////////////////////////////////////
        //
        // parameter for parallelisation
        //
        /////////////////////////////////////////////////////////
        
        //! perform miscellaneous computations in parallel
        bool miscellaneous_parallel; 
        
        //! condense different THAMLS subproblems in parallel
        bool condense_subproblems_in_parallel;
        
        //! compute the reduced matrix K_red and M_red in parallel
        bool K_and_M_red_in_parallel;
        
        //! defines how often in the code the affinity partitioner is used for TBB's parallel for-loops
        //! (0=no usage, 1=in auxilliary routines, 2=also in higher-level routines) 
        uint level_of_affinity_partitioner_usage;
        
        //! compute the matrix "Temp = M_tilde_ij * S_j" in parallel using following auxilliary variables
        bool   M_red_ij_step_1_in_parallel;
        size_t step_1___max_row_size;
        size_t step_1___block_partition_size;
        size_t mul_block_with_dense_max_size;
        
        //! compute the matrix "M_red_ij = S_i^{T} * Temp" in parallel using following auxilliary variables
        bool   M_red_ij_step_2_in_parallel;
        size_t step_2___max_row_size;
        size_t step_2___block_partition_size;
        
        
        //! maximal size of eigenvalue problem where we still use sequential MKL routines for the LAPACK eigensolver
        size_t max_size_EVP_for_seqential_MKL;
        
        //! parameter for the parallel computation of the transformed eigenvectors
        size_t transform_with_S_i___block_partition_size;
        size_t transform_with_L___block_partition_size;
        bool   transform_with_L___partition_by_threads;
      
        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        parameter_parallel_t ( )
        {
            //--------------------------------------------------
            // set default parameter for general parallelisation
            //--------------------------------------------------
            miscellaneous_parallel           = true;
            K_and_M_red_in_parallel          = true;
            condense_subproblems_in_parallel = true;
            
            //--------------------------------------------------
            // set default parameter for advanced parallelisation
            //--------------------------------------------------
            M_red_ij_step_1_in_parallel   = true;   
            step_1___max_row_size         = 15000;  
            step_1___block_partition_size = 150;    
            mul_block_with_dense_max_size = 20000;  
            //---------------------------------------------------------
            M_red_ij_step_2_in_parallel   = true;   
            step_2___max_row_size         = 15000;  
            step_2___block_partition_size = 150;    
            //---------------------------------------------------------
            max_size_EVP_for_seqential_MKL  = 4000; 
            //---------------------------------------------------------
            transform_with_S_i___block_partition_size = 4000;
            transform_with_L___block_partition_size   = 20;  
            transform_with_L___partition_by_threads   = true;
            //---------------------------------------------------------
            level_of_affinity_partitioner_usage       = 0;   
            //---------------------------------------------------------
        }
        
        /////////////////////////////////////////////////////////
        //
        // misc functions
        //
        void set_parallel_options ( const bool  active ) 
        {
            miscellaneous_parallel           = active;
            condense_subproblems_in_parallel = active;
            K_and_M_red_in_parallel          = active;
            M_red_ij_step_1_in_parallel      = active;
            M_red_ij_step_2_in_parallel      = active;
        }
    };
    
    
    //!
    //! \struct  accuracy_t
    //! \brief   Datatype to summarize the different 'Truncation Accuracies' used in THAMLS
    //!
    struct trunc_acc_t
    {   
        /////////////////////////////////////////////////////////
        //
        // truncation accuracies used in THAMLS
        //
        /////////////////////////////////////////////////////////
    
        //! accuracy used to factorize the matrix K = L * K_tilde * L^{T}
        TTruncAcc  transform_K;
                 
        //! accuracy used for computing the matrix M_tilde = L^{-1} * M * L^{-T}
        TTruncAcc  transform_M;

        /////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        trunc_acc_t ( )
        {
            //-------------------------------------------
            // set the default parameters of the accuracy
            //-------------------------------------------
            const real default_eps = real(1e-8);
            
            transform_K  = TTruncAcc( default_eps, real(0) );
            transform_M  = TTruncAcc( default_eps, real(0) );
        }
    };
    
              
    //!
    //! \struct  subprob_data_t
    //! \brief   Auxilliary datatype indicating a set of subproblems which shall be condensed
    //!
    struct subprob_data_t
    {               
        ////////////////////////////////////////////////////////////////////
        //
        // parameter describing a subproblem which is going to be condensed
        //
        ////////////////////////////////////////////////////////////////////
    
        //! Index indicating the first subproblem
        size_t         first;
        
        //! Index indicating the last subproblem
        size_t         last;
        
        //! Cluster to which the subproblems are associated 
        TCluster *     cluster;
        
        //! Level of the cluster in a superordinated cluster tree
        size_t         level;
        
        //! Pointer to the approximated eigensolution of the eigenvalue problem 
        //! (K_tilde,M_tilde) restricted to the subproblems described above
        TDenseMatrix * D_approx;
        TDenseMatrix * Z_approx;
        
        ////////////////////////////////////////////////////////////////////
        //
        // constructor and destructor
        //
        subprob_data_t ( )
        {
            //-------------------------------------
            // set default values of the subproblem 
            //-------------------------------------
            first    = 0;
            last     = 0;
            level    = 0;
            
            D_approx = nullptr;
            Z_approx = nullptr;
            cluster  = nullptr;
        }
    };
    
    
    //========================================================
    // Several parameters of THAMLS
    //========================================================    
    //! data cotaining all general parameters of the H-AMLS method
    parameter_base_t            _para_base;
    
    //! data cotaining all the parameters for the subsequent subspace iteration
    parameter_impro_t           _para_impro;
    
    //! data cotaining all the parameters for the parallelisation
    parameter_parallel_t        _para_parallel;
    
    //! data cotaining all the parameters for mode selection
    parameter_mode_selection_t  _para_mode_sel;
    
    //! data containing all the used 'Truncation Accuracies'
    trunc_acc_t                 _trunc_acc;
    
    //========================================================
    // Auxiliary data in THAMLS
    //========================================================
    //! pointers to the eigenvalue matrices of the subproblems
    std::vector < TDenseMatrix * >  _D;
    
    //! pointers to the eigenvector matrices of the subproblems
    std::vector < TDenseMatrix * >  _S;
    
    //! offsets of the subproblem matrices
    std::vector < idx_t >           _subproblem_ofs;
    
    //! offsets of the subproblem matrices
    std::vector < idx_t >           _reduced_subproblem_ofs;
    
    //! auxiliary data which helps to manage the several subproblems in
    //! the AMLS algorithm and which models their dependencies to each ohter 
    TSeparatorTree                  _sep_tree;
    
    //! cluster tree which represents the applied domain substructuring in the AMLS method
    //! NOTE: This attribut is a TCluster and not a TClusterTree but it represents the root of this tree
    TCluster *                      _root_amls_ct;
    

protected:

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // miscellaneous methods 
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
    //! return the number of subproblems
    size_t get_number_of_subproblems () const { return _sep_tree.n_subproblems(); }
    
    //! return true if subproblem with index 'i' is associated to a domain, otherwise return false
    bool subproblem_is_domain ( const idx_t i ) const { return _sep_tree.is_domain( i ); }
        
    //! determine offsets of the subproblem matrices
    void set_subproblem_ofs () ;
    
    //! determine offsets of the reduced subproblem matrices
    void set_reduced_subproblem_ofs () ;
    
    //! return the cluster representing the in the AMLS method applied domain subtructuring
    const TCluster * get_root_amls_ct () const { return _root_amls_ct; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Initialisation of H-AMLS
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //! Initialize auxiliary data 
    void init_amls_data ( const TClusterTree * ct ) ;   
      
    //! Delete auxiliary data
    void delete_amls_data () ;
                                                                                    
    //! Routine which truncates the cluster tree, representing the H-matrix structure, to a
    //! the cluster tree representing the domain substructuring applied in the AMLS method
    //! NOTE: The return value is a TCluster and not a TClusterTree but this return value 
    //! (saved as an internal attribut of the class) represents the root of this "AMLS cluster tree"
    void create_amls_clustertree ( const TClusterTree * ct ) ;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for the computation of the matrices of the transformed problem (K_tilde,M_tilde)
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //! Block diagonalize the matrix K = L * K_tilde * L^T and compute M_tilde = L^{-1} * M * L^{-T}
    void transform_problem            ( const TMatrix *  K, 
                                        const TMatrix *  M, 
                                        TMatrix * &      K_tilde,
                                        TMatrix * &      M_tilde,
                                        TMatrix * &      L ) ;  
                                        
    //! compute the block diagonal factorisation K = L * K_tilde * L^{T} 
    void transform_K_version_ldl      ( const TMatrix *  K,
                                        TMatrix * &      K_tilde,
                                        TMatrix * &      L ) ;
                                         
    
    //! different versions to compute the transformed matrix M_tilde = L^{-1} * M * L^{-T} 
    void transform_M_version_ldl      ( const TMatrix *  M,
                                        TMatrix * &      M_tilde,
                                        const TMatrix *  L ) ;
                                                                
    void transform_M_version_experimantal_1 ( const TMatrix *  M,
                                              TMatrix * &      M_tilde,
                                              const TMatrix *  L ) ;
                                        
    void transform_M_version_experimantal_2 ( const TMatrix *  M,
                                              TMatrix * &      M_tilde,
                                              const TMatrix *  L ) ;                                        
                                        
    //! save matrices of the tranformed problem (K_tilde,M_tilde)
    void save_transformed_problem( const TMatrix *  K_tilde, 
                                   const TMatrix *  M_tilde, 
                                   const TMatrix *  L ) ;
                                   
    //! load matrices of the tranformed problem (K_tilde,M_tilde)
    void load_transformed_problem( TMatrix * &  K_tilde, 
                                   TMatrix * &  M_tilde, 
                                   TMatrix * &  L ) ;
                                   
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for computing the partial eigensolutions of (K_tilde_ii, M_tilde_ii)
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                 
    enum subproblem_t
    { 
        SUBDOMAIN_SUBPROBLEM = 0,
        INTERFACE_SUBPROBLEM = 1,
        CONDENSE_SUBPROBLEM  = 2
    };
    
    //! Determine the number of wanted eigenvectors for the corresponding problem type
    size_t get_number_of_wanted_eigenvectors ( const subproblem_t  kind_of_subproblem,
                                               const size_t        dof ) const;
                                                                    
    //! Configure the eigensolver according to the selected truncation strategy
    void configure_lapack_eigensolver_subproblem ( TEigenLapack &     eigensolver, 
                                                   const size_t        n,
                                                   const subproblem_t  kind_of_subproblem ) const;
    //! Return true if the eigenvalue problem associated with index 'i' is small
    bool is_small_problem( const idx_t  i ) const;
    
    //! Get the subproblem M_tilde_ii for the index \a i of the matrix \a M_tilde
    void get_subproblem_matrix ( const TMatrix *    M_tilde,
                                 const idx_t        i,
                                 const TMatrix * &  M_tilde_ii ) const;
                                 
    //! Compute the eigen decompostion of small interface subproblem 
    size_t eigen_decomp_small_interface_problem ( const TMatrix *  K_tilde_ii,
                                                  const TMatrix *  M_tilde_ii,
                                                  TDenseMatrix  *  D_i,
                                                  TDenseMatrix  *  S_i,
                                                  const size_t     i ) ;
                                                  
    //! Compute the eigen decompostion of large interface subproblem 
    size_t eigen_decomp_large_interface_problem ( const TMatrix *  K_tilde_ii,
                                                  const TMatrix *  M_tilde_ii,
                                                  TDenseMatrix  *  D_i,
                                                  TDenseMatrix  *  S_i,
                                                  const size_t     i ) ;
                                            
    //! Compute the eigen decompostion of subdomain subproblem 
    size_t eigen_decomp_subdomain_problem ( const TMatrix *  K_tilde_ii,
                                            const TMatrix *  M_tilde_ii,
                                            TDenseMatrix  *  D_i,
                                            TDenseMatrix  *  S_i,
                                            const size_t     i ) ;
                                               
                                                               
    //! Compute the partial eigensolutions of the subproblems
    void comp_partial_eigensolutions ( const TMatrix *  K_tilde,
                                       const TMatrix *  M_tilde ) ;
                                       
                                       
    //! return eigenvalue matrix of the subproblem associated to the index \a i
    TDenseMatrix * get_D ( const uint i ) { return _D[i]; }
    
    //! return eigenvector matrix of the subproblem associated to the index \a i
    TDenseMatrix * get_S ( const uint i ) { return _S[i]; }

    //! set eigenvalue matrix of the subproblem associated to the index \a i
    void           set_D ( const uint i, TDenseMatrix * D_i ) { _D[i] = D_i; }
    
    //! set eigenvector matrix of the subproblem associated to the index \a i
    void           set_S ( const uint i, TDenseMatrix * S_i ) { _S[i] = S_i; }


                                       
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for computing the reduced matrices K_red and M_red
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    //! Step 1: Compute Temp = M_tilde_ij * S_j
    void comp_M_red_ij_aux_step_1 ( const TMatrix *      M_tilde_ij,
                                    const TDenseMatrix * S_j,
                                    TDenseMatrix *       Temp ) const;
                                    
    //! Step 1: Compute Temp = M_tilde_ij * S_j (this routine has a better parallel efficiency than 'comp_M_red_ij_aux_step_1')
    void comp_M_red_ij_aux_step_1_task ( const TMatrix *      M_tilde_ij,
                                         const TDenseMatrix * S_j,
                                         TDenseMatrix *       Temp ) const;
    
    //! Step 2: Compute M_red_ij = S_i^{T} * Temp
    void comp_M_red_ij_aux_step_2 ( const TDenseMatrix * S_i,
                                    const TDenseMatrix * Temp,
                                    TDenseMatrix *       M_red_ij ) const;                                    
                                    
    //! Compute M_red_ij = S_i^{T} * M_tilde_ij * S_j
    void comp_M_red_ij_aux ( const TDenseMatrix * S_i,
                             const TMatrix *      M_tilde_ij,
                             const TDenseMatrix * S_j,
                             TDenseMatrix *       M_red_ij ) ;
                                    
    //! Compute the submatrix of M_red in blockrow \a i and blockcolumn \a j
    void comp_M_red_ij ( const TMatrix *  M_tilde,
                         const idx_t      i,
                         const idx_t      j,
                         TDenseMatrix *   M_red_ij ) ;
                         
    //! Compute the reduced matrices K_red and M_red together in parallel
    //! (Note: Only the lower block triangular matrix of 'M_red' is computed because 'M_red' is symmetric)                     
    size_t comp_K_and_M_red_parallel ( const TMatrix *  K_tilde,
                                       const TMatrix *  M_tilde,
                                       TDenseMatrix *   K_red,
                                       TDenseMatrix *   M_red ) ;
                         
    //! Compute the matrices of the reduced problem (\a K_red, \a M_red )
    //! where K_red = S^T * K_tilde * S and M_red = S^T * M_tilde * S
    //! Input:  symmetric blockdiagonalized matrix \a K_tilde and symmetric positive definit matrix \a M_tilde 
    //! Output: symmetric dense matrices \a K_red and \a M_red,
    //! (Note: Only the lower block triangular matrix of 'M_red' is computed because 'M_red' is symmetric)
    void comp_reduced_matrices  ( const TMatrix *    K_tilde,
                                  const TMatrix *    M_tilde,
                                  TDenseMatrix *     K_red,
                                  TDenseMatrix *     M_red ) ;   
                                                                    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for solving the reduced eigenvalue problem (K_red,M_red)
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    
    //! auxiliary routine
    bool adjust_MKL_threads_for_eigen_decomp ( const size_t n ) const;
    
    //! auxiliary routine
    void set_MKL_threads_to_one () const ;
                                  
    //! Compute the eigen decompostion of the reduced problem
    size_t eigen_decomp_reduced_problem ( const TMatrix *  K_red,
                                          const TMatrix *  M_red,
                                          TDenseMatrix  *  D_red,
                                          TDenseMatrix  *  Z_red ) ;
                                         
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for improving the eigenvector approximations which are computed by H-AMLS
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // subroutine of 'improve_eigensolutions' and 'improve_eigensolutions_sparse'
    size_t improve_eigensolutions___eigen_decomp_reduced_problem ( const TMatrix *   K_tilde_c,
                                                                   const TMatrix *   M_tilde_c,
                                                                   TDenseMatrix  *   D_tilde_c,
                                                                   TDenseMatrix  *   S_tilde_c ) const;

    // subroutine of 'improve_eigensolutions'
    void improve_eigensolutions___comp_reduced_matrices ( const TMatrix *       K_tilde,
                                                          const TMatrix *       M_tilde,
                                                          const TDenseMatrix *  Q_tilde_new,
                                                          TDenseMatrix *        K_tilde_c,
                                                          TDenseMatrix *        M_tilde_c ) ;
    
    // subroutine of 'improve_eigensolutions'
    void improve_eigensolutions___iterate_eigenvectors ( const TMatrix *       K_tilde,
                                                         const TMatrix *       M_tilde,
                                                         const TDenseMatrix *  S_tilde,
                                                         TDenseMatrix *        Q_tilde_new ) ;
                                                           
    // subroutine of 'improve_eigensolutions_sparse'
    void improve_eigensolutions_sparse___comp_reduced_matrices ( const TSparseMatrix *  K,
                                                                 const TSparseMatrix *  M,
                                                                 const TDenseMatrix *   Q_new,
                                                                 const TDenseMatrix *   MQ,
                                                                 TDenseMatrix *         K_c,
                                                                 TDenseMatrix *         M_c ) ;
                                                                 
    // subroutine of 'improve_eigensolutions_sparse'
    void improve_eigensolutions_sparse___comp_reduced_matrices ( const TSparseMatrix *  K,
                                                                 const TSparseMatrix *  M,
                                                                 const TDenseMatrix *   Q_new,
                                                                 TDenseMatrix *         K_c,
                                                                 TDenseMatrix *         M_c ) ;

    // subroutine of 'improve_eigensolutions_sparse'
    void improve_eigensolutions_sparse___iterate_eigenvectors ( const TMatrix *        K_tilde,
                                                                const TMatrix *        L,
                                                                const TSparseMatrix *  M,
                                                                const TDenseMatrix *   S,
                                                                TDenseMatrix *         MQ,
                                                                TDenseMatrix *         Q_new ) ;
                                                                
    // subroutine of 'improve_eigensolutions_sparse'
    void improve_eigensolutions_sparse___iterate_eigenvectors ( const TMatrix *        K_tilde,
                                                                const TMatrix *        L,
                                                                const TSparseMatrix *  M,
                                                                const TDenseMatrix *   S,
                                                                TDenseMatrix *         Q_new ) ;
                                                                
    // subroutine of 'improve_eigensolutions_sparse'                                                            
    void improve_eigensolutions_sparse___iterate_eigenvectors_with_precond ( const TSparseMatrix *  K,
                                                                             const TSparseMatrix *  M,
                                                                             const TDenseMatrix *   S,
                                                                             TDenseMatrix *         MQ,
                                                                             TDenseMatrix *         Q_new ) ;
                                                                
    //-----------------------------------------------------------------------------------------------------
    //NOTE: The routines 'improve_eigensolutions_sparse' and 'improve_eigensolutions' improve the eigenpair
    //      approximations obtained by H-AMLS in the same way. However, the computational time of 
    //      'improve_eigensolutions_sparse' is much less than 'improve_eigensolutions'
    //-----------------------------------------------------------------------------------------------------
        
    // improve eigensolutions by subsequent subspace iteration
    void improve_eigensolutions_sparse ( const TSparseMatrix *  K,
                                         const TSparseMatrix *  M,
                                         const TMatrix *        K_tilde,
                                         const TMatrix *        L,
                                         TDenseMatrix *         D,
                                         TDenseMatrix *         S ) ;
                                         
    // improve eigensolutions by subsequent subspace iteration
    void improve_eigensolutions ( const TMatrix *   K_tilde,
                                  const TMatrix *   M_tilde,
                                  TDenseMatrix *    D_tilde,
                                  TDenseMatrix *    S_tilde ) ;
                                  
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Back transformation of the eigenvectors
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                           

    //! Transform the eigenvectors of the reduced EVP to eigenvector approximations of the original EVP
    //! and compute the corresponding Rayleigh quotients if wanted
    //! Input:  \a S_red matrix containing the eigenvectors of the reduced problem
    //!         \a K_tilde and \a M_tilde are the transformed mass and stiffness matrix
    //!         \a L transformation matrix L
    //!         \a K_sparse and \a M_sparse are the sparse matrices of K and M
    //!         \a ct is the cluster tree containing the permutation info between sparse and H-matrix format
    //! Output: \a S_approx matrix containing the eigenvector approxmations for the original problem (K,M)
    //!         \a D_approx is the diagonal matrix containing the corresponding eigenvalue approximations
    void transform_eigensolutions ( const TSparseMatrix * K_sparse, 
                                    const TSparseMatrix * M_sparse, 
                                    const TClusterTree *  ct,
                                    const TMatrix *       K_tilde,
                                    const TMatrix *       M_tilde,
                                    const TMatrix *       L,
                                    const TDenseMatrix *  S_red,
                                    TDenseMatrix *        D_approx,
                                    TDenseMatrix *        S_approx ) ;

    //! Compute S_tilde:= diag[S_1,...,S_m] * S_red
    void backtransform_eigenvectors_with_S_i ( const TDenseMatrix * S_red,
                                               TDenseMatrix *       S_tilde ) ;
                                               
    //! Compute S_tilde:= diag[S_1,...,S_m] * S_red for corresponding subproblem 
    void backtransform_eigenvectors_with_S_i ( const TDenseMatrix * S_red,
                                               TDenseMatrix *       S_tilde,
                                               subprob_data_t *     subprob_data ) ;

    //! Compute S_tilde:= op(L)^{-1} * S_tilde
    void backtransform_eigenvectors_with_L ( const TMatrix * L,
                                             const matop_t   op_L,
                                             TDenseMatrix *  S_tilde ) const;
                                                                            
    void comp_S_i_times_S_i_red ( const TDenseMatrix * S_i,
                                  const TDenseMatrix * S_red_i,
                                  TDenseMatrix *       S_tilde_i ) const;
                                      
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Routines for the recursive approach 
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                               
    //! This routine gets the indices of all subproblems which are associated 
    //! with the given \a cluster and initialise the corresponding 'subprob_data_t'.
    //! Note that the \a cluster gets conneted to 'subprob_data_t' with this function.
    void get_subprob_data ( const size_t      lvl,
                            TCluster *        cluster,
                            subprob_data_t *  subprob_data ) const;
                                           
    //! Get the number of the selected eigenvectors which are
    //! associated to the subproblems in the 'subprob_data'.
    size_t get_reduced_dofs_SUB ( const subprob_data_t *  subprob_data ) const;
             
    //! Get the DOF which are associated to the subproblems in the 'subprob_data'.
    size_t get_dofs_SUB ( const subprob_data_t *  subprob_data ) const;
    
    //! Decide if the subproblems contained in the 'subprob_data' should be condensed
    bool shall_problem_be_condensed ( const size_t            lvl,
                                      const size_t            max_lvl,
                                      const subprob_data_t *  subprob_data ) const;
    
    
    //! Save subproblems in the given \a subproblems_2_condense list which shall be condensed.
    //! Return true if problems have been found otherwise false. In this routine also different
    //! strategies can be implemented on how to find problems.
    bool find_subproblems_which_shall_be_condensed ( std::list< subprob_data_t * > & subproblems_2_condense ) const;
                          
        
    //! Compute the reduced matrices K_red and M_red associated to the subproblems contained in 'subprob_data' together in parallel
    //! (Note: Only the lower block triangular matrix of 'M_red_sub' is computed because 'M_red_sub' is symmetric)
    size_t comp_K_and_M_red_parallel ( const TMatrix *         K_tilde,
                                       const TMatrix *         M_tilde,
                                       TDenseMatrix *          K_red_sub,
                                       TDenseMatrix *          M_red_sub,
                                       const subprob_data_t *  subprob_data ) ;
                                            
    //! Compute the reduced matrices K_red_sub and M_red_sub associated to the subproblems contained in 'subprob_data' 
    //! (Note: Only the lower block triangular matrix of 'M_red_sub' is computed because 'M_red_sub' is symmetric)
    void comp_reduced_matrices_SUB ( const TMatrix *   K_tilde,
                                     const TMatrix *   M_tilde,
                                     TDenseMatrix *    K_red_sub,
                                     TDenseMatrix *    M_red_sub,
                                     subprob_data_t *  subprob_data ) ;
                                                              
    //! Solve the reduced eigenvalue problem restricted to the subproblems contained 
    //! in 'subprob_data' and return the number of computed eigenpairs
    size_t eigen_decomp_reduced_SUB ( const TMatrix *   K_red_sub,
                                      const TMatrix *   M_red_sub,
                                      TDenseMatrix  *   D,
                                      TDenseMatrix  *   Z,
                                      subprob_data_t *  subprob_data ) const;
                                                   
    //! Determine the Rayleigh Quotients of the column vectors contained in 'S_approx' 
    //! according to the general eigenvalue problem (K_tilde,M_tilde) associated to the
    //! subproblems contained in 'subprob_data' and substitute the values in 'D_approx'
    void compute_rayleigh_quotients_SUB ( const TMatrix *       K_tilde, 
                                          const TMatrix *       M_tilde, 
                                          TDenseMatrix *        D_approx, 
                                          const TDenseMatrix *  Z_approx, 
                                          subprob_data_t *      subprob_data ) ;
                                                           
    //! Compute the approximative eigensolution of the EVP (K_tilde,M_tilde) restricted
    //! to the subproblems contained in 'subprob_data'. The approximate eigensolution 
    //! is obtained by applying H-AMLS to the subproblems contained in 'subprob_data'   
    void apply_HAMLS_subproblem ( const TMatrix *   K_tilde, 
                                  const TMatrix *   M_tilde, 
                                  TDenseMatrix *    D_approx, 
                                  TDenseMatrix *    Z_approx, 
                                  subprob_data_t *  subprob_data ) ;
    
    //! Print information of subproblem described in 'subprob_data'
    void output_SUB ( const TMatrix *         K_tilde, 
                      const TMatrix *         M_tilde, 
                      const subprob_data_t *  subprob_data,
                      const size_t            problem_number ) const;
                                           
    //! Delete all sons of \a cluster_2_find in the cluster tree \a new_root_amls_ct
    void truncate_amls_clustertree ( const TCluster * cluster_2_find,
                                     TCluster *       new_root_amls_ct ) const;  
                                                                                        
    //! Try to condense subproblems together and update the corresponding auxiliary data.
    //! Return how often several subproblems could be condensed.
    size_t try_2_condense_suproblems ( const TMatrix *  K_tilde,
                                       const TMatrix *  M_tilde ) ;
                           
    //! Print detailed inforamtion about the time consumption
    void print_performance_SUB ( const subprob_data_t *  subprob_data,
                                 const bool              print_WALL_TIME ) const;
                               
    //! Print short summary of the finished method 
    //! ( 'D_approx' is the matrix containing the approximated eigenvalues)
    void print_summary_SUB ( const TMatrix *         D_approx,
                             const subprob_data_t *  subprob_data ) const;
                                         
    //! Print the structural informtion
    void print_structural_info_SUB ( const subprob_data_t *  subprob_data,
                                     const bool              is_real_subproblem = true ) const;
    
    //! Condense subproblems together as long as possible
    void apply_condensing ( const TMatrix *  K_tilde,
                            const TMatrix *  M_tilde ) ;
                            
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Analyse routines
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //! Print short summary of the finished method 
    //! ( 'D' is the matrix containing the approximated eigenvalues)
    void print_summary ( const TMatrix * D_approx ) const;
    
    //! Print memory consumption of the involved matrices 
    void print_matrix_memory ( const TSparseMatrix * K_sparse, 
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
                               const TMatrix *       Z_approx ) const;
             
    //! Print the structural informtion
    void print_structural_info () const;
    
    //! Visualisation of the stiffness matrix K or mass matrix M
    void visualize_matrix ( const TMatrix *      M,
                            const std::string &  filename,
                            const bool           show_full_sym = false ) const;
                            
    void visualize_biggest_interface_problem ( const TMatrix *      M,
                                               const std::string &  filename_M,
                                               const bool           show_full_sym = false ) const;
    //! Visualisation of a cluster 
    void visualize_cluster ( const TCluster *    cl,
                            const std::string &  filename ) const;
        
    //! Print the options used for the eigensolver
    void print_options () const;
    
    //! Print detailed inforamtion about the time consumption
    //! (if parameter is true wall time is used otherwise cpu time)
    void print_performance ( const bool print_WALL_TIME ) const;
                                                                                                 
    //! Debug Routine: Test the applied problem transformation by computing the norm of the residuals
    void test_problem_transformation ( const TMatrix * K,
                                       const TMatrix * M,
                                       const TMatrix * K_tilde,
                                       const TMatrix * M_tilde,
                                       const TMatrix * L ) ;
                                 
    

public:
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // constructor and destructor
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //! constructor
    THAMLS () 
    {
        _para_mode_sel.mode_selection = MODE_SEL_AUTO_H;
        // other mode selection strategies not support right now, but can be easily implemented
    }
            
    //! dtor
    virtual ~THAMLS () {}
        
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // access local variables
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    //====================================================================================
    // set/get basic parameters
    //====================================================================================
    bool   get_do_condensing () const { return _para_base.do_condensing; }
    void   set_do_condensing ( const bool do_condensing ) { _para_base.do_condensing = do_condensing; }    
    
    bool   get_stable_rayleigh () const { return _para_base.stable_rayleigh; }
    void   set_stable_rayleigh ( const bool stable_rayleigh ) { _para_base.stable_rayleigh = stable_rayleigh; }    
    
    bool   get_print_info () const { return _para_base.print_info; }
    void   set_print_info ( const bool print_info ) { _para_base.print_info = print_info; }

    bool   get_do_debug () const { return _para_base.do_debug; }
    void   set_do_debug ( const bool do_debug ) { _para_base.do_debug = do_debug; }
    
    bool   get_use_arnoldi () const { return _para_base.use_arnoldi; }
    void   set_use_arnoldi ( const bool use_arnoldi ) { _para_base.use_arnoldi = use_arnoldi; }    
    
    bool   get_coarsen () const { return _para_base.coarsen; }
    void   set_coarsen ( const bool coarsen ) { _para_base.coarsen = coarsen; }
    
    size_t get_n_ev_searched () const { return _para_base.n_ev_searched; }
    void   set_n_ev_searched ( const size_t n_ev_searched ) { _para_base.n_ev_searched = n_ev_searched; }
    
    bool   get_load_problem () const { return _para_base.load_problem; }
    void   set_load_problem ( const bool load_problem ) { _para_base.load_problem = load_problem; }
    
    bool   get_save_problem () const { return _para_base.save_problem; }
    void   set_save_problem ( const bool save_problem ) { _para_base.save_problem = save_problem; }
    
    void   set_io_location ( const std::string io_location ) { _para_base.io_location = io_location; }
    
    void   set_io_prefix ( const std::string io_prefix ) { _para_base.io_prefix = io_prefix; }
    
    size_t get_max_dof_subdomain () const { return _para_base.max_dof_subdomain; }
    void   set_max_dof_subdomain ( const size_t max_dof_subdomain ) { _para_base.max_dof_subdomain = max_dof_subdomain; }

    //====================================================================================
    // set/get miscellaneous parameters
    //====================================================================================
    
    bool   get_do_improving () const { return _para_impro.do_improving; }
    void   set_do_improving ( const bool do_improving ) { _para_impro.do_improving = do_improving; }   
    
    //====================================================================================
    // set/get mode selection parameters
    //====================================================================================
    mode_selection_t get_mode_selection () const { return _para_mode_sel.mode_selection; }
    void   set_mode_selection ( const mode_selection_t mode_selection ) { _para_mode_sel.mode_selection = mode_selection; }   
    
    real   get_rel () const { return _para_mode_sel.rel; }
    void   set_rel ( const real rel ) { _para_mode_sel.rel = rel; }
    
    size_t get_abs () const { return _para_mode_sel.abs; }
    void   set_abs ( const size_t abs ) { _para_mode_sel.abs = abs; }
            
    size_t get_nmin_ev () const { return _para_mode_sel.nmin_ev; }
    void   set_nmin_ev ( const size_t nmin_ev ) { _para_mode_sel.nmin_ev = nmin_ev; }
        
    real   get_trunc_bound () const { return _para_mode_sel.trunc_bound; }
    void   set_trunc_bound ( const real trunc_bound ) { _para_mode_sel.trunc_bound = trunc_bound; }
            
    size_t get_condensing_size () const { return _para_mode_sel.condensing_size; }
    void   set_condensing_size ( const size_t condensing_size ) { _para_mode_sel.condensing_size = condensing_size; }
    
    real   get_factor_condense () const { return _para_mode_sel.factor_condense; }
    void   set_factor_condense ( const real factor_condense ) { _para_mode_sel.factor_condense = factor_condense; }
        
    real   get_factor_subdomain () const { return _para_mode_sel.factor_subdomain; }
    void   set_factor_subdomain ( const real factor_subdomain ) { _para_mode_sel.factor_subdomain = factor_subdomain; _para_mode_sel.factor_condense = 4*factor_subdomain; }
    
    real   get_factor_interface () const { return _para_mode_sel.factor_interface; }
    void   set_factor_interface ( const real factor_interface ) { _para_mode_sel.factor_interface = factor_interface; }
    
    real  get_exponent_interface () const { return _para_mode_sel.exponent_interface; }
    void   set_exponent_interface ( const real exponent_interface ) { _para_mode_sel.exponent_interface = exponent_interface; }
    
    real get_exponent_subdomain () const { return _para_mode_sel.exponent_subdomain; }
    void   set_exponent_subdomain ( const real exponent_subdomain ) { _para_mode_sel.exponent_subdomain = exponent_subdomain; }
    
    //====================================================================================
    // set/get truncation accuracy parameters 
    //====================================================================================
    const TTruncAcc & get_acc_transform_K () const { return _trunc_acc.transform_K; }
    void              set_acc_transform_K ( const TTruncAcc & acc ) { _trunc_acc.transform_K = acc; }
    
    const TTruncAcc & get_acc_transform_M () const { return _trunc_acc.transform_M; }
    void              set_acc_transform_M ( const TTruncAcc & acc ) { _trunc_acc.transform_M = acc; }
    
    void  set_arithmetic_acc ( real eps ) { const TTruncAcc acc( eps, real(0) ); _trunc_acc.transform_K = acc;  _trunc_acc.transform_M = acc; }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // miscellaneous methods 
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                      
    //! Solve the generalized eigenvalue problem K*Z = M*Z*D with (Z^T)*M*Z = Id approximatively.
    //! The return value of the method is the number of computed eigenvalues.
    //!
    //! Input:  
    //! \a K and \a M are symmetric H-matrices and \a M is positive definite,
    //! \a ct is the cluster tree associated tp the H-matrix format of the H-matrices \a K and \a M
    //!
    //! optional Input:        
    //! \a K_sparse and \a M_sparse are the original sparse matrices from which the H-matrices \a K and \a M
    //! have been derived. These matrices as addtional input (recommended but not mandatory) when a subseqent 
    //! subspace iteration step is performed or the Rayleig-Quotients are computed.
    //!
    //! Output: 
    //! \a D is a diagonal matrix containing the selected eigenvalues in ascending order,
    //! \a Z is the matrix containing the corresponding eigenvectors
    size_t comp_decomp ( const TClusterTree *   ct,
                         const TMatrix *        K,
                         const TMatrix *        M,                      
                         TDenseMatrix  *        D,
                         TDenseMatrix  *        Z,
                         const TSparseMatrix *  K_sparse   = nullptr, 
                         const TSparseMatrix *  M_sparse   = nullptr ) ;
};                    


}// namespace HAMLS

#endif  // __HAMLS_THAMLS_HH
