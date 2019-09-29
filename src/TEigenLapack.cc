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
// File        : TEigenLapack.cc
// Description : class for the eigen decomposition of symmetric eigenvalue problems using LAPACK
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include <vector>

#include "hamls/TEigenLapack.hh"

namespace HLIB // for blas_int_t
{

extern "C"
{

// compute eigenvalues and eigenvectors of the generalized symmetric eigenvalue problem
CFUNCDECL
void 
ssygv_ (   const blas_int_t *    itype, 
           const char *          jobz, 
           const char *          uplo, 
           const blas_int_t *    n, 
           float *               A, 
           const blas_int_t *    lda, 
           float *               B, 
           const blas_int_t *    ldb, 
           float *               w, 
           float *               work, 
           const blas_int_t *    lwork, 
           blas_int_t *          info );      
              
// compute selected eigenvalues and eigenvectors of the generalized symmetric eigenvalue problem
CFUNCDECL
void 
ssygvx_ (  const blas_int_t *   itype, 
           const char *         jobz, 
           const char *         range, 
           const char *         uplo,
           const blas_int_t *   n, 
           float *              A, 
           const blas_int_t *   ldA, 
           float *              B, 
           const blas_int_t *   ldb, 
           const float *        vl, 
           const float *        vu, 
           const blas_int_t *   il, 
           const blas_int_t *   iu,
           const float *        abstol, 
           blas_int_t *         m, 
           float *              W, 
           float *              Z, 
           const blas_int_t *   ldZ,
           float *              work, 
           const blas_int_t *   lwork, 
           blas_int_t *         iwork, 
           blas_int_t *         ifail, 
           blas_int_t *         info );

// compute eigenvalues and eigenvectors of the generalized symmetric eigenvalue problem
CFUNCDECL
void 
dsygv_ (   const blas_int_t *    itype, 
           const char *          jobz, 
           const char *          uplo,  
           const blas_int_t *    n, 
           double *              A, 
           const blas_int_t *    lda, 
           double *              B, 
           const blas_int_t *    ldb, 
           double *              w, 
           double *              work, 
           const blas_int_t *    lwork, 
           blas_int_t *          info );

// compute selected eigenvalues and eigenvectors of the generalized symmetric eigenvalue problem
CFUNCDECL
void 
dsygvx_ (  const blas_int_t *   itype, 
           const char *         jobz, 
           const char *         range, 
           const char *         uplo,
           const blas_int_t *   n, 
           double *             A, 
           const blas_int_t *   ldA, 
           double *             B, 
           const blas_int_t *   ldb, 
           const double *       vl, 
           const double *       vu, 
           const blas_int_t *   il, 
           const blas_int_t *   iu,
           const double *       abstol, 
           blas_int_t *         m, 
           double *             W, 
           double *             Z, 
           const blas_int_t *   ldZ,
           double *             work, 
           const blas_int_t *   lwork, 
           blas_int_t *         iwork, 
           blas_int_t *         ifail, 
           blas_int_t *         info ); 

}// extern "C"

}// namespace HLIB

namespace HAMLS
{

using std::unique_ptr;

namespace
{
    
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//
// local functions
//
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

//
// *sygv/*hegv
//
template <typename T>  
void hegv ( const blas_int_t                  itype, 
            const char                        jobz, 
            const char                        uplo, 
            const blas_int_t                  n,
            T *                               A, 
            const blas_int_t                  ldA, 
            T *                               B, 
            const blas_int_t                  ldB, 
            typename real_type< T >::type_t * W,
            T *                               work, 
            const blas_int_t                  lwork, 
            typename real_type< T >::type_t * rwork, 
            blas_int_t &                      info );

#define HLIB_HEGV_FUNC( type, func )                                    \
    template <> inline void hegv<type> ( const blas_int_t             itype, \
                                         const char                   jobz, \
                                         const char                   uplo, \
                                         const blas_int_t             n, \
                                         type *                       A, \
                                         const blas_int_t             ldA, \
                                         type *                       B, \
                                         const blas_int_t             ldB, \
                                         real_type< type >::type_t *  W, \
                                         type *                       work, \
                                         const blas_int_t             lwork, \
                                         real_type< type >::type_t *  , \
                                         blas_int_t &                 info ) { \
        info = 0;                                                       \
        func( & itype, & jobz, & uplo, & n, A, & ldA, B,                \
              & ldB, W, work, & lwork, & info ); }

HLIB_HEGV_FUNC( float,  ssygv_ )
HLIB_HEGV_FUNC( double, dsygv_ )

#undef HLIB_HEGV_FUNC

//
// *sygvx/*hegvx
//
template <typename T>  
void hegvx ( const blas_int_t                  itype, 
             const char                        jobz, 
             const char                        range, 
             const char                        uplo,
             const blas_int_t                  n, 
             T *                               A, 
             const blas_int_t                  ldA, 
             T *                               B, 
             const blas_int_t                  ldB,
             typename real_type< T >::type_t   vl, 
             typename real_type< T >::type_t   vu, 
             const blas_int_t                  il, 
             const blas_int_t                  iu, 
             blas_int_t &                      m,
             typename real_type< T >::type_t * W, 
             T *                               Z, 
             const blas_int_t                  ldZ,
             T *                               work, 
             const blas_int_t                  lwork, 
             typename real_type< T >::type_t * rwork, 
             blas_int_t *                      iwork, 
             blas_int_t *                      ifail, 
             blas_int_t &                      info );

#define HLIB_HEGVX_FUNC( type, func )                                   \
    template <> inline void hegvx<type> ( const blas_int_t                 itype, \
                                          const char                       jobz, \
                                          const char                       range, \
                                          const char                       uplo, \
                                          const blas_int_t                 n, \
                                          type *                           A, \
                                          const blas_int_t                 ldA, \
                                          type *                           B, \
                                          const blas_int_t                 ldB, \
                                          const real_type< type >::type_t  vl, \
                                          const real_type< type >::type_t  vu, \
                                          const blas_int_t                 il, \
                                          const blas_int_t                 iu, \
                                          blas_int_t &                     m, \
                                          real_type< type >::type_t *      W, \
                                          type *                           Z, \
                                          const blas_int_t                 ldZ, \
                                          type *                           work, \
                                          const blas_int_t                 lwork, \
                                          real_type< type >::type_t *      , \
                                          blas_int_t *                     iwork, \
                                          blas_int_t *                     ifail, \
                                          blas_int_t &                     info ) { \
        real_type< type >::type_t  abstol = 0;                          \
        info = 0;                                                       \
        func( & itype, & jobz, & range, & uplo, & n, A, & ldA, B, & ldB, & vl, & vu, & il, & iu, \
              & abstol, & m, W, Z, & ldZ, work, & lwork, iwork, ifail, & info ); }

HLIB_HEGVX_FUNC( float,  ssygvx_ )
HLIB_HEGVX_FUNC( double, dsygvx_ )

#undef HLIB_HEGVX_FUNC   


// LAPACK constant for a workspace query
const blas_int_t  LAPACK_WS_QUERY = blas_int_t(-1);

//
// compute eigenvalues and eigenvectors of the generalized symmetric-definite eigenvalue problem
//
template < typename T1,
           typename T2 >
typename enable_if_res< BLAS::is_matrix< T1 >::value &&
                        BLAS::is_matrix< T2 >::value &&
                        is_same_type< typename T1::value_t, typename T2::value_t >::value,
                        size_t >::result
eigen ( const T1 &                              K,
        const T2 &                              M,
        BLAS::Vector< typename T1::value_t > &  eig_val,
        BLAS::Matrix< typename T1::value_t > &  eig_vec,
        BLAS::Matrix< typename T1::value_t > &  cholesky )
{
    using  value_t = typename T1::value_t;
    using  real_t  = typename real_type< value_t >::type_t;
    
    /////////////////////////////////////////
    //
    // do some consistency checks
    //
    /////////////////////////////////////////
    
    if ( M.nrows() != K.nrows() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );
        
    if ( M.ncols() != K.ncols() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );    
    
    const size_t n = M.nrows();
    const size_t m = M.ncols();

    blas_int_t  info = 0;
    
    if ( n == m )
    {            
        if (( eig_vec.nrows() != M.nrows() ) || ( eig_vec.ncols() != M.ncols() ))
            eig_vec = std::move( BLAS::Matrix< value_t >( M.nrows(), M.ncols() ) );
            
        if (( cholesky.nrows() != M.nrows() ) || ( cholesky.ncols() != M.ncols() ))
            cholesky = std::move( BLAS::Matrix< value_t >( M.nrows(), M.ncols() ) );
                    
        /////////////////////////////////////////
        //
        // determine size of work space
        //
        /////////////////////////////////////////             
        
        value_t  work_query = value_t(0);
        
        // work space query                 
        hegv<value_t>( 1, 'V', 'L', blas_int_t(n), eig_vec.data(), blas_int_t(eig_vec.col_stride()), 
                 cholesky.data(), blas_int_t(cholesky.col_stride()),  
                 nullptr, & work_query, LAPACK_WS_QUERY, nullptr, info );
                 
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gv", -info ) );
            
        /////////////////////////////////////////
        //
        // compute eigenpairs
        //
        /////////////////////////////////////////   
        
        const blas_int_t         lwork = blas_int_t( re( work_query ) );
        BLAS::Vector< real_t >   seig_val( n );
        std::vector< value_t >   work( lwork );
        std::vector< real_t >    rwork( is_complex_type< value_t >::value ? 3*n-2 : 0 );
        
        BLAS::copy( K, eig_vec  );
        BLAS::copy( M, cholesky );
        
        //
        // Note: only the lower half of K and M is accessed
        //
        hegv<value_t>( 1, 'V', 'L', blas_int_t(n), eig_vec.data(), blas_int_t(eig_vec.col_stride()), 
                       cholesky.data(), blas_int_t(cholesky.col_stride()), 
                       seig_val.data(), & work[0], lwork, & rwork[0], info );
                 
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gv", -info ) );
                 
        /////////////////////////////////////////
        //
        // set output data
        //
        ///////////////////////////////////////// 
                 
        if ( eig_val.length() != n )
            eig_val = std::move( BLAS::Vector< value_t >( n ) );

        for ( idx_t  i = 0; i < idx_t(n); ++i )
            eig_val(i) = seig_val(i);
                    
        return eig_val.length();
    }// if
    else 
    {
        HERROR( ERR_NOT_IMPL, "(BLAS) eigen", "unsymmetric case" );
    }// else
}

//
// compute selected eigenvalues and eigenvectors of the generalized symmetric-definite eigenvalue problem
//
template < typename T1,
           typename T2 >
typename enable_if_res< BLAS::is_matrix< T1 >::value &&
                        BLAS::is_matrix< T2 >::value &&
                        is_same_type< typename T1::value_t, typename T2::value_t >::value,
                        size_t >::result
eigen ( const T1 &                              K,
        const T2 &                              M,
        const BLAS::Range &                     eig_range,
        BLAS::Vector< typename T1::value_t > &  eig_val,
        BLAS::Matrix< typename T1::value_t > &  eig_vec,
        BLAS::Matrix< typename T1::value_t > &  cholesky )
{
    using  value_t = typename T1::value_t;
    using  real_t  = typename real_type< value_t >::type_t;
    
    /////////////////////////////////////////
    //
    // do some consistency checks
    //
    /////////////////////////////////////////  
    
    //
    // first check, if K and M are symmetric or hermitian
    //
    if ( M.nrows() != K.nrows() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );
        
    if ( M.ncols() != K.ncols() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );
        
    if ( eig_range.first() > eig_range.last() )
        HERROR( ERR_ARG, "(BLAS) eigen", "argument range is not consistent" );
    

    const size_t  n    = M.nrows();
    const size_t  m    = M.ncols();
    blas_int_t    info = 0;
    
    if ( n == m )
    {
        if (( eig_vec.nrows() != M.nrows() ) || ( eig_vec.ncols() != eig_range.size() ))
            eig_vec = std::move( BLAS::Matrix< value_t >( M.nrows(), eig_range.size() ) );
    
        if (( cholesky.nrows() != M.nrows() ) || ( cholesky.ncols() != M.ncols() ))
            cholesky = std::move( BLAS::Matrix< value_t >( M.nrows(), M.ncols() ) );
    
        /////////////////////////////////////////
        //
        // determine size of work space
        //
        /////////////////////////////////////////             
        
        value_t     work_query = value_t(0);
        blas_int_t  number_ev   = 0;
        
        BLAS::Matrix< value_t >  K_temp ( n, n ); 
        
        // work space query
        hegvx<value_t>( 1, 'V', 'I', 'L', blas_int_t(n), K_temp.data(), blas_int_t(K_temp.col_stride()), 
                        cholesky.data(), blas_int_t(cholesky.col_stride()),
                        0.0, 0.0, blas_int_t(eig_range.first())+1, blas_int_t(eig_range.last())+1,
                        number_ev, nullptr, eig_vec.data(), blas_int_t(eig_vec.col_stride()),
                        & work_query, LAPACK_WS_QUERY, nullptr, nullptr, nullptr, info );
                
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gvx", -info ) );
            
        /////////////////////////////////////////
        //
        // compute eigenpairs
        //
        /////////////////////////////////////////   
                        
        const blas_int_t           lwork = blas_int_t( re( work_query ) );
        BLAS::Vector< real_t >     seig_val( n );
        std::vector< value_t >     work( lwork );
        std::vector< real_t >      rwork( is_complex_type< value_t >::value ? 7*n : 0 );
        std::vector< blas_int_t >  iwork( 5*n );
        std::vector< blas_int_t >  ifail(  n  );
        
        BLAS::copy( K, K_temp  );
        BLAS::copy( M, cholesky );

        //
        // Note: only the lower half of K and M is accessed
        //
        hegvx<value_t>( 1, 'V', 'I', 'L', blas_int_t(n), K_temp.data(), blas_int_t(K_temp.col_stride()), 
                        cholesky.data(), blas_int_t(cholesky.col_stride()),
                        0.0, 0.0, blas_int_t(eig_range.first())+1, blas_int_t(eig_range.last())+1,
                        number_ev, seig_val.data(), eig_vec.data(), blas_int_t(eig_vec.col_stride()),
                        & work[0], lwork, & rwork[0], & iwork[0], & ifail[0], info );
                
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gvx", -info ) );
            
        if ( number_ev != blas_int_t(eig_range.size()) )
        {
            // std::cout<<std::endl;
            // std::cout<<std::endl<<" problem size     = "<<n;
            // std::cout<<std::endl;
            // std::cout<<std::endl<<" number_ev        = "<<number_ev;
            // std::cout<<std::endl<<" eig_range.size() = "<<eig_range.size();
            // std::cout<<std::endl;
            // std::cout<<std::endl;
            
            HERROR( ERR_NOT_IMPL, "", "" );
        }// if
                    
        /////////////////////////////////////////
        //
        // set output data
        //
        ///////////////////////////////////////// 
        
        if ( eig_val.length() != size_t(number_ev) )
            eig_val = std::move( BLAS::Vector< value_t >( number_ev ) );
        
        for ( idx_t  i = 0; i < idx_t(number_ev); ++i )
            eig_val(i) = seig_val(i);
                
        return number_ev;
    }// if
    else 
    {
        HERROR( ERR_NOT_IMPL, "(BLAS) eigen", "unsymmetric case" );
    }// else
}

//
// compute selected eigenvalues and eigenvectors of the generalized symmetric-definite eigenvalue problem
//
template < typename T1,
           typename T2,
           typename T3 >
typename enable_if_res< BLAS::is_matrix< T1 >::value &&
                        BLAS::is_matrix< T2 >::value &&
                        is_same_type< T3, typename real_type< typename T1::value_t >::type_t >::value &&
                        is_same_type< T3, typename real_type< typename T2::value_t >::type_t >::value,
                        size_t >::result
eigen ( const T1 &                              K,
        const T2 &                              M,
        const T3                                lbound,
        const T3                                ubound,
        BLAS::Vector< typename T1::value_t > &  eig_val,
        BLAS::Matrix< typename T1::value_t > &  eig_vec,
        BLAS::Matrix< typename T1::value_t > &  cholesky )
{
    using  value_t = typename T1::value_t;
    using  real_t  = typename real_type< value_t >::type_t;
    
    /////////////////////////////////////////
    //
    // do some consistency checks
    //
    /////////////////////////////////////////  
    
    //
    // first check, if K and M are symmetric or hermitian
    //
    if ( M.nrows() != K.nrows() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );
        
    if ( M.ncols() != K.ncols() )
        HERROR( ERR_CONSISTENCY, "(BLAS) eigen", "matrices have different size" );
        
    if ( lbound >= ubound )
        HERROR( ERR_ARG, "(BLAS) eigen", "argument lower bound is not smaller than upper bound" );
    

    const size_t  n = M.nrows();
    const size_t  m = M.ncols();

    blas_int_t  info = 0;
    
    if ( n == m )
    {    
        if (( cholesky.nrows() != M.nrows() ) || ( cholesky.ncols() != M.ncols() ))
            cholesky = std::move( BLAS::Matrix< value_t >( M.nrows(), M.ncols() ) );
    
        /////////////////////////////////////////
        //
        // determine size of work space
        //
        /////////////////////////////////////////    
    
        value_t     work_query = value_t(0);
        blas_int_t  number_ev   = 0;
        
        BLAS::Matrix< value_t >  K_temp  ( n, n ); 
        BLAS::Matrix< value_t >  seig_vec( n, n ); 
        BLAS::Vector< real_t >   seig_val( n );
            
        // work space query
        hegvx<value_t>( 1, 'V', 'V', 'L', blas_int_t(n), K_temp.data(), blas_int_t(K_temp.col_stride()), 
                        cholesky.data(), blas_int_t(cholesky.col_stride()),
                        lbound, ubound, 0, 0,
                        number_ev, nullptr, seig_vec.data(), blas_int_t(seig_vec.col_stride()),
                        & work_query, LAPACK_WS_QUERY, nullptr, nullptr, nullptr, info );
        
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gvx", -info ) );
            
        /////////////////////////////////////////
        //
        // compute eigenpairs
        //
        /////////////////////////////////////////   
                        
        const blas_int_t           lwork = blas_int_t( re( work_query ) );
        std::vector< value_t >     work( lwork );
        std::vector< real_t >      rwork( is_complex_type< value_t >::value ? 7*n : 0 );
        std::vector< blas_int_t >  iwork( 5*n );
        std::vector< blas_int_t >  ifail(  n  );
            
        BLAS::copy( K, K_temp  );
        BLAS::copy( M, cholesky );
        
        //
        // Note: only the lower half of K and M is accessed
        //
        hegvx<value_t>( 1, 'V', 'V', 'L', blas_int_t(n), K_temp.data(), blas_int_t(K_temp.col_stride()), 
                        cholesky.data(), blas_int_t(cholesky.col_stride()),
                        lbound, ubound, 0, 0,
                        number_ev, seig_val.data(), seig_vec.data(), blas_int_t(seig_vec.col_stride()),
                        & work[0], lwork, & rwork[0], & iwork[0], & ifail[0], info );
                  
        if ( info != 0 )
            HERROR( ERR_ARG, "(BLAS) eigen", to_string( "argument %d to LAPACK::*(sy|he)gvx", -info ) );
        
        /////////////////////////////////////////
        //
        // set output data
        //
        ///////////////////////////////////////// 
        
        if ( eig_val.length() != size_t(number_ev) )
            eig_val = std::move( BLAS::Vector< value_t >( number_ev ) );
        
        for ( idx_t  i = 0; i < idx_t(number_ev); ++i )
            eig_val(i) = seig_val(i);
            
        if (( eig_vec.nrows() != M.nrows() ) || ( eig_vec.ncols() != size_t(number_ev) ))
            eig_vec = std::move( BLAS::Matrix< value_t >( M.nrows(), number_ev ) );
            
        for ( idx_t  j = 0; j < idx_t(number_ev); ++j )
        {
            for ( idx_t  i = 0; i < idx_t(M.nrows()); ++i )
            {
                eig_vec(i,j) = seig_vec(i,j);
            }// for
        }// for
            
        return number_ev;
    }// if
    else 
    {
        HERROR( ERR_NOT_IMPL, "(BLAS) eigen", "unsymmetric case" );
    }// else
}
    
}// namespace anonymous
    


size_t
TEigenLapack::comp_decomp ( const TMatrix * K,
                             TDenseMatrix  * D,
                             TDenseMatrix  * Z ) const
{
    //////////////////////////////////////////////////////////////////////////
    //
    // Do some consistency checks
    //
    //////////////////////////////////////////////////////////////////////////

    if ( K == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigen_arpack) comp_decomp", "argument is nullptr" );
        
    if ( !K->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "matrix is not symmetric" );    

    const size_t  n = K->rows();
    
    if ( n != K->cols() )
        HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "matrix not quadratic" ); 
        
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Handle special case of zero sized matrices
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    if ( K->rows() * K->cols() == 0 )
    {
        D->set_size( 0, 0 );
        Z->set_size( 0, 0 );
        
        D->set_ofs( K->col_ofs(), K->col_ofs() );
        Z->set_ofs( K->col_ofs(), K->col_ofs() );
        
        return 0;
    }// if
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Test that the input matrix K is symmetric
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    
    if ( _test_pos_def )
    {    
        if ( ! EIGEN_MISC::is_symmetric( K ) )
            HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "K is not symmetric" ); 
    }// if
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Copy the input matrix information of the corresponding BLAS data structure
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    BLAS::Matrix< real >  KMatrix ( n, n );
    BLAS::Matrix< real >  e_vec   ( n, n );
    BLAS::Vector< real >  e_val   ( n );
        
    //---------------------------------------------------------------------------------------------------
    // NOTE: Only the lower half of K is accessed by the subsequent used LAPACK eigensolver.
    //       
    //       If K is a block matrix it is actually relative expensive to extract the matrix entries like
    //       this. If K is a dense matrix this is much cheaper. Probably it takes much
    //       time to find position of each entry in main memory when K has many different blocks.
    //---------------------------------------------------------------------------------------------------
    for ( size_t j = 0; j < n; j++ )
    {
        for ( size_t i = j; i < n; i++ )
        {
            KMatrix( i, j ) = K->entry( i, j );
            KMatrix( j, i ) = K->entry( i, j );
        }// for
    }// for

    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the selected eigenvalues and eigenvectors
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    size_t n_ev;
    
    if ( _ev_selection == EV_SEL_FULL )
    {
        n_ev = n;
         
        //
        // Compute all eigenvalues/eigenvectors
        //
        BLAS::eigen( KMatrix, e_val, e_vec );   
    }// if
    else if ( _ev_selection == EV_SEL_INDEX )
    {
        //
        // The 'n' eigenvalues of the problem are numbered from '1' to 'n' beginning 
        // with the smallest eigenvalue. The class 'Range' is based on the class 
        // 'TIndexset' and those indices are indicated by 0,...,n-1
        //
        BLAS::Range eig_range( _lindex-1, _uindex-1 );
    
        //
        // Compute all eigenvalues/eigenvectors with the index = lindex,...,uindex
        //
        BLAS::eigen( KMatrix, eig_range, e_val, e_vec );   
        
        n_ev = e_vec.ncols();
    }// else if
    else if ( _ev_selection == EV_SEL_BOUND )
    {
        HERROR( ERR_ARG, "(TEigenLapack) comp_decomp_lapack", "not implemented" );
    
        // Compute all eigenvalues/eigenvectors where the eigenvalues 
        // are in the half-open interval ( _lbound, _ubound ]
//         n_ev = BLAS::eigen( KMatrix,_lbound, _ubound, e_val, e_vec );

//         n_ev = e_vec.ncols();
    }// else if
    else
        HERROR( ERR_ARG, "(TEigenLapack) comp_decomp_lapack", "" );
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Adapt the size and the offset of the matrices D and Z in order that the matrix 
    // operation K*Z-Z*D is well defined and copy the solution to the output data
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    D->set_size( n_ev, n_ev );
    Z->set_size( n   , n_ev );
    
    D->set_ofs( K->col_ofs(), K->col_ofs() );
    Z->set_ofs( K->col_ofs(), K->col_ofs() );
    
    D->scale( real(0) );
    
    for ( size_t j = 0; j < n_ev; j++ )
        D->set_entry( j, j, e_val( j ) );
    
    for ( size_t j = 0; j < n_ev; j++ )
    {
        for ( size_t i = 0; i < n; i++ )
        {
            Z->set_entry( i, j, e_vec( i, j ) );
        }// for
    }// for
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Test results if wanted
    //
    //
    ///////////////////////////////////////////////////////////////////////////////////////    
    
    if ( _test_residuals )
    {
        TEigenAnalysis  eigen_analysis;
            
        eigen_analysis.set_verbosity( 0 );
        
        const bool large_errors_detected = eigen_analysis.analyse_vector_residual ( K, D, Z );
        
        if ( large_errors_detected )
        {
//             eigen_analysis.set_verbosity( 3 );
//         
//             eigen_analysis.analyse_vector_residual ( K, D, Z );
        
            HERROR( ERR_NCONVERGED, "(TEigenLapack) comp_decomp", "large errors detected" );
        }// if
    }// if
        
    return n_ev;
}
 






size_t
TEigenLapack::comp_decomp ( const TMatrix * K,
                             const TMatrix * M,
                             TDenseMatrix  * D,
                             TDenseMatrix  * Z ) const
{
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Do some consistency checks
    //
    ///////////////////////////////////////////////////////////////////////////////////////

    if ( M == nullptr || K == nullptr || D == nullptr || Z == nullptr )
        HERROR( ERR_ARG, "(TEigenLapack) comp_decomp", "argument is nullptr" );
        
    if ( !K->is_symmetric() || !M->is_symmetric() )
        HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "matrices are not symmetric" );
        
    const size_t  n = K->rows();
    
    if ( n != M->rows() || n != K->cols() || n != M->cols() )
        HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "matrices have different size" ); 
        
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Handle special case of zero sized matrices
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    if ( K->rows() * K->cols() * M->rows() * M->cols() == 0 )
    {
        D->set_size( 0, 0 );
        Z->set_size( 0, 0 );
        
        D->set_ofs( K->col_ofs(), K->col_ofs() );
        Z->set_ofs( K->col_ofs(), K->col_ofs() );
        
        return 0;
    }// if
        
        
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Verify that  matrix 'M' is positive definit. This is needed for the LAPACK eigensolver 
    // (cf. manuel of LAPACK routine dsygv or [Templates for solution of Algebraic EVP]
    //
    // Test as well that the input matrices K and M are symmetric.
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    
    if ( _test_pos_def )
    {    
        if ( ! EIGEN_MISC::is_symmetric( K ) )
            HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "K is not symmetric" ); 
    
        if ( ! EIGEN_MISC::is_symmetric( M ) )
            HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "M is not symmetric" ); 
    
        if ( ! EIGEN_MISC::is_pos_def( M ) )
            HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "M is not positive definit" ); 
    }// if
        
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Copy the input matrix information of the corresponding BLAS data structure
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    BLAS::Matrix< real >  KMatrix ( n, n );
    BLAS::Matrix< real >  MMatrix ( n, n );    
    BLAS::Matrix< real >  e_vec   ( n, n );
    BLAS::Matrix< real >  cholesky( n, n );
    BLAS::Vector< real >  e_val   ( n );
    

    //---------------------------------------------------------------------------------------------------
    // NOTE: Only the lower half of K and M is accessed by the subsequent used LAPACK eigensolver.
    //
    //       If K and M are block matrices it is actually relative expensive to extract the matrix 
    //       entries like this. If K and M are dense matries this is much cheaper. Probably it takes much
    //       time to find position of each entry in main memory when K and M have many different blocks
    //---------------------------------------------------------------------------------------------------
    for ( size_t j = 0; j < n; j++ )
    {
        for ( size_t i = j; i < n; i++ )
        {
            KMatrix( i, j ) = K->entry( i, j );
            MMatrix( i, j ) = M->entry( i, j );
            KMatrix( j, i ) = K->entry( i, j );
            MMatrix( j, i ) = M->entry( i, j );
        }// for
    }// for

    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Compute the selected eigenvalues and eigenvectors
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t n_ev;
    
    if ( _ev_selection == EV_SEL_FULL )
    {
        n_ev = n;
         
        //
        // Compute all eigenvalues/eigenvectors
        //
        eigen( KMatrix, MMatrix, e_val, e_vec, cholesky );   
    }// if
    else if ( _ev_selection == EV_SEL_INDEX )
    {
    
        if ( idx_t(n) < _uindex )
            HERROR( ERR_CONSISTENCY, "(TEigenLapack) comp_decomp", "upper index is too large" ); 
    
        //
        // The 'n' eigenvalues of the problem are numbered from '1' to 'n' 
        // beginning with the smallest eigenvalue. The class 'Range' is
        // based on the class 'TIndexset' and those indices are indicated
        // by 0,...,n-1
        //
        BLAS::Range eig_range( _lindex-1, _uindex-1 );
    
        //
        // Compute all eigenvalues/eigenvectors with the index = lindex,...,uindex
        //
        n_ev = eigen( KMatrix, MMatrix, eig_range, e_val, e_vec, cholesky );   
    }// else if
    else if ( _ev_selection == EV_SEL_BOUND )
    {
        //
        // Compute all eigenvalues/eigenvectors where the eigenvalues 
        // are in the half-open interval ( _lbound, _ubound ]
        //
        n_ev = eigen( KMatrix, MMatrix, _lbound, _ubound, e_val, e_vec, cholesky );
    }// else if
    else 
        HERROR( ERR_ARG, "(TEigenLapack) comp_decomp_lapack", "" );

    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // Adapt the size and the offset of the matrices D and Z in order that the matrix 
    // operation K*Z-M*Z*D is well defined and copy the solution to the output data
    //
    ///////////////////////////////////////////////////////////////////////////////////////
    D->set_size( n_ev, n_ev );
    Z->set_size( n   , n_ev );
    
    D->set_ofs( K->col_ofs(), K->col_ofs() );
    Z->set_ofs( K->col_ofs(), K->col_ofs() );
    
    D->scale( real(0) );
    
    for ( size_t j = 0; j < n_ev; j++ )
        D->set_entry( j, j, e_val( j ) );
    
    for ( size_t j = 0; j < n_ev; j++ )
    {
        for ( size_t i = 0; i < n; i++ )
        {
            Z->set_entry( i, j, e_vec( i, j ) );
        }// for
    }// for
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    //
    // Test results if wanted
    //
    //
    ///////////////////////////////////////////////////////////////////////////////////////    
    
    if ( _test_residuals )
    {
        TEigenAnalysis  eigen_analysis;
            
        eigen_analysis.set_verbosity( 0 );
        
        const bool large_errors_detected = eigen_analysis.analyse_vector_residual ( K, M, D, Z );
        
        if ( large_errors_detected )
        {
//             eigen_analysis.set_verbosity( 3 );
//         
//             eigen_analysis.analyse_vector_residual ( K, M, D, Z );
            
            HERROR( ERR_NCONVERGED, "(TEigenLapack) comp_decomp", "large errors detected" );
        }// if
    }// if
            
    return n_ev;
}




}// namespace
