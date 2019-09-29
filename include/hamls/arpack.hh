#ifndef __HAMLS_ARPACK_HH
#define __HAMLS_ARPACK_HH

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
// File        : arpack.hh
// Description : definition of ARPACK functions in c-format (model was lapack.hh)
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

namespace HAMLS
{

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// definition of external ARPACK functions
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

extern "C" {
    
//////////////////////////////////////////////////////////////
//
// real-valued functions (single precision)
//
//////////////////////////////////////////////////////////////


// solve the standard/generalized symmetric eigenvalue problem 
CFUNCDECL
void ssaupd_ ( int *          ido, 
               const char *   bmat, 
               const int *    n, 
               const char *   which,
               const int *    nev, 
               const float *  tol, 
               float *        resid, 
               const int *    ncv,
               float *        v, 
               const int *    ldv, 
               int *          iparam, 
               int *          ipntr,
               float *        workd, 
               float *        workl, 
               const int *    lworkl,
               int *          info );

// post processing routine for the eigenvalue problem solved by ssaupd_
CFUNCDECL
void sseupd_ ( const int *    rvec, 
               const char *   All, 
               int *          select, 
               float *        d,
               float *        z, 
               const int *    ldz, 
               const float *  sigma, 
               const char *   bmat, 
               const int *    n, 
               const char *   which, 
               const int *    nev,
               const float *  tol, 
               float *        resid, 
               const int *    ncv, 
               float *        v,
               const int *    ldv, 
               int *          iparam, 
               int *          ipntr, 
               float *        workd,
               float *        workl, 
               const int *    lworkl, 
               int *          ierr );


//////////////////////////////////////////////////////////////
//
// real-valued functions (double precision)
//
//////////////////////////////////////////////////////////////


// solve the standard/generalized symmetric eigenvalue problem 
CFUNCDECL
void dsaupd_ ( int *           ido, 
               const char *    bmat, 
               const int *     n, 
               const char *    which,
               const int *     nev, 
               const double *  tol, 
               double *        resid, 
               const int *     ncv,
               double *        v, 
               const int *     ldv, 
               int *           iparam, 
               int *           ipntr,
               double *        workd, 
               double *        workl, 
               const int *     lworkl,
               int *           info );

// post processing routine for the eigenvalue problem solved by dsaupd_
CFUNCDECL
void dseupd_ ( const int *     rvec, 
               const char *    All, 
               int *           select, 
               double *        d,
               double *        z, 
               const int *     ldz, 
               const double *  sigma, 
               const char *    bmat, 
               const int *     n, 
               const char *    which, 
               const int *     nev,
               const double *  tol, 
               double *        resid, 
               const int *     ncv, 
               double *        v,
               const int *     ldv, 
               int *           iparam, 
               int *           ipntr, 
               double *        workd,
               double *        workl, 
               const int *     lworkl, 
               int *           ierr );               


//////////////////////////////////////////////////////////////
//
// complex-valued functions (single precision)
//
//////////////////////////////////////////////////////////////

         
               

//////////////////////////////////////////////////////////////
//
// complex-valued functions (double precision)
//
//////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////
//
// misc. helpers
//
//////////////////////////////////////////////////////////////


}// extern "C"



//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// template wrappers for ARPACK functions (used in TEigen_arpack.cc)
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

namespace ARPACK
{

namespace
{



//
// *saupd (template version of 'dsaupd' and 'ssaupd' used in HLIBpro)
//
template <typename T>  
void saupd ( int &         ido, 
             const char *  bmat, 
             const int     n, 
             const char *  which,
             const int     nev, 
             const T       tol, 
             T *           resid, 
             const int     ncv,
             T *           v, 
             const int     ldv, 
             int *         iparam, 
             int *         ipntr,
             T *           workd, 
             T *           workl, 
             const int     lworkl,
             int &         info );
               
#define SAUPD_FUNC( type, func )                              \
    template <>  void saupd<type> ( int &         ido,        \
                                    const char *  bmat,       \
                                    const int     n,          \
                                    const char *  which,      \
                                    const int     nev,        \
                                    const type    tol,        \
                                    type *        resid,      \
                                    const int     ncv,        \
                                    type *        v,          \
                                    const int     ldv,        \
                                    int *         iparam,     \
                                    int *         ipntr,      \
                                    type *        workd,      \
                                    type *        workl,      \
                                    const int     lworkl,     \
                                    int &         info ) {    \
        func( & ido, bmat, & n, which, & nev, & tol, resid, & ncv, v, & ldv,\
              iparam, ipntr, workd, workl, & lworkl,  & info ); }

SAUPD_FUNC( float,  ssaupd_ )
SAUPD_FUNC( double, dsaupd_ )

#undef SAUPD_FUNC



//
// *seupd (template version of 'dsaupd' and 'ssaupd' used in HLIBpro)
//
template <typename T>  
void seupd ( const int     rvec, 
             const char *  All, 
             int *         select, 
             T *           d,
             T *           z, 
             const int     ldz, 
             const T       sigma, 
             const char *  bmat, 
             const int     n, 
             const char *  which,
             const int     nev, 
             const T       tol, 
             T *           resid, 
             const int     ncv,
             T *           v, 
             const int     ldv, 
             int *         iparam, 
             int *         ipntr,
             T *           workd, 
             T *           workl, 
             const int     lworkl,
             int &         info );
             
#define SEUPD_FUNC( type, func )                              \
    template <>  void seupd<type> ( const int     rvec,       \
                                    const char *  All,        \
                                    int *         select,     \
                                    type *        d,          \
                                    type *        z,          \
                                    const int     ldz,        \
                                    const type    sigma,      \
                                    const char *  bmat,       \
                                    const int     n,          \
                                    const char *  which,      \
                                    const int     nev,        \
                                    const type    tol,        \
                                    type *        resid,      \
                                    const int     ncv,        \
                                    type *        v,          \
                                    const int     ldv,        \
                                    int *         iparam,     \
                                    int *         ipntr,      \
                                    type *        workd,      \
                                    type *        workl,      \
                                    const int     lworkl,     \
                                    int &         info ) {    \
        func( & rvec, All, select, d, z, & ldz, & sigma, \
              bmat, & n, which, & nev, & tol, resid, & ncv, v, & ldv,\
              iparam, ipntr, workd, workl, & lworkl,  & info ); }

SEUPD_FUNC( float,  sseupd_ )
SEUPD_FUNC( double, dseupd_ )

#undef SEUPD_FUNC


}// namespace anonymous

}// namespace ARPACK

}// namespace HAMLS

#endif // __HAMLS_ARPACK_HH
