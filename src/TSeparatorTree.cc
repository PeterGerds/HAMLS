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
// File        : TSeparatorTree.hh
// Description : This class represents the so called 'Separator Tree'. This tree is derived from
//               a cluster tree representing a domain decomposition obtained by nested dissection
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include "hamls/TSeparatorTree.hh"

namespace HAMLS
{

using std::unique_ptr;
using std::vector;
using std::pair;
using std::map;
using std::list;

namespace
{
    
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//
// local functions
//
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


    
}// namespace anonymous
    

    


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//
// TSeparatorTree
//
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////



//
// Constructor
// 
TSeparatorTree::TSeparatorTree ( const TCluster * root_amls_ct )
{
    if ( root_amls_ct == nullptr )
        HERROR( ERR_ARG, "(TSeparatorTree) TSeparatorTree", "argument is nullptr" );
          
    list< const TCluster * >  cluster_list;
    cluster_idxset_map_t      subproblem_map;
    const TCluster          * cluster;
   
    ///////////////////////////////////////////////////////////////////////////////////////////
    //
    // The input cluster represents the domain substructing applied in the AMLS method.
    // That means all leaf clusters of the input cluster either represent a subdomain 
    // subproblem or an interface subproblem. In the following all these leaf clusters 
    // are collected in a list and their ancestor descendant relationship is collected. 
    // Because we want to save this relationship between the clusters it is not sufficient 
    // to just call the routine 'TCluster::collect_leaves' 
    //
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    cluster_list.push_back( root_amls_ct ); 
    
    while ( !cluster_list.empty() )
    {
        cluster = cluster_list.front();

        cluster_list.pop_front();
            
        size_t nsons = cluster->nsons();
        
        
        ///////////////////////////////////////////////
        //
        //
        // Handle the different cases which can occur
        //
        //
        ///////////////////////////////////////////////
        
        
        if ( cluster->is_leaf() )
        {
            // If this case happens the cluster should represent a subdomain or the domain 
            // problem which has no descendants (empty descendant set) in the separator tree
            
            const TIndexSet descendant_set;
            
            cluster_idxset_t pair( cluster, descendant_set );
                        
            subproblem_map[*cluster] = pair;
        }// if
        else if ( nsons == 2 )
        {
            // If this case happens the cluster is divided into two subdomains
            // which are seperated in a natural way i.e. without the need of 
            // an additional interface cluster
        
            if ( cluster->son(0)->is_domain() && cluster->son(1)->is_domain() )
            {
                cluster_list.push_back( cluster->son(0) );
                cluster_list.push_back( cluster->son(1) );
            }// if
            else
                HERROR( ERR_CONSISTENCY, "(TSeparatorTree) TSeparatorTree", "" );
        }// else if
        else if ( nsons == 3 )
        {
            if ( cluster->son(0)->is_domain() && cluster->son(1)->is_domain() && !cluster->son(2)->is_domain() )
            {
                // If this case happens the cluster is separated by the classical nested dissection,
                // two subdomain separated by an interface. Append the subdomain clusters to the 
                // cluster list and create the descendant set of the interface. The descendant set of 
                // the interface consists of the index sets of the two subdomains but doesn't include
                // the index set of the interface because only the clusters belonging to the subdomains
                // are descendants of the interface in the separator tree.
            
                cluster_list.push_back( cluster->son(0) );
                cluster_list.push_back( cluster->son(1) );
                
                const TIndexSet dof_is_0( *(cluster->son(0)) );
                const TIndexSet dof_is_1( *(cluster->son(1)) );
                
                const TIndexSet descendant_set( join( dof_is_0, dof_is_1 ) );
                
                cluster_idxset_t pair( cluster->son(2), descendant_set );
                            
                subproblem_map[*(cluster->son(2))] = pair;
            }// if
            else 
                HERROR( ERR_CONSISTENCY, "(TSeparatorTree) TSeparatorTree", "" );    
        }// else if
        else 
            HERROR( ERR_CONSISTENCY, "(TSeparatorTree) TSeparatorTree", "" );
    }// while
    
    ////////////////////////////////////////////////////////////////////////
    //
    // Copy the subproblem information in the auxiliary array and arrange
    // the subproblem information according to the indexsets containing
    // the degrees of freedom of each subproblem.
    //
    ////////////////////////////////////////////////////////////////////////
    
    const size_t n_subprobs = subproblem_map.size(); 
    
    _data.resize( n_subprobs );
    
    size_t count = 0;
    
    for ( cluster_idxset_map_t::const_iterator it = subproblem_map.begin(); it != subproblem_map.end(); it++ )
    {
        _data[count] = it->second;
        
        count++;
    }// for
    
    //////////////////////////////////////////////////////////////////////
    //
    // Compute the ancestor and descendants node sets of each subproblem
    //
    //////////////////////////////////////////////////////////////////////
    
    //
    // Initialize the ancestor and descendants node sets
    //
    _ancestor_set.resize  ( n_subprobs );
    _descendant_set.resize( n_subprobs );
    
    for ( size_t i = 0; i < n_subprobs; i++ )
    {
        _ancestor_set  [i] = TNodeSet( n_subprobs ); 
        _descendant_set[i] = TNodeSet( n_subprobs );
    }// for
    
    //
    // Compute the ancestor and descendant set for each subproblem
    //
    for ( size_t i = 0; i < n_subprobs; i++ )
    {
        for ( size_t j = 0; j < n_subprobs; j++ )
        {
            if ( is_descendant( j, i ) )
            {
                _descendant_set[i].append( node_t( j ) );
                _ancestor_set  [j].append( node_t( i ) );
            }// if
        }// for
    }// for
    
    
    //////////////////////////////////////////////////////////////////////
    //
    // Initialise the auxiliary data if the eigensolution associated 
    // to the subproblem is exact or not
    //
    //////////////////////////////////////////////////////////////////////
    
    _exact.resize( n_subprobs );
    
    for ( size_t i = 0; i < n_subprobs; i++ )
        _exact[i] = false;
}




TIndexSet
TSeparatorTree::get_dof_is ( const idx_t i ) const
{
    const TCluster * cluster = _data[i].first;

    return TIndexSet( cluster->first(), cluster->last() );
}

bool 
TSeparatorTree::is_domain ( const idx_t i ) const
{
    const TCluster * cluster = _data[i].first;
    
    return cluster->is_domain();
} 

bool 
TSeparatorTree::is_descendant ( const idx_t i, 
                                const idx_t j ) const
{

    TIndexSet dof_is = get_dof_is( i );
    
    TIndexSet descendant_set( _data[j].second );
    
    return descendant_set.is_sub( dof_is );
}

size_t
TSeparatorTree::depth () const
{
    size_t max_depth, temp_depth;
     
    max_depth = 0;
    
    for ( size_t i = 0; i < n_subproblems(); i++ )
    {   
        const TNodeSet set = _ancestor_set[i];
        
        temp_depth = set.nnodes();
        
        if ( max_depth < temp_depth )
            max_depth = temp_depth;
    }
    
    return max_depth;
}
    
    
size_t 
TSeparatorTree::n_domain_subproblems () const
{
    size_t count = 0;
    
    for ( size_t i = 0; i < n_subproblems(); i++ )
    {
        if ( is_domain( i ) )
            count++;
    }// for
    
    return count;
}
    
    
 


size_t 
TSeparatorTree::n_interface_subproblems () const
{
    size_t count = 0;
    
    for ( size_t i = 0; i < n_subproblems(); i++ )
    {
        if ( ! is_domain( i ) )
            count++;
    }// for
    
    return count;
}



    
bool 
TSeparatorTree::is_subproblem ( const TIndexSet is ) const
{
    for ( size_t i = 0; i < n_subproblems(); i ++ )
    {
        const TIndexSet subproblem_is = get_dof_is( i );
        
        if ( is == subproblem_is )
            return true;
    }// for
    
    return false;
}

   
bool
TSeparatorTree::matrix_block_is_zero ( const idx_t i,
                                       const idx_t j ) const
{
    bool block_is_zero = true;
                     
    if ( is_descendant(j,i) || i == j || is_descendant(i,j) )
        block_is_zero = false;
        
    return block_is_zero;
}
                                   

idx_t
TSeparatorTree::get_index( const TIndexSet is ) const
{
    for ( size_t i = 0; i < n_subproblems(); i++ )
    {
        const TIndexSet subproblem_is = get_dof_is(i);
        
        if ( subproblem_is == is )
            return i;
    }// for
    
    HERROR( ERR_CONSISTENCY, "(TSeparatorTree) get_index", "" );
}




}// namespace HAMLS
