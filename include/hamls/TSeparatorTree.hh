#ifndef __HAMLS_TSEPARATORTREE_HH
#define __HAMLS_TSEPARATORTREE_HH

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
// Description : This class represents the so called 'separator tree' which is used in the
//               implementation of the AMLS algorithm by [Gao et al.]. Each node in this tree
//               represents a subproblem of the AMLS algorithm which is here identified by an 
//               index. The descendant and ancestor relations between the nodes in the separator 
//               tree are modelling the dependencies between the subproblems in the AMLS algorithm. 
// Author      : Peter Gerds
// Copyright   : Peter Gerds, 2019. All Rights Reserved.

#include <vector>
#include <map>

#include "cluster/TCluster.hh"
#include "cluster/TNodeSet.hh"
#include "cluster/TClusterTree.hh"

namespace HAMLS
{

using namespace HLIB;

//!
//! \ingroup  HAMLS_Module
//! \class    TSeparatorTree
//! \brief    class modelling the 'Separator Tree' 
//!
class TSeparatorTree
{
public:
    // This datatype describes the associated ancestor set associated to a cluster
    typedef std::pair< const TCluster *, TIndexSet > cluster_idxset_t;    

    // This datatype allows an efficient access to a the ancestor set of an given cluster
    typedef std::map< TIndexSet, cluster_idxset_t, TIndexSet::map_cmp_t > cluster_idxset_map_t;

private:

    //! auxiliary array containing all the information which are necessary to 
    //! derive the ancestor and descendant relation between the subproblems
    std::vector< cluster_idxset_t >  _data;
    
    //! array containing the ancestor sets of each subproblem 
    std::vector< TNodeSet >          _ancestor_set;
    
    //! array containing the descendant sets of each subproblem 
    std::vector< TNodeSet >          _descendant_set;
    
    //! array containing information if the eigensolution associated to the subproblem is exact
    std::vector< bool >              _exact;
public:
    ///////////////////////////////////////////////
    //
    // constructor and destructor
    //

    //! construct empty TSeparatorTree 
    TSeparatorTree () {}

    //! construct TSeparatorTree from the cluster \a root_amls_ct representing 
    //! the domain substructuring applied in AMLS
    //! Note: Existence of the separator tree depends on existence of \a root_amls_ct, 
    //! i.e., do not delete \a root_amls_ct before deleting the separator tree
    TSeparatorTree ( const TCluster * root_amls_ct );
    
    //! dtor
    virtual ~TSeparatorTree () {}
    
    ////////////////////////////////////////////////////////
    //
    // misc. methods
    //
    
    //! Return the number of subproblems in the AMLS algorithm
    size_t n_subproblems () const { return _data.size(); }
     
    //! Return the number of domain subproblems in the AMLS algorithm
    size_t n_domain_subproblems () const;
    
    //! Return the number of interface subproblems in the AMLS algorithm
    size_t n_interface_subproblems () const;
    
    //! Return the depth of the Separator Tree
    size_t depth () const;
    
    //! Return the 'degrees of freedom' belonging to the subproblem \a i
    TIndexSet get_dof_is ( const idx_t  i ) const;
        
    //! Return true if the subproblem \a i is an descendant of the subproblem \a j
    bool is_descendant ( const idx_t  i, 
                         const idx_t  j ) const;
    
    //! Return true if the subproblem \a i represents a 'domain' problem
    bool is_domain ( const idx_t  i ) const;
    
    //! Return true if the indexset \a is represents a subdomain or an interface problem
    bool is_subproblem ( const TIndexSet is ) const; 
    
    //! Return true if the matrix block associated to the indices \a i and \a j is zero
    bool matrix_block_is_zero ( const idx_t  i,
                                const idx_t  j ) const;
                                 
    //! Return true if the eigensolution associated to the subproblem with index \a i is exact
    bool has_exact_eigensolution ( const idx_t  i ) const { return _exact[i]; }
    
    //! Set true if the eigensolution associated to the subproblem
    //! with index \a i is exact otherwise set false
    void set_exact_eigensolution ( const idx_t i, const bool is_exact) { _exact[i] = is_exact; }
                                   
    //! Get the index of the subproblem which is associated with the index set \a is
    idx_t get_index ( const TIndexSet is ) const;
    
};

}// namespace HAMLS

#endif  // __HAMLS_TSEPARATORTREE_HH
