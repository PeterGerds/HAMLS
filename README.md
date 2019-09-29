# HAMLS 

### Eigensovler Module for HLIBpro's Hierarchical Matrices and Elliptic PDE Eigenvalue Problems


This software implements the so called *H-AMLS* method which has been introduced in  (Gerds 2017; Gerds and Grasedyck 2015). H-AMLS is a very efficient method for the solution of elliptic PDE eigenvalue problems. The method is combining a recursive version of the *automated multi-level substructuring* (short AMLS) method - which is a domain decomposition technique for the solution of elliptic PDE eigenvalue problems - with the concept of *hierarchical matrices* (short *H-matrices*). H-AMLS allows the efficient solution of large scale eigenvalue problems with millions degrees of freedom on today's workstations. In particular, the method is well suited for problems posed in three dimensions and allows the computation of a large amount of eigenpair approximations in optimal complexity.

Besides the implementation of the H-AMLS method, this software provides classical iterative eigensolvers combined with the fast H-matrix arithmetic. The software allows to solve the generalized eigenvalue problem *K x = c M x* where *K* and *M* are symmetric sparse or H-matrices, and where *M* is positive definite.

This software requires an installed [HLIBpro library](https://www.hlibpro.com/ "HLIBpro's homepage"), and in order to use all provided iterative eigensolvers an installed [ARPACK library](https://github.com/opencollab/arpack-ng "ARPACK distribution on GitHub") is required as well. As this software is based on the HLIBpro library it is recommended to be familiar with the basic operation of HLIBpro.

An **introduction to this software including installation instructions** can be found in this package under */documentation/hamls_documentation.html*.

This software is licensed under the MIT License. For questions or needed support contact peter_gerds@gmx.de. 

---

######
*Gerds, Peter. 2017. “Solving an elliptic PDE eigenvalue problem via automated multi-level substructuring and hierarchical matrices.” Dissertation, RWTH Aachen University; Fakultät für Mathematik, Informatik und Naturwissenschaften der RWTH Aachen University. https://doi.org/10.18154/RWTH-2017-10520.*

######
*Gerds, Peter, and Lars Grasedyck. 2015. “Solving an Elliptic Pde Eigenvalue Problem via Automated Multi-Level Substructuring and Hierarchical Matrices.” Computing and Visualization in Science 16 (6): 283–302. https://doi.org/10.1007/s00791-015-0239-x.*
