Neutron Diffusion Solution for the One-Dimensional Steady State Equation.

To Do
======================================
 - Block submatrix solve of Ax = b --> prevent building A (Gauss Seidel)
 - Fix dictionary problem when the same material is used more than once in a problem
 - Create a graphing class (inheritance)
 - Time Dependency
 - All the PyTest functions
 - Be able to run 618 group problem with I = 400

Reference
======================================
 - computational\_engineering.py: Reference code from Ryan G. McClarren's book "Computational Nuclear Engineering and Radiological Science Using Python".
 - diffusion.py: The main code currently used with C-function speed-up.
 - diffusion\_lambda.py: The original multi-group diffusion problem written in python.

Functions for Graphing Class
======================================
 - Convergence vs Iteration
 - Critical, subcritical - integrate flux over energy vs position
 - Different fixed spatial points flux vs energy
 - Theoretical memory vs Rank (error Keff)
