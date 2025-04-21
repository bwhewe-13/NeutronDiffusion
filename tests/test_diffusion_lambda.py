"""
Notes about output
=============================================================================
- The B matrix in A x = (1/k) B x is constructed in two ways, either initially
or the B x vector is constructed each step. The tests ending in 'B' construct 
the B matrix beforehand while the tests ending in 'Bx' construct the vector 
each step.

- There is an option to construct the A and B matrices using C functions. The
tests that use these C functions are in the test_diffusion_lambda_fast.py file.

"""

from NeutronDiffusion import diffusion_lambda as df

import numpy as np
import pytest


def test_Diffusion_1G_1mat_slab_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 50
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'slab')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-5 )
 
def test_Diffusion_1G_1mat_slab_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 50
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'slab')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-5 )

def test_Diffusion_1G_1mat_cylinder_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 76.5535
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'cylinder')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-4 )
 
def test_Diffusion_1G_1mat_cylinder_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 76.5535
    I = 50
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'cylinder')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_1G_1mat_sphere_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 100
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-4 )
 
def test_Diffusion_1G_1mat_sphere_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.1570]
    D_func = lambda r: [3.850204978408833]
    Sigmaa_func = lambda r: [0.1532]
    BC = np.array([[1,0]])
    G = 1
    R = 100
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_1G_2mat_slab_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 100
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'slab')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff_exact = 1.2955
    book_keff_approx = 1.29524
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_Diffusion_1G_2mat_slab_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 100
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'slab')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff_exact = 1.2955
    book_keff_approx = 1.29524
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_Diffusion_1G_2mat_cylinder_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 100
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'cylinder')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff_exact = 1.14147
    book_keff_approx = 1.14068
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_Diffusion_1G_2mat_cylinder_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 100
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'cylinder')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff_exact = 1.14147
    book_keff_approx = 1.14068
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_Diffusion_1G_2mat_sphere_B():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 150
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff_exact = 0.95888
    book_keff_approx = 0.95735
    assert( abs(k - book_keff_approx) < 2.0e-3 )

def test_Diffusion_1G_2mat_sphere_Bx():
    Sigmas_func = lambda r: [[0.0]]
    chi_func = lambda r: [1.0]
    nuSigmaf_func = lambda r: [0.7*(r<=5) + 0.0*(r>5)]
    D_func = lambda r: [5.0*(r<=5) + 1.0*(r>5)]
    Sigmaa_func = lambda r: [0.5*(r<=5) + 0.01*(r>5)]
    BC = np.array([[1,0]])
    G = 1
    R = 10
    I = 100
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff_exact = 0.95888
    book_keff_approx = 0.95735
    assert( abs(k - book_keff_approx) < 2.0e-3 )

def test_Diffusion_2G_1mat_sphere_keff_B():
    G = 2
    Sigmas_func = lambda r: np.zeros((G,G))
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.1570]*G
    D_func = lambda r: [3.850204978408833]*G
    Sigmaa_func = lambda r: [0.1532]*G
    BC = np.array([[1,0.0],[1,0.0]])
    R = 100
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.00002955111
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_2G_1mat_sphere_keff_Bx():
    G = 2
    Sigmas_func = lambda r: np.zeros((G,G))
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.1570]*G
    D_func = lambda r: [3.850204978408833]*G
    Sigmaa_func = lambda r: [0.1532]*G
    BC = np.array([[1,0.0],[1,0.0]])
    R = 100
    I = 20
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmaa_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.00002955111
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_2G_1mat_sphere_kinf_B():
    G = 2
    Sigmas_func = lambda r: np.array([[0.0,0.0],[0.0241,0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.0085,0.185]
    D_func = lambda r: [0.1,0.1]
    Sigmar_func = lambda r: [0.0362,0.121]
    BC = np.array([[0.0,1],[0.0,1]])
    R = 5
    I = 50
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.25268252483
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_2G_1mat_sphere_kinf_Bx():
    G = 2
    Sigmas_func = lambda r: np.array([[0.0,0.0],[0.0241,0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.0085,0.185]
    D_func = lambda r: [0.1,0.1]
    Sigmar_func = lambda r: [0.0362,0.121]
    BC = np.array([[0.0,1],[0.0,1]])
    R = 5
    I = 50
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.25268252483
    assert( abs(k - book_keff) < 1.0e-4 )

def test_Diffusion_2G_1mat_sphere_flux_B():
    G = 2
    Sigmas_func = lambda r: np.array([[0.0,0.0],[0.0241,0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.0085,0.185]
    D_func = lambda r: [0.1,0.1]
    Sigmar_func = lambda r: [0.0362,0.121]
    BC = np.array([[0.0,1],[0.0,1]])
    R = 5
    I = 50
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_flux_ratio = 5.021
    assert( abs((phi[0,0] / phi[0,1]) - book_flux_ratio) < 1.0e-3 )

def test_Diffusion_2G_1mat_sphere_flux_Bx():
    G = 2
    Sigmas_func = lambda r: np.array([[0.0,0.0],[0.0241,0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.0085,0.185]
    D_func = lambda r: [0.1,0.1]
    Sigmar_func = lambda r: [0.0362,0.121]
    BC = np.array([[0.0,1],[0.0,1]])
    R = 5
    I = 50
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_flux_ratio = 5.021
    assert( abs((phi[0,0] / phi[0,1]) - book_flux_ratio) < 1.0e-3 )

def test_Diffusion_2G_2mat_sphere_refl_B():
    G = 2
    R_reac = 50.0
    Sigmas_func = lambda r: np.array([[0.0,0.0],
                                      [0.001*(r<=R_reac) + 0.009*(r>R_reac),0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.00085*(r<=R_reac) + 0.0, 0.057*(r<=R_reac) + 0.0]
    D_func = lambda r: [1.0,1.0]
    Sigmar_func = lambda r: [0.01, 0.01*(r<=R_reac) + 0.00049*(r>R_reac)]
    R = 100
    I = 100
    BC = np.array([[0.25,0.5*D_func(R)[0]],[0.25,0.5*D_func(R)[1]]])
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 1.06508498598
    assert( abs( k - book_keff ) < 1.0e-6 )

def test_Diffusion_2G_2mat_sphere_refl_Bx():
    G = 2
    R_reac = 50.0
    Sigmas_func = lambda r: np.array([[0.0,0.0],
                                      [0.001*(r<=R_reac) + 0.009*(r>R_reac),0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.00085*(r<=R_reac) + 0.0, 0.057*(r<=R_reac) + 0.0]
    D_func = lambda r: [1.0,1.0]
    Sigmar_func = lambda r: [0.01, 0.01*(r<=R_reac) + 0.00049*(r>R_reac)]
    R = 100
    I = 100
    BC = np.array([[0.25,0.5*D_func(R)[0]],[0.25,0.5*D_func(R)[1]]])
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 1.06508498598
    assert( abs( k - book_keff ) < 1.0e-6 )

def test_Diffusion_2G_1mat_sphere_refl_B():
    G = 2
    R_reac = 500.0
    Sigmas_func = lambda r: np.array([[0.0,0.0],
                                      [0.001*(r<=R_reac) + 0.009*(r>R_reac),0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.00085*(r<=R_reac) + 0.0, 0.057*(r<=R_reac) + 0.0]
    D_func = lambda r: [1.0,1.0]
    Sigmar_func = lambda r: [0.01, 0.01*(r<=R_reac) + 0.00049*(r>R_reac)]
    R = 50
    I = 100
    BC = np.array([[0.25,0.5*D_func(R)[0]],[0.25,0.5*D_func(R)[1]]])
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A,B = problem.construct_A_B(RHS=True)
    phi,k = problem.solver(A,B,tol=1e-10)
    book_keff = 0.368702897492
    assert( abs( k - book_keff ) < 1.0e-6 )

def test_Diffusion_2G_1mat_sphere_refl_Bx():
    G = 2
    R_reac = 500.0
    Sigmas_func = lambda r: np.array([[0.0,0.0],
                                      [0.001*(r<=R_reac) + 0.009*(r>R_reac),0.0]])
    chi_func = lambda r: [1.0,0.0]
    nuSigmaf_func = lambda r: [0.00085*(r<=R_reac) + 0.0, 0.057*(r<=R_reac) + 0.0]
    D_func = lambda r: [1.0,1.0]
    Sigmar_func = lambda r: [0.01, 0.01*(r<=R_reac) + 0.00049*(r>R_reac)]
    R = 50
    I = 100
    BC = np.array([[0.25,0.5*D_func(R)[0]],[0.25,0.5*D_func(R)[1]]])
    problem = df.Diffusion(G,R,I,D_func,Sigmas_func,chi_func,
            nuSigmaf_func,Sigmar_func,BC,'sphere')
    problem.geometry()
    A = problem.construct_A_B(RHS=False)
    phi,k = problem.solver(A,tol=1e-10)
    book_keff = 0.368702897492
    assert( abs( k - book_keff ) < 1.0e-6 )
 
if __name__ == '__main__':
    print('File works')

