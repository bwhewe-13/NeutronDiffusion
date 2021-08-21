from NeutronDiffusion import computational_engineering as book

import numpy as np
import pytest

def test_LU_factor():
    A = np.array([(3.0,2,1,-2),(-1,4,5,4),(2,-8,10,3),(-2,-8,10,0.1)])
    answer = np.ones(4)
    b = np.dot(A,answer)
    
    row_order = book.LU_factor(A)
    assert( np.array_equal( row_order,np.array([0,2,1,3]) ) )

def test_LU_solve():
    A = np.array([(3.0,2,1,-2),(-1,4,5,4),(2,-8,10,3),(-2,-8,10,0.1)])
    answer = np.ones(4,dtype='double')

    b = np.dot(A,answer)
    
    row_order = book.LU_factor(A)
    x = book.LU_solve(A,b,row_order)
    assert( np.sum(x - answer) == 0. )

def test_DiffusionEigenvalue_1G_1mat_slab():
    nuSigmaf_func = lambda r: 0.1570
    D_func = lambda r: 3.850204978408833
    Sigmaa_func = lambda r: 0.1532
    R = 50
    I = 20
    k,phi,centers = book.DiffusionEigenvalue(R,I,D_func,
            Sigmaa_func,nuSigmaf_func,[1,0],0,epsilon=1.0e-10)
    
    book_keff = 1.00001243892
    assert( abs(k - book_keff) < 1.0e-5 )

def test_DiffusionEigenvalue_1G_1mat_cylinder():
    nuSigmaf_func = lambda r: 0.1570
    D_func = lambda r: 3.850204978408833
    Sigmaa_func = lambda r: 0.1532
    R = 76.5535
    I = 20
    k,phi,centers = book.DiffusionEigenvalue(R,I,D_func,
            Sigmaa_func,nuSigmaf_func,[1,0],1,epsilon=1.0e-10)
    
    book_keff = 1.00002258361
    assert( abs(k - book_keff) < 1.0e-6 )

def test_DiffusionEigenvalue_1G_1mat_sphere():
    nuSigmaf_func = lambda r: 0.1570
    D_func = lambda r: 3.850204978408833
    Sigmaa_func = lambda r: 0.1532
    R = 100
    I = 20
    k,phi,centers = book.DiffusionEigenvalue(R,I,D_func,
            Sigmaa_func,nuSigmaf_func,[1,0],2,epsilon=1.0e-10)
    
    book_keff = 1.00002955108
    assert( abs(k - book_keff) < 1.0e-6 )

def test_DiffusionEigenvalue_1G_2mat_slab():
    D = lambda r: 5.0*(r<=5) + 1.0*(r>5)
    Sigma_a = lambda r: 0.5*(r<=5) + 0.01*(r>5)
    nuSigma_f = lambda r: 0.7*(r<=5) + 0.0*(r>5)
    R = 10
    I = 100
    k,phi,centers = book.DiffusionEigenvalue(R,I,D,Sigma_a,
            nuSigma_f,[1,0],0,epsilon=1.0e-10)
    book_keff_exact = 1.2955
    book_keff_approx = 1.29524
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_DiffusionEigenvalue_1G_2mat_cylinder():
    D = lambda r: 5.0*(r<=5) + 1.0*(r>5)
    Sigma_a = lambda r: 0.5*(r<=5) + 0.01*(r>5)
    nuSigma_f = lambda r: 0.7*(r<=5) + 0.0*(r>5)
    R = 10
    I = 100
    k,phi,centers = book.DiffusionEigenvalue(R,I,D,Sigma_a,
            nuSigma_f,[1,0],1,epsilon=1.0e-10)
    book_keff_exact = 1.14147
    book_keff_approx = 1.14068
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_DiffusionEigenvalue_1G_2mat_sphere():
    D = lambda r: 5.0*(r<=5) + 1.0*(r>5)
    Sigma_a = lambda r: 0.5*(r<=5) + 0.01*(r>5)
    nuSigma_f = lambda r: 0.7*(r<=5) + 0.0*(r>5)
    R = 10
    I = 150
    k,phi,centers = book.DiffusionEigenvalue(R,I,D,Sigma_a,
            nuSigma_f,[1,0],2,epsilon=1.0e-10)
    book_keff_exact = 0.95888
    book_keff_approx = 0.95735
    assert( abs(k - book_keff_approx) < 1.0e-3 )

def test_inversePowerBlock():
    # define A
    M11 = np.identity(2)
    M11[0,0] = 10.0
    M11[1,1] = 0.5
    M22 = np.identity(2)
    M22[1,1] = 0.1
    M21 = -np.identity(2)
    # Define P
    P11 = np.identity(2)
    P12 = np.identity(2)
    l, x1, x2 = book.inversePowerBlock(M11,M21,M22,P11,P12,
            epsilon=1.0e-8,LOUD=False)
    assert ( abs(l - (1/22)) < 1.0e-6 )

def test_TwoGroupEigenvalue_2G_1mat_sphere_keff():
    nuSigmaf_func = lambda r: 0.1570
    D_func = lambda r: 3.850204978408833
    Sigmaa_func = lambda r: 0.1532
    Sigmas_func = lambda r: 0.0
    R = 100
    I = 20
    k,phi_f,phi_t,centers = book.TwoGroupEigenvalue(R,I,D_func,D_func,
            Sigmaa_func,Sigmaa_func,nuSigmaf_func,nuSigmaf_func,Sigmas_func,
            [1,0.0],[1,0.0],2,epsilon=1.0e-10)
    book_keff = 1.00002955111
    assert( abs(k - book_keff) < 1.0e-4 )

def test_TwoGroupEigenvalue_2G_1mat_sphere_kinf():
    nuSigmaf1_func = lambda r: 0.0085
    nuSigmaf2_func = lambda r: 0.185
    D_func = lambda r: 0.1
    Sigmar1_func = lambda r: 0.0362
    Sigmar2_func = lambda r: 0.121
    Sigmas12_func = lambda r: 0.0241
    R = 5
    I = 50
    k,phi_f,phi_t,centers = book.TwoGroupEigenvalue(R,I,D_func,D_func,
            Sigmar1_func,Sigmar2_func,nuSigmaf1_func,nuSigmaf2_func,Sigmas12_func,
            [0,1.0],[0,1.0],2,epsilon=1.0e-8)
    book_keff = 1.25268252483
    assert( abs(k - book_keff) < 1.0e-4 )

def test_TwoGroupEigenvalue_2G_1mat_sphere_flux():
    nuSigmaf1_func = lambda r: 0.0085
    nuSigmaf2_func = lambda r: 0.185
    D_func = lambda r: 0.1
    Sigmar1_func = lambda r: 0.0362
    Sigmar2_func = lambda r: 0.121
    Sigmas12_func = lambda r: 0.0241
    R = 5
    I = 50
    k,phi_f,phi_t,centers = book.TwoGroupEigenvalue(R,I,D_func,D_func,
            Sigmar1_func,Sigmar2_func,nuSigmaf1_func,nuSigmaf2_func,Sigmas12_func,
            [0,1.0],[0,1.0],2,epsilon=1.0e-8)
    book_flux_ratio = 5.021
    assert( abs((phi_f[0] / phi_t[0]) - book_flux_ratio) < 1.0e-3 )

def test_TwoGroupEigenvalue_2G_2mat_sphere_refl():
    R_reac = 50.0
    nuSigmaf1_func = lambda r: 0.00085*(r<=R_reac) + 0.0
    nuSigmaf2_func = lambda r: 0.057*(r<=R_reac) + 0.0
    D_func = lambda r: 1.0
    Sigmar1_func = lambda r: 0.01
    Sigmar2_func = lambda r: 0.01*(r<=R_reac) + 0.00049*(r>R_reac)
    Sigmas12_func = lambda r: 0.001*(r<=R_reac) + 0.009*(r>R_reac)
    R = 100
    I = 100
    k,phi_f,phi_t,centers = book.TwoGroupEigenvalue(R,I,D_func,D_func,
            Sigmar1_func,Sigmar2_func,nuSigmaf1_func,nuSigmaf2_func,Sigmas12_func,
            [0.25,0.5*D_func(R)],[0.25,0.5*D_func(R)],2,epsilon=1.0e-8)
    book_keff = 1.06508498598
    assert( abs( k - book_keff ) < 1.0e-6 )

def test_TwoGroupEigenvalue_2G_1mat_sphere_refl():
    R_reac = 500.0
    nuSigmaf1_func = lambda r: 0.00085*(r<=R_reac) + 0.0
    nuSigmaf2_func = lambda r: 0.057*(r<=R_reac) + 0.0
    D_func = lambda r: 1.0
    Sigmar1_func = lambda r: 0.01
    Sigmar2_func = lambda r: 0.01*(r<=R_reac) + 0.00049*(r>R_reac)
    Sigmas12_func = lambda r: 0.001*(r<=R_reac) + 0.009*(r>R_reac)
    R = 50
    I = 100
    k,phi_f,phi_t,centers = book.TwoGroupEigenvalue(R,I,D_func,D_func,
            Sigmar1_func,Sigmar2_func,nuSigmaf1_func,nuSigmaf2_func,Sigmas12_func,
            [0.25,0.5*D_func(R)],[0.25,0.5*D_func(R)],2,epsilon=1.0e-8)
    book_keff = 0.368702897492
    assert( abs( k - book_keff ) < 1.0e-6 )

if __name__ == '__main__':
    test_LU_factor()
