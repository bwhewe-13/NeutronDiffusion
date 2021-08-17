import numpy as np
from NeutronDiffusion import diffusion

# Problem 1 (pg 358 - 360)
# One Group, One Material
G = 1
R = [100]
I = 20
nuSigf = np.array([[[0.1570]]]); 
D = np.array([[3.850204978408833]]); 
Siga = np.array([[0.1532]]); 
Scat = np.array([[[0.]]]); 

BC = np.array([[1.,0.]])

geo = 'cylinder'
problem = diffusion.Diffusion(G,R,I,D,Scat,None,nuSigf,Siga,BC,geo)
problem.geometry()
# phi = np.load('phi_{}.npy'.format(geo))
# b = problem.constructing_b_fast_list(phi)
# np.save('b_vector_{}'.format(geo),b)

A = problem.constructing_A_fast_list()
np.save('A_matrix_{}'.format(geo),A)

phi,keff = problem.solver(A)


# sphere: keff = 1.00002966108
# cylinder: keff = 1.00002258361
# slab: keff = 1.00001243892

# problem = Diffusion(1,R,I,)


# Problem 2 (pg 360)
# One Group, Two Materials
problem = {}
problem['inner'] = 5
problem['outer'] = 5

nuSigf_inner = 0.7 * np.array([0.1570])
D_inner = 5.0 * np.array([3.850204978408833])
Siga_inner = 0.5 * np.array([0.1532])
Scat_inner = np.array([0])

nuSigf_outer = 0.0 * np.array([0.1570])
D_outer = 1.0 * np.array([3.850204978408833])
Siga_outer = 0.01 * np.array([0.1532])
Scat_outer = np.array([0])

# sphere: keff = 0.95888
# cylinder: keff = 1.14147
# slab: keff = 1.2955


# Problem 3 (pg 373)