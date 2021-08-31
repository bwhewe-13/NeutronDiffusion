
from NeutronDiffusion.diffusion import Diffusion

import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Which problem to select')
parser.add_argument('-problem',action='store',dest='problem')
parser.add_argument('-I',action='store',dest='I')
usr_input = parser.parse_args()

G = 12; 
if usr_input.I is not None:
    I = int(usr_input.I)
else:
    I = 50
# G = 87; #I = 20

# problem = 'plutonium_carbon_01' if usr_input.problem is None else usr_input.problem
# problem = "plutonium_mix_03"

problem = 'plutonium_01'

phi,keff = Diffusion.run(problem,G,I,geo='sphere')

#np.save('../../phi_diffusion_plutonium_01_{}'.format(I),phi)
#np.save('../../keff_diffusion_plutonium_01_{}'.format(I),keff)


from NeutronDiffusion.create import selection
problem = 'plutonium_01'
G = 12; I = 2
attributes = selection(problem,G,I,BC=0)
initialize = Diffusion(*attributes,geo='sphere')
initialize.geometry()
A = initialize.construct_A_fast()
phi,_ = initialize.solver(A,fast=True)
np.save('../A_matrix',A)
np.save('../phi_vector',phi)
np.save('../b_vector',initialize.construct_b_fast(phi))

