
from NeutronDiffusion.diffusion import Diffusion

import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Which problem to select')
parser.add_argument('-problem',action='store',dest='problem')
usr_input = parser.parse_args()

G = 618; I = 20

problem = 'plutonium_carbon_01' if usr_input.problem is None else usr_input.problem
# problem = "plutonium_mix_03"


phi,keff = Diffusion.run(problem,G,I,geo='sphere')

#np.save('../../phi_diffusion_plutonium_mix_02',phi)
#np.save('../../keff_diffusion_plutonium_mix_02',keff)


