
from diffusion import Diffusion
import create

import numpy as np

G = 618; I = 20
problem = "plutonium_01"
phi_true,keff_true = Diffusion.run(problem,G,I,geo='sphere')

# problem = "plutonium_mix_02"

problem = 'example_01'
# problem = "plutonium_01"
phi1,keff1 = Diffusion.run(problem,G,I,geo='sphere')


problem = 'example_02'
phi2,keff2 = Diffusion.run(problem,G,I,geo='sphere')

print('/n/n')
print('True keff {}'.format(keff_true))
print('Example 1 keff {}'.format(keff1))
print('Example 2 keff {}'.format(keff2))