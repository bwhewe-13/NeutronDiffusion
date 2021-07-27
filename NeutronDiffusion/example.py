
from diffusion import Diffusion

import numpy as np


G = 618; I = 20

# problem = "plutonium_mix_03"
# problem = "plutonium_carbon_01"
problem = "plutonium_mix_02"

phi,keff = Diffusion.run(problem,G,I,geo='sphere')

np.save('../../phi_diffusion_plutonium_mix_02',phi)
np.save('../../keff_diffusion_plutonium_mix_02',keff)


