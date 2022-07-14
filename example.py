
from NeutronDiffusion import ndiffusion 
# from NeutronDiffusion.diffusion import Diffusion

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Which problem to select")
parser.add_argument("-problem", action="store", dest="problem")
parser.add_argument("-I", action="store", dest="I")
parser.add_argument("-G", action="store", dest="G")
usr_input = parser.parse_args()

problem = usr_input.problem if usr_input.problem is not None else "plutonium_carbon_01"
I = int(usr_input.I) if usr_input.I is not None else 20
G = int(usr_input.G) if usr_input.G is not None else 70

flux, keff = ndffusion.run_diffusion(problem, G, I, geo="sphere")
# flux, keff = Diffusion.run(problem, G, I, geo="sphere")

np.save("{}/phi_diffusion_G{}_I{}".format(problem, str(G).zfill(3), \
                                           str(I).zfill(4)), flux)
np.save("{}/keff_diffusion_G{}_I{}".format(problem, str(G).zfill(3), \
                                           str(I).zfill(4)), keff)

