from diffusion.main import Diffusion

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Which problem to select")
parser.add_argument("-problem", action="store", dest="problem")
parser.add_argument("-I", action="store", dest="I", type=int)
parser.add_argument("-G", action="store", dest="G", type=int)
parser.add_argument("-save", action="store", dest="save")
usr_input = parser.parse_args()

problem = "plutonium_carbon_01" if usr_input.problem is None else usr_input.problem
G = 70 if usr_input.G is None else usr_input.G
I = 20 if usr_input.I is None else usr_input.I

phi, keff = Diffusion.run(problem, G, I, geo="sphere")

if usr_input.save:
    np.save(
        "../jonas_data/{}/phi_G{}_I{}".format(
            problem, str(G).zfill(3), str(I).zfill(4)
        ),
        phi,
    )
    np.save(
        "../jonas_data/{}/keff_G{}_I{}".format(
            problem, str(G).zfill(3), str(I).zfill(4)
        ),
        keff,
    )
