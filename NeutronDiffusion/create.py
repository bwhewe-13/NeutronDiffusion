""" Setting up diffusion problems """


import numpy as np
from collections import OrderedDict
import json

DATA_PATH = 'data/'

problem_dictionary = json.load(open('problem_setup.json','r'))

def selection(problem_name,G,I):
    # Call the ordered dictionary
    problem = problem_dictionary[problem_name]
    
    R = []
    diffusion = []
    scatter = []
    chi = []
    fission = []
    removal = []

    for mat,r in problem.items():
        t1, t2, t3, t4, t5 = loading_data(G,mat)

        # diffusion.append(t1[::-1])
        # scatter.append(np.flip(t2,axis=1)[::-1])
        # chi.append(t3)
        # fission.append(np.flip(t4,axis=1)[::-1])
        # removal.append(t5[::-1])

        diffusion.append(t1)
        scatter.append(t2)
        chi.append(t3)
        fission.append(t4)
        removal.append(t5)
        R.append(r)

    BC = np.zeros((G,2)) + 0.25
    BC[:,1] = 0.5*diffusion[-1] # outer edge

    diffusion = np.array(diffusion)
    scatter = np.array(scatter)
    chi = np.array(chi)
    fission = np.array(fission)
    removal = np.array(removal)

    # This is for full matrix fission
    if np.all(chi == None):
        return G,R,I,diffusion,scatter,None,fission,removal,BC

    return G,R,I,diffusion,scatter,chi,fission,removal,BC

def loading_data(G,material):
    diffusion = np.loadtxt('{}D_{}G_{}.csv'.format(DATA_PATH,G,material))
    absorb = np.loadtxt('{}Siga_{}G_{}.csv'.format(DATA_PATH,G,material))
    scatter = np.loadtxt('{}Scat_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    removal = [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
    np.fill_diagonal(scatter,0)
    # Some have fission matrix, others have birth rate and fission vector
    try:
        chi = np.loadtxt('{}chi_{}G_{}.csv'.format(DATA_PATH,G,material))
        fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material))
    except OSError:
        chi = None
        fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    return diffusion,scatter,chi,fission,removal

def add_problem_to_file(layers,materials,name):
    """ Adding problem to json for easy access 
    Inputs:
        layers: list of numbers (width of each material)
        materials: list of string (for loading csv data)
        name: string of name to be called 
    Returns:
        status string    """

    # Load current dictionary
    with open('problem_setup.json','r') as fp:
        problems = json.load(fp)

    # Check to see duplicate names
    if name in problems.keys():
        return "Name already exists"

    # Working inside (center) outward to edge
    od = OrderedDict()
    for layer,material in zip(layers,materials):
        od[material] = layer

    # Add to existing dictionary
    problems[name] = od

    # Save new dictionary
    with open('problem_setup.json', 'w') as fp:
        json.dump(problems, fp, sort_keys=True, indent=4)

    return "Successfully Added Element"

    