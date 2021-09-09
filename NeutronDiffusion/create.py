""" Setting up diffusion problems """

import numpy as np
import json
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('NeutronDiffusion','../data/')
# DATA_PATH = '../data/'

problem_dictionary = json.load(open(DATA_PATH+'problem_setup.json','r'))

def selection(name,G,I,BC=0):
    # BC defaulted to vacuum at edge
    problem = problem_dictionary[name]
    
    R = []
    diffusion = []
    scatter = []
    chi = []
    fission = []
    removal = []

    for mat,r in problem.items():
        t1, t2, t3, t4, t5 = loading_data(G,mat)

        diffusion.append(t1)
        scatter.append(t2)
        chi.append(t3)
        fission.append(t4)
        removal.append(t5)
        R.append(r)

    bounds = boundary_conditions(diffusion[-1],alpha=BC)

    diffusion = np.array(diffusion)
    scatter = np.array(scatter)
    chi = np.array(chi)
    fission = np.array(fission)
    removal = np.array(removal)

    # This is for full matrix fission
    if np.all(chi == None):
        return G,R,I,diffusion,scatter,None,fission,removal,bounds

    return G,R,I,diffusion,scatter,chi,fission,removal,bounds

def loading_data(G,material):
    # Still run with .csv files but want .npz
    try:
        data_dict = np.load('{}{}_{}G.npz'.format(DATA_PATH,material,G))
        diffusion = data_dict['D'].copy()
        absorb = data_dict['Siga'].copy()
        scatter = data_dict['Scat'].copy()
        centers = data_dict['group_centers'].copy()
    except FileNotFoundError:
        diffusion = np.loadtxt('{}D_{}G_{}.csv'.format(DATA_PATH,G,material))
        absorb = np.loadtxt('{}Siga_{}G_{}.csv'.format(DATA_PATH,G,material))
        scatter = np.loadtxt('{}Scat_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
        centers = np.loadtxt('{}group_centers_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    
    if np.argmax(centers) == 0:
        removal = [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
    else:
        removal = [absorb[gg] + np.sum(scatter,axis=1)[gg] - scatter[gg,gg] for gg in range(G)]
    np.fill_diagonal(scatter,0)

    # Separate chi and nu fission vectors - .npz
    try:
        data_dict = np.load('{}{}_{}G.npz'.format(DATA_PATH,material,G))
        fission = data_dict['nuSigf'].copy()
        try:
            chi = data_dict['chi'].copy()
        except KeyError:
            chi = None
    except FileNotFoundError:
        try:
            chi = np.loadtxt('{}chi_{}G_{}.csv'.format(DATA_PATH,G,material))
            fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material))
        except OSError:
            chi = None
            fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    return diffusion,scatter,chi,fission,removal

def boundary_conditions(Dg,alpha):
    """ Boundary Conditions Setup
    A = (1 - alpha) / (4 * (1 + alpha))
    B = D / 2
    
    alpha:
        0 for vacuum
        1 for reflective
        float for albedo
    """
    A = lambda alpha: (1 - alpha) / (4 * (1 + alpha))

    bounds = np.zeros((len(Dg),2)) + A(alpha)
    bounds[:,1] = 0.5 * Dg 

    # Special Reflective Case
    if alpha == 1:
        bounds[:,1] = 1

    return bounds
