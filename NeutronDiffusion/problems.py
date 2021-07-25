""" Setting up diffusion problems """

import numpy as np
import re

DATA_PATH = 'data/'

def selection(problem,R,I):
    G = int(re.findall(r'\d+',problem)[0])
    problem = re.findall(r'^[A-Za-z]+',problem)[0]
    prob = getattr(Problems,problem)
    return prob(G,R,I)

class Problems:
    def compound(G,ext=''):
        diffusion = lambda r: np.loadtxt('{}D_{}G_Pu_20pct240{}.csv'.format(DATA_PATH,G,ext))
        chi = lambda r: np.loadtxt('{}chi_{}G_Pu_20pct240{}.csv'.format(DATA_PATH,G,ext))
        fission = lambda r: np.loadtxt('{}nuSigf_{}G_Pu_20pct240{}.csv'.format(DATA_PATH,G,ext))
        absorb = np.loadtxt('{}Siga_{}G_Pu_20pct240{}.csv'.format(DATA_PATH,G,ext))
        scatter = np.loadtxt('{}Scat_{}G_Pu_20pct240{}.csv'.format(DATA_PATH,G,ext),delimiter=',')
        removal = lambda r: [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
        np.fill_diagonal(scatter,0)
        return diffusion,scatter,chi,fission,removal

    def Pu(G,R,I):
        diffusion,scat,chi,fission,removal = Problems.compound(G,ext='')
        scatter = lambda r: scat
        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion(R)
        return G,R,I,diffusion,scatter,chi,fission,removal,BC

    def PuC(G,R,I):
        diffusion,scat,chi,fission,removal = Problems.compound(G,ext='C')
        scatter = lambda r: scat
        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion(R)
        return G,R,I,diffusion,scatter,chi,fission,removal,BC

    def PuPuC(G,R,I):
        diffusion_outer,scatter_outer,chi_outer,fission_outer,removal_outer = Problems.compound(G,ext='')
        diffusion_inner,scatter_inner,chi_inner,fission_inner,removal_inner = Problems.compound(G,ext='C')
        # Combine the materials
        split = 10 # Separation point between materials
        removal = lambda r: removal_inner(r)*(r <= split) + removal_outer(r)*(r>split)
        scatter = lambda r: scatter_inner*(r <= split)+ scatter_outer*(r>split)
        diffusion = lambda r: diffusion_inner(r)*(r <= split)+ diffusion_outer(r)*(r>split)
        chi = lambda r: chi_inner(r)*(r <= split)+ chi_outer(r)*(r>split)
        fission = lambda r: fission_inner(r)*(r <= split)+ fission_outer(r)*(r>split)
        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion(R)
        return G,R,I,diffusion,scatter,chi,fission,removal,BC

