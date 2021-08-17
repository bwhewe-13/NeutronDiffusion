#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 06:04:37 2021

@author: bwhewell
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

path = '../Desktop/NeutronDiffusion/data/'
Dg = np.loadtxt(path+'D_618G_Pu_20pct240C_mat.csv',delimiter=',')
fission = np.loadtxt(path+'nuSigf_618G_Pu_20pct240C_mat.csv',delimiter=',')
scatter = np.loadtxt(path+'Scat_618G_Pu_20pct240C_mat.csv',delimiter=',')
absorb = np.loadtxt(path+'Siga_618G_Pu_20pct240C_mat.csv',delimiter=',')
energy = np.loadtxt(path+'group_centers_618G_Pu_20pct240C_mat.csv',delimiter=',')

dat = np.load('mydata/pu240/Pu240-618group.npz')
total1 = dat['Pu_sig_t'].copy()
scatter1 = dat['Pu_scat'].copy()
fission1 = dat['Pu_chi'][:,None] @ dat['Pu_nu_sig_f'][None,:]

print(fission1.shape)


# %%
scatter = np.load('discrete1/xs/pu239/scatter_000.npy')
scatter1 = np.load('discrete1/xs/puc/scatter_000.npy')


temp = scatter.copy()
temp[temp == 0] = np.nan
plt.imshow(temp); plt.colorbar()
plt.show()

temp = scatter1.copy()
temp[temp == 0] = np.nan
plt.imshow(temp); plt.colorbar()
plt.show()

# %%

plt.plot(energy*1E6,1/(3*Dg),label='diffusion',c='r',alpha=0.6)
plt.loglog(energy*1E6,total1,label='transport',c='k',ls='--')
plt.legend(loc='best'); plt.grid()

# %%
edges = np.loadtxt(path+'group_edges_618G_Pu_240.csv',delimiter=',')
edges1 = dat['Pu_group_edges']

centers = np.array([float(edges1[ii]+edges1[jj])/2 for ii,jj in 
                         zip(range(len(edges1)-1),range(1,len(edges1)))])

print(np.array_equal(energy,centers))

# %%
from discrete1.util import sn,display
from discrete1.keigenvalue import Problem2

_,_,_,_,_,scatter,fission,_,_ = Problem2.steady('hdpe',0,orient='flip')

phi_d = np.load('../Desktop/phi_diffusion_plutonium_mix_02.npy')
keff_d = np.load('../Desktop/keff_diffusion_plutonium_mix_02.npy')

phi_t16 = np.load('../Desktop/phi_transport_s16_plutonium_mix_02.npy')
keff_t16 = np.load('../Desktop/keff_transport_s16_plutonium_mix_02.npy')

phi_t256 = np.load('../Desktop/phi_transport_s256_plutonium_mix_02.npy')
keff_t256 = np.load('../Desktop/keff_transport_s256_plutonium_mix_02.npy')


xs = scatter.copy()
xspace = np.linspace(0,10,20)
plt.plot(xspace,sn.totalFissionRate(xs,phi_d),label='Diffusion keff {}'.format(np.round(keff_d,5)),c='k',ls='--')
plt.plot(xspace,sn.totalFissionRate(xs,phi_t16),label='Transport S16 keff {}'.format(np.round(keff_t16,5)),c='r',alpha=0.6)
plt.plot(xspace,sn.totalFissionRate(xs,phi_t256),label='Transport S256 keff {}'.format(np.round(keff_t256,5)),c='b',alpha=0.6)

plt.xlabel('Distance from Center (cm)')
plt.ylabel(r'Scatter Rate Density cc$^{-1}$ s$^{-1}$')
# plt.title(r'Diffusion vs. Transport, Pu Sphere, $\Sigma_t$ = $\Sigma_a$ + $\Sigma_s$')
plt.title(r'Diffusion vs. Transport, Plutonium Mix')
plt.grid(); plt.legend(loc='best')
# plt.savefig('diffusion_vs_transport_04.png',bbox_inches='tight')

# %%

_,_,_,_,_,scatter,fission,_,_ = Problem2.steady('hdpe',0,orient='flip')

phi_t16 = np.load('../Desktop/phi_transport_s16t_plutonium_mix_02.npy')
keff_t162 = np.load('../Desktop/keff_transport_s16t_plutonium_mix_02.npy')

full = sn.totalFissionRate(scatter,phi_t16)


xs = scatter.copy()
xspace = np.linspace(0,10,100)

plt.plot(xspace,sn.totalFissionRate(xs,phi_t16),label='Transport S16 keff {}'.format(np.round(keff_t16,5)),c='r',alpha=0.6)
plt.semilogy(xspace,sn.totalFissionRate(xs,phi_t16),label='Transport S256 keff {}'.format(np.round(keff_t256,5)),c='b',alpha=0.6)

plt.xlabel('Distance from Center (cm)')
plt.ylabel(r'Fission Rate Density cc$^{-1}$ s$^{-1}$')
# plt.title(r'Diffusion vs. Transport, Pu Sphere, $\Sigma_t$ = $\Sigma_a$ + $\Sigma_s$')
plt.title(r'Diffusion vs. Transport, Plutonium Mix')
plt.grid(); plt.legend(loc='best')
