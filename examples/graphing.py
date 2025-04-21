# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/Pu_01pct239C_618G.npz')
pu = np.load('data/Pu_239_618G.npz')
hdpe = np.load('data/HDPE_618G.npz')

# %% Absorption

# plt.figure()
# plt.loglog(data['group_centers'],data['Siga'],label='Combo',c='k',ls='--')
# plt.loglog(data['group_centers'],pu['Siga'],label='Pu239',c='r',alpha=0.7)
# plt.loglog(data['group_centers'],hdpe['Siga'],label='HDPE',c='b',alpha=0.7)
# plt.legend(loc=0); plt.grid(); 

plt.figure()
plt.semilogy(data['Siga'],label=r'$\Sigma_a$',c='k',ls='-',alpha=0.7)
# plt.semilogy(pu['Siga'][::-1],label='Pu239',c='r',alpha=0.7)
# plt.semilogy(hdpe['Siga'][::-1],label='HDPE',c='b',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Absorption Cross Section')

# %% Scatter

plt.figure()
plt.loglog(data['group_centers'],np.sum(data['Scat'],axis=1),label='Scatter',c='k',ls='-',alpha=0.7)
# plt.loglog(data['group_centers'],np.sum(pu['Scat'],axis=1),label='Pu239',c='r',alpha=0.7)
# plt.loglog(data['group_centers'],np.sum(hdpe['Scat'],axis=1),label='HDPE',c='b',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Scatter Cross Section')

plt.figure()
plt.semilogy(np.sum(data['Scat'],axis=1),label=r'$\Sigma_s$',c='k',ls='-',alpha=0.7)
# plt.semilogy(np.sum(pu['Scat'],axis=1)[::-1],label='Pu239',c='r',alpha=0.7)
# plt.semilogy(np.sum(hdpe['Scat'],axis=1)[::-1],label='HDPE',c='b',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Scatter Cross Section')


# %% Fission

plt.figure()
plt.loglog(data['group_centers'],np.sum(data['nuSigf'],axis=1),label='Fission',c='k',ls='-',alpha=0.7)
# plt.loglog(data['group_centers'],np.sum(pu['nuSigf'],axis=1),label='Pu239',c='r',alpha=0.7)
# plt.loglog(data['group_centers'],np.sum(hdpe['nuSigf'],axis=1),label='HDPE',c='b',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Fission Cross Section')

plt.figure()
plt.semilogy(np.sum(data['nuSigf'],axis=1),label=r'$\chi \nu \Sigma_f$',c='k',ls='-',alpha=0.7)
# plt.semilogy(np.sum(pu['nuSigf'],axis=1)[::-1],label='Pu239',c='r',alpha=0.7)
# plt.semilogy(np.sum(hdpe['nuSigf'],axis=1)[::-1],label='HDPE',c='b',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Fission Cross Section')

# print(data['group_centers'][::-1][350:450])

# %% Total 

# plt.figure()
# plt.loglog(data['group_centers'],1/(3*data['D']),label='Total',c='k',ls='--')
# plt.loglog(data['group_centers'],np.sum(data['nuSigf'],axis=1),label='Fission',c='b',alpha=0.7)
# plt.loglog(data['group_centers'],np.sum(data['Scat'],axis=1),label='Scatter',c='r',alpha=0.7)
# plt.legend(loc=0); plt.grid(); 

plt.figure()
plt.semilogy(1/(3*data['D']),label=r'$\Sigma_t$',c='k',ls='-')
# plt.semilogy(np.sum(data['nuSigf'],axis=1)[::-1],label='Fission',c='b',alpha=0.7)
# plt.semilogy(np.sum(data['Scat'],axis=1)[::-1],label='Scatter',c='r',alpha=0.7)
plt.legend(loc=0); plt.grid(); 
plt.title('Total Cross Section')



