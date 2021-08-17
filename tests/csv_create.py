# Convert Stainless to CSV

from discrete1.generate import XSGenerate087,XSGenerate618
from discrete1.util import display
from discrete1.fixed import Tools

import numpy as np
import matplotlib.pyplot as plt

path = '../Desktop/NeutronDiffusion/data/'

# total,scatter,fission = XSGenerate087('SS440').cross_section()

# diffusion = 1 / (3 * total)
# # diffusion = 1 / (3 * np.sum(scatter,axis=1))

# file = 'SS440'
# # Energy Edges
# energy = np.load('discrete1/data/energyGrid.npy')
# centers = display.gridPlot()

# np.savetxt(path+'D_87G_{}.csv'.format(file), diffusion, delimiter=",")
# np.savetxt(path+'group_edges_87G_{}.csv'.format(file), energy, delimiter=",")
# np.savetxt(path+'group_centers_87G_{}.csv'.format(file), centers, delimiter=",")
# np.savetxt(path+'nuSigf_87G_{}.csv'.format(file), fission, delimiter=",")
# np.savetxt(path+'Scat_87G_{}.csv'.format(file), scatter, delimiter=",")
# np.savetxt(path+'Siga_87G_{}.csv'.format(file), total - np.sum(scatter,axis=0), delimiter=",")

# material = 'UH3'
# enrichments = [0,0.25,0.3,0.5,0.6,1]
# labels = ['000','025','030','050','060','100']

# for enrich in range(len(labels)):

#     file = '{}_{}pct235'.format(material,labels[enrich])
#     print(file)

#     total,scatter,fission = XSGenerate087(material,enrich=enrichments[enrich]).cross_section()

#     diffusion = 1 / (3 * total)
#     # diffusion = 1 / (3 * np.sum(scatter,axis=1))

#     # Energy Edges
#     energy = np.load('discrete1/data/energyGrid.npy')
#     centers = display.gridPlot()
    
#     np.savetxt(path+'D_87G_{}.csv'.format(file), diffusion, delimiter=",")
#     np.savetxt(path+'group_edges_87G_{}.csv'.format(file), energy, delimiter=",")
#     np.savetxt(path+'group_centers_87G_{}.csv'.format(file), centers, delimiter=",")
#     np.savetxt(path+'nuSigf_87G_{}.csv'.format(file), fission, delimiter=",")
#     np.savetxt(path+'Scat_87G_{}.csv'.format(file), scatter, delimiter=",")
    # np.savetxt(path+'Siga_87G_{}.csv'.format(file), total - np.sum(scatter,axis=0), delimiter=",")


# total,scatter,fission = XSGenerate618.cross_section(0)
# total_239 = total[1].copy(); total_240 = total[2].copy()
# scatter_239 = scatter[1].copy(); scatter_240 = scatter[2].copy()
# fission_239 = fission[1].copy(); fission_240 = fission[2].copy()
data = np.load('mydata/pu239/Pu239-618group.npz')

# ['Pu_scat', 'Pu_sig_t', 'Pu_chi', 'Pu_nu_sig_f', 'Pu_inv_speed', 'Pu_group_edges']

diffusion = 1 / (3 * data['Pu_sig_t'])

# Energy Edges 87 Group 
# edges = np.load('discrete1/data/energyGrid.npy')[::-1] * 1E-6
edges = data['Pu_group_edges']
centers = np.array([float(edges[ii]+edges[jj])/2 for ii,jj in 
                         zip(range(len(edges)-1),range(1,len(edges)))])
# centers = display.gridPlot() 

# edges_239 = np.load('mydata/pu239/energy_edges_618.npy')
# centers_239 = np.load('mydata/pu239/energy_618.npy')

path = '../Desktop/NeutronDiffusion/data/'

np.savetxt(path+'D_618G_Pu_239.csv', diffusion, delimiter=",")
np.savetxt(path+'group_edges_618G_Pu_239.csv', edges, delimiter=",")
np.savetxt(path+'group_centers_618G_Pu_239.csv', centers, delimiter=",")
np.savetxt(path+'nuSigf_618G_Pu_239.csv', data['Pu_chi'][:,None] @ data['Pu_nu_sig_f'][None,:], delimiter=",")
np.savetxt(path+'Scat_618G_Pu_239.csv', data['Pu_scat'], delimiter=",")
np.savetxt(path+'Siga_618G_Pu_239.csv', data['Pu_sig_t']- np.sum(data['Pu_scat'],axis=0), delimiter=",")

"""
# Change to 87 Group

reduced = []
for ii,orig in enumerate(edges):
    count = 0
    for group in edges_239:
        if (group > edges[ii]):
            count += 1
    reduced.append(count)
inds = np.array(reduced)

total_239, scatter_239, fission_239 = Tools.group_reduction(87,edges_239,total_239,scatter_239,fission_239,inds=inds)

edges_239 = edges_239[inds].copy()
centers_239 = np.array([float(edges_239[ii]+edges_239[jj])/2 for ii,jj in 
                         zip(range(len(edges_239)-1),range(1,len(edges_239)))])

diffusion_239 = 1 / (3 * total_239)

np.savetxt(path+'D_87G_Pu239.csv', diffusion_239, delimiter=",")
np.savetxt(path+'group_edges_87G_Pu239.csv', edges_239, delimiter=",")
np.savetxt(path+'group_centers_87G_Pu239.csv', centers_239, delimiter=",")
np.savetxt(path+'nuSigf_87G_Pu239.csv', fission_239, delimiter=",")
np.savetxt(path+'Scat_87G_Pu239.csv', scatter_239, delimiter=",")
np.savetxt(path+'Siga_87G_Pu239.csv', total_239 - np.sum(scatter_239,axis=0), delimiter=",")

"""


print('Done!')

# plt.figure()
# plt.plot(total - np.sum(scatter.T,axis=1),label='Axis = 1',c='r',alpha=0.6)
# plt.plot(total - np.sum(scatter.T,axis=0),label='Axis = 0',c='b',alpha=0.6)
# plt.legend(loc='best'); plt.grid()
# plt.show()