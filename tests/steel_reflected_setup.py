
import numpy as np
import json
import re
import matplotlib.pyplot as plt


# Constants
DATA_PATH = '../../../llnl/discrete1/data/'
DATA_PATH_XS = '../../../llnl/discrete1/xs/'

avagadro = 6.022E23
barn = 1E-24
# Dictionaries
isotope_abundance = json.load(open(DATA_PATH + 'isotope_abundance.json','r')) # ['Abundance %','molar mass']
compound_density = json.load(open(DATA_PATH + 'compound_density.json','r'))   # ['molar mass','density']
temperature = '00'

def natural_compound(element,isotopes):
    return 1/sum([isotope_abundance[iso][0]/isotope_abundance[iso][1] for iso in isotopes if element in iso]) 


# Stainless Steel Grade 3
isotopes = ['fe54','fe56','fe57','cr50','cr52','cr53','cr54','si28','si29',
            'si30','mn55','cnat','cu63','cu65']
elements = ['fe','cr','si','mn','cnat','cu']


fe_percent = 0.986
cr_percent = 0.003
cnat_percent = 0.003
si_percent = 0.001 
mn_percent = 0.004
cu_percent = 0.003


# Percent of Each Element
percent_dictionary = {}
percent = [fe_percent,cr_percent,si_percent,mn_percent,cnat_percent,cu_percent]
for ii in range(len(elements)):
    percent_dictionary[elements[ii]] = percent[ii]
    
print('Percent Check',np.sum(percent))



fe_molar = natural_compound('fe',isotopes)
cr_molar = natural_compound('cr',isotopes)
cnat_molar = natural_compound('cnat',isotopes)
si_molar = natural_compound('si',isotopes)
mn_molar = natural_compound('mn',isotopes)
# Combine for Molar Mass of SS440
ss440_molar = 1/(fe_percent/fe_molar + cr_percent/cr_molar + 
    cnat_percent/cnat_molar + si_percent/si_molar + mn_percent/mn_molar)


# Get elements of isotopes
elements = [re.sub(r'[0-9]','',ii) for ii in isotopes]


# Density of SS440
density = [7.4805, 7.5837]
labels = ['layer2_ss', 'layer3_ss']

for rho,name in zip(density,labels):

    density_list = {}
    for iso,ele in zip(isotopes,elements):
        density_list[iso] = (isotope_abundance[iso][0]*percent_dictionary[ele]*rho*avagadro)/ \
            isotope_abundance[iso][1]*barn 

    total = [np.load(DATA_PATH_XS + '{}/vecTotal.npy'.format(ii))[eval(temperature)] for ii in density_list.keys()]
    total_xs = sum([total[count]*nd for count,nd in enumerate(density_list.values())])
            
    scatter = [np.load(DATA_PATH_XS + '{}/scatter_0{}.npy'.format(ii,temperature))[0] for ii in density_list.keys()]
    scatter_xs = sum([scatter[count]*nd for count,nd in enumerate(density_list.values())])

    fission_xs = np.zeros((scatter_xs.shape))
    
    # I don't know if I should do this
    # scatter_xs = scatter_xs.T.copy()
    # fission_xs = fission_xs.T.copy()

    # Get what diffusion equation needs
    sigma_t = total_xs.copy()
    sigma_s = scatter_xs.copy()
    sigma_f = fission_xs.copy()

    diffusion = 1 / (3 * total_xs)
    sigma_a = sigma_t - np.sum(sigma_s,axis=0)
    sigma_r = [sigma_a[gg] + np.sum(sigma_s,axis=0)[gg] - sigma_s[gg,gg] for gg in range(len(sigma_t))]

    # np.savetxt('../data/Siga_87G_{}.csv'.format(name),sigma_a,delimiter=',')
    # np.savetxt('../data/Scat_87G_{}.csv'.format(name),sigma_s,delimiter=',')
    # np.savetxt('../data/D_87G_{}.csv'.format(name),diffusion,delimiter=',')
    # np.savetxt('../data/nuSigf_87G_{}.csv'.format(name),sigma_f,delimiter=',')
    
    # print('\n{} Stainless'.format(name))
    # print(sigma_t.shape)
    # print(sigma_s.shape)
    # print(sigma_f.shape)
    # print(sigma_a.shape)

# Uranium (layer 1)
isotopes = ['u235','u238']
enrich = 0.366

u_molar = enrich * compound_density['U235'][0]        # Add U-235 molar mass
u_molar += (1 - enrich) * compound_density['U238'][0] # Add U-238 molar mass
                
rho = 18.4213
density_list = {}
        
# Add Enriched U235
density_list['u235'] = (enrich * rho * avagadro) / compound_density['U235'][0] * \
    (enrich * compound_density['U235'][0] + (1 - enrich) * compound_density['U238'][0]) / u_molar * barn
# Add U238
density_list['u238'] = ((1 - enrich) * rho * avagadro) / compound_density['U238'][0] * \
    (enrich * compound_density['U235'][0] + (1 - enrich) * compound_density['U238'][0]) / u_molar * barn
        
        
# Total Cross Section List
total = [np.load(DATA_PATH_XS + '{}/vecTotal.npy'.format(ii))[eval(temperature)] for ii in density_list.keys()]
total_xs = sum([total[count]*nd for count,nd in enumerate(density_list.values())])

scatter = [np.load(DATA_PATH_XS + '{}/scatter_0{}.npy'.format(ii,temperature))[0] for ii in density_list.keys()]
scatter_xs = sum([scatter[count]*nd for count,nd in enumerate(density_list.values())])

fission = [np.load(DATA_PATH_XS + '{}/nufission_0{}.npy'.format(ii,temperature))[0] for ii in density_list.keys()]
fission_xs = sum([fission[count]*nd for count,nd in enumerate(density_list.values())])
        
# I don't know if I should do this
# scatter_xs = scatter_xs.T.copy()
# fission_xs = fission_xs.T.copy()

sigma_t = total_xs.copy()
sigma_s = scatter_xs.copy()
sigma_f = fission_xs.copy()

diffusion = 1 / (3 * total_xs)
sigma_a = sigma_t - np.sum(sigma_s,axis=0)
sigma_r = [sigma_a[gg] + np.sum(sigma_s,axis=0)[gg] - sigma_s[gg,gg] for gg in range(len(sigma_t))]

# np.savetxt('../data/Siga_87G_U_36pct235.csv',sigma_a,delimiter=',')
# np.savetxt('../data/Scat_87G_U_36pct235.csv',sigma_s,delimiter=',')
# np.savetxt('../data/D_87G_U_36pct235.csv',diffusion,delimiter=',')
# np.savetxt('../data/nuSigf_87G_U_36pct235.csv',sigma_f,delimiter=',')

# print('\nUranium')
# print(sigma_t.shape)
# print(sigma_s.shape)
# print(sigma_f.shape)
# print(sigma_a.shape)
