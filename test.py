import numpy as np
import matplotlib.pyplot as plt

diff = np.loadtxt('data/D_618G_Pu_20pct240.csv',delimiter=',')
absorb = np.loadtxt('data/Siga_618G_Pu_20pct240.csv',delimiter=',')
scatter = np.loadtxt('data/Scat_618G_Pu_20pct240.csv',delimiter=',')

total = absorb + np.sum(scatter,axis=0)
# total = 1 / (3 * diff)

print(np.argwhere(total < 0))

np.save('../../llnl/discrete1/xs/pumix/vecTotal',total)


