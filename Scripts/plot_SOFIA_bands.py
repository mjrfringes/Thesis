import numpy as np
import matplotlib.pyplot as plt
from plot_library import *

### make plot of transmission curves from atmosphere

data11 = np.loadtxt('data/11pt3mu.txt')
data19 = np.loadtxt('data/FORCAST_19.7um_dichroic.txt')
data31 = np.loadtxt('data/FORCAST_31.5um_dichroic.txt')
data37o = np.loadtxt('data/FORCAST_37.1um_dichroic.txt')
data37d = np.loadtxt('data/FORCAST_37.1um_direct.txt')

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)

ax.plot(data11[:,0],data11[:,1],label='11.3 microns',color=colors[0])
ax.plot(data19[:,0],data19[:,1],label='19.7 microns',color=colors[1])
ax.plot(data31[:,0],data31[:,1],label='31.5 microns',color=colors[2])
ax.plot(data37o[:,0],data37o[:,1],label='37.1 microns Dichroic',color=colors[3])
ax.plot(data37d[:,0],data37d[:,1],label='37.1 microns Open',color=colors[4])

ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Throughput')
ax.grid(True)
ax.legend(loc='best')

fig.tight_layout()
fig.savefig('../Figures/SOFIA_bands.pdf')
plt.show()