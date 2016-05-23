import numpy as np
import matplotlib.pyplot as plt
from plot_library import *
import seaborn.apionly as sns

### make plot of transmission curves

data = np.loadtxt('data/BETTII_transmission.txt')

print data.shape
wav = data[:,0]
SW1 = data[:,1]
SW2 = data[:,2]
LW1 = data[:,3]
LW2 = data[:,4]

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)

ax.plot(wav,SW1*100,blue,lw=2,label='Band 1')
ax.plot(wav,LW1*100,red,lw=2,label='Band 2')
ax.set_xlim(25,130)
ax.set_ylim(0,80)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Transmission (%)')
ax.grid(True)
ax.legend(loc='best')

fig.tight_layout()
fig.savefig('../Figures/BETTII_transmission.pdf')
plt.show()