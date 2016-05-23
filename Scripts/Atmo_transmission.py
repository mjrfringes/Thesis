import numpy as np
import matplotlib.pyplot as plt
from plot_library import *

### make plot of transmission curves from atmosphere

data = np.loadtxt('data/MK.txt')

print data.shape
wavMK = data[:,0]
MK = data[:,1]

data = np.loadtxt('data/balloon.txt')

print data.shape
wavBalloon = data[:,0]
balloon = data[:,1]



fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)

ax.plot(wavMK,MK,blue,lw=2,label='4 km altitude')
ax.plot(wavBalloon,balloon,red,lw=2,label='35 km altitude')
ax.set_xlim(1,2.5)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Transmission (%)')
ax.grid(True)
ax.legend(loc='best')

fig.tight_layout()
fig.savefig('../Figures/BETTII_atmo_transmission.pdf')
plt.show()