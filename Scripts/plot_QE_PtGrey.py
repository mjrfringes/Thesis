import numpy as np
import matplotlib.pyplot as plt
from plot_library import *

### make plot of transmission curves from atmosphere

data = np.loadtxt('data/QE_pointgrey.txt')

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor,color=blue)
print data
ax.plot(data[:,0],data[:,1])

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Quantum efficiency')
ax.grid(True)

fig.tight_layout()
fig.savefig('../Figures/QEPtGrey.pdf')
plt.show()