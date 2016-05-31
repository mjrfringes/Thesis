import numpy as np
import matplotlib.pyplot as plt
from plot_library import *

### make plot of transmission curves from atmosphere

data = np.loadtxt('data/visibility.txt')
data2 = np.loadtxt('data/visibilitylong.txt')

print data.shape
ang = data[:,0]
vis = data[:,1]
intvis = data[:,2]
ang2 = data2[:,0]
intvis2 = data2[:,2]

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)

ax.plot(ang,abs(intvis),blue,lw=2,label='Band 1 integrated visibility')
ax.plot(ang2,abs(intvis2),red,lw=2,label='Band 2 integrated visibility')
ax.annotate('Nepture', xy=(2.3, 0), xycoords='data',
			xytext=(2.3, 0.2),
			arrowprops=dict(arrowstyle="->")
			)
ax.annotate('Uranus', xy=(3.75, 0), xytext=(3.75, 0.2),
			arrowprops=dict(arrowstyle="->")
			)
#ax.set_xlim(1,2.5)
ax.set_ylim(-0.1,1.0)
ax.set_xlabel('Angular size (arcsec)')
ax.set_ylabel('Visibility or fringe contrast')
ax.grid(True)
ax.legend(loc='best')

fig.tight_layout()
fig.savefig('../Figures/Visibilities.pdf')
plt.show()