import numpy as np
import matplotlib.pyplot as plt
import plot_library as pl
import seaborn.apionly as sns
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['font.size'] = 11
mpl.rcParams['font.weight'] = 100
#mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

data = np.loadtxt('BETTII_transmission.txt')

print data.shape
wav = data[:,0]
SW1 = data[:,1]
SW2 = data[:,2]
LW1 = data[:,3]
LW2 = data[:,4]

blue = sns.xkcd_rgb['denim blue']
red = sns.xkcd_rgb['pale red']
green = sns.xkcd_rgb['medium green']

fig,ax = plt.subplots(figsize=pl.figsize,facecolor=pl.facecolor)

ax.plot(wav,SW1*100,blue,lw=2)
ax.plot(wav,LW1*100,red,lw=2)
ax.set_xlim(25,130)
ax.set_ylim(0,80)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Transmission (%)')
ax.grid(True)

fig.tight_layout()
fig.savefig('../../Figures/BETTII_transmission.pdf')
plt.show()