import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
#import seaborn

folder = '/Users/mrizzo/Downloads/hochunk3d_20140131/models/parfiles/'
icy_grains = ['kmh_ice090.par','kmh_ice095.par','r400_ice095_extrap.par','barb.r550.ice095summary','dust_ice_icsgra3_full.txt_d2g.txt','OH5.par']
standard = ['kmh.par','www003.par','ww04.par','draine_opac_new.dat']

fig,((ax1,ax2,ax3,ax4))=plt.subplots(4,1,sharex=True,figsize=(6,12))
linestyles = ['-', '--', ':','-.']
for dust in icy_grains+standard:
	# load file
	d = np.loadtxt(folder+dust)
	wav = d[:,0]
	alb = d[:,2]/d[:,1]
	op = d[:,3]
	g = d[:,4]
	pmax = d[:,5]
	if dust in icy_grains: val=0
	else: val=3
	ax1.plot(wav,g,label=dust,linestyle=linestyles[val])
	#ax.set_xlim([0,100])
	ax1.set_xscale("log")
	ax1.set_title('average cosine of scattering angle')
	ax2.plot(wav,op,label=dust,linestyle=linestyles[val])
	#ax.set_xlim([0,100])
	ax2.set_xscale("log")
	ax2.set_yscale("log")
	ax2.set_title('Opacity')
	ax3.plot(wav,alb,label=dust,linestyle=linestyles[val])
	#ax.set_xlim([0,100])
	ax3.set_xscale("log")
	ax3.set_title('albedo')
	ax4.plot(wav,pmax,label=dust,linestyle=linestyles[val])
	#ax.set_xlim([0,100])
	ax4.set_xscale("log")
	ax4.set_xlabel("Wavelength ($\mu m$)")
	ax4.set_title('Max cosine of polarization angle')

#ax1.legend(loc='top right')
# ax2.legend(loc='top right')
# ax3.legend(loc='top right')
# ax4.legend(loc='top right')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
fig.tight_layout()
plt.show()
fig.savefig('dust_types.png')