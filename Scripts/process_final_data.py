import numpy as np
import matplotlib.pyplot as plt
from plot_library import *
from astropy.table import Table,Column
import pickle
import pandas as pd
from astropy.coordinates import SkyCoord
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn.apionly as sns
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from astropy import constants as const
from astropy.coordinates import SkyCoord,FK5
import astropy.units as u
from astropy.analytic_functions import blackbody_nu
import itertools

def distpc(target):
	if "IRAS20050" in target:
		return 700.
	elif "NGC2071" in target:
		return 410.
	elif "Oph" in target:
		return 140.
	elif "1333" in target:
		return 240.
	elif "2264" in target:
		return 913.
	elif "7129" in target:
		return 1000.
	elif "CepA" in target:
		return 730.
	elif "CepC" in target:
		return 730.
	elif "S140" in target:
		return 900.
	elif "S171" in target:
		return 850.


# load data
clean_table = pickle.load(open('data/clean_table.data','r'))


clean_table['dist'] = [distpc(target) for target in clean_table['Cluster']]

lam = 1.1*u.mm
nu = const.c/lam
nu = nu.to(u.Hz)
Fnu = blackbody_nu(nu,20)
clean_table['F1100'].unit = u.Jy
clean_table['S1300'].unit = u.Jy
clean_table['calc_mass_F1100'] = ((clean_table['dist']*u.pc)**2*clean_table['F1100'].to(u.erg/u.cm**2/u.s/u.Hz)/(blackbody_nu((const.c/(1.1*u.mm)).to(u.Hz),20)*u.sr*0.0114*u.cm**2/u.g)).to(u.M_sun)
clean_table.mask['calc_mass_F1100'] = clean_table.mask['F1100']
clean_table['calc_mass_S1300'] = ((clean_table['dist']*u.pc)**2*clean_table['S1300'].to(u.erg/u.cm**2/u.s/u.Hz)/(blackbody_nu((const.c/(1.3*u.mm)).to(u.Hz),20)*u.sr*0.009*u.cm**2/u.g)).to(u.M_sun)
clean_table.mask['calc_mass_S1300'] = clean_table.mask['S1300']
clean_table['calc_mass'] = clean_table['calc_mass_S1300'].copy()
clean_table.mask['calc_mass'] = clean_table.mask['calc_mass_S1300'].copy()
for i in range(len(clean_table)):
	if clean_table.mask['calc_mass_S1300'][i]:
		if ~clean_table.mask['calc_mass_F1100'][i]:
			clean_table['calc_mass'][i] = clean_table['calc_mass_F1100'][i]
			clean_table.mask['calc_mass'][i] = False

clean_table['SOFIA_name','env_mass','calc_mass'].more()

# add ratios
clean_table['R19'] = clean_table['R50_19']/clean_table['R50_cal_19']
clean_table['R31'] = clean_table['R50_31']/clean_table['R50_cal_31']
clean_table['R37'] = clean_table['R50_37']/clean_table['R50_cal_37']

clean_table.sort(['F37','R37'])
clean_table.reverse()
clean_table['SOFIA_name','RA','DEC','Property','R37','F37'].more()

df = clean_table.to_pandas()
df.index = df['SOFIA_name']
c = SkyCoord(df['RA'],df['DEC'],frame=FK5,unit=u.deg)
df['Coordinates'] = c.to_string('hmsdms',precision=1)
print df['Coordinates']

### main photometry table
#dftot = df.loc[df['Cluster'].isin(['NGC1333','Oph','IRAS20050'])]
dftot = df.loc[df['Property'].isin(['Clustered','Extended','Isolated'])]
print dftot
columns = ['j','h','ks','i1','i2','i3','i4','F11','F19','m1','F31','F37','m2','H70','H160','H250','H350','H500','S850','F1100','S1300']
e_columns = ["e_"+col for col in columns]
flag_columns = ["flag_"+col for col in columns]
cols = list(itertools.chain.from_iterable(zip(columns,e_columns)))
print cols

# total table
dftot[['Coordinates','Property']+cols+['R37','Lbol','alpha','R','env_mass','env_mass_std','calc_mass','sLsun','sLsun_std','Lbol','inc','ext','s']].to_csv('Data/alldata.csv',na_rep='')
print dftot[['Coordinates','Property']+cols+['R37','Lbol','alpha','R','env_mass','env_mass_std','calc_mass','sLsun','sLsun_std','Lbol','inc','ext','s']].to_latex(longtable=True,na_rep='--')

# params-only table
dftot[['Coordinates','Property','R37','Lbol','alpha','R','env_mass','env_mass_std','calc_mass','sLsun','sLsun_std','Lbol','inc','ext','s']].to_csv('Data/alldata_params.csv',na_rep='')

# targets table
df[['SOFIA_name','Coordinates','RA','DEC','Property','R37','F37']].to_csv('Data/BETTII_Targets.csv')

## IRAS20050
df1 = df.loc[df['Cluster'].isin(['IRAS20050'])]

columns = ['ks','i1','i2','i3','i4','F11','F19','m1','F31','F37']
e_columns = ["e_"+col for col in columns]
cols = list(itertools.chain.from_iterable(zip(columns,e_columns)))
print cols
df1[cols+['Coordinates']].to_csv('Data/IRAS20050.csv')


## export parameters for IRAS20050
columns = ['Coordinates','R37','alpha','R','env_mass','env_mass_std','sLsun','sLsun_std','Lbol','inc','ext','s']
dftot = df.loc[df['Cluster'].isin(['IRAS20050'])]
dftot[columns].to_csv('Data/IRAS20050_params.csv')
dftot = df.loc[df['Cluster'].isin(['NGC2071'])]
dftot[columns].to_csv('Data/NGC2071_params.csv')


# load only isolated sources
isolated = clean_table.group_by('Property').groups[2]

alpha1 = isolated['alpha']
alpha2 = isolated['alpha2']

fig,(ax,ax2) = plt.subplots(1,2,figsize=figsize,facecolor=facecolor)
hist,bin_edges = np.histogram(alpha1,bins=30)
w = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1],hist,width=w,color=blue)
ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
ax.set_ylabel('Number of objects')
ax.grid(True)

hist,bin_edges = np.histogram(alpha2,bins=30)
ax2.bar(bin_edges[:-1],hist,width=w,color=blue)
ax2.set_xlabel(r'Spectral index $\alpha_{2.2-24}$')
ax2.set_ylabel('Number of objects')
ax2.grid(True)

fig.tight_layout()
fig.savefig('../Figures/SpectralIndex_cardini.pdf',dpi=300)



### alpha plot
# load alphas
#isolated.add_column(Column(np.zeros(len(isolated)),name="alpha3"))
isolated.add_column(Column(np.zeros(len(isolated)),name="e_alpha"))
isolated.add_column(Column(np.zeros(len(isolated)),name="e_alpha2"))

columnlist_alpha =['ks','i1','i2','i3','i4','F11','F19','F31','F37']
errorbars_alpha = ["e_"+col for col in columnlist_alpha]
wllist = Table(names=columnlist_alpha)
wllist.add_row([2.2,3.6,4.5,5.8,8.,11.1,19.7,31.5,37.1])
df['e_alpha']= 0.0
df['e_alpha2']= 0.0
df['alpha2']= 0.0
for i in range(len(isolated)):
	vals = np.array([isolated[wl][i] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
	errs = np.array([isolated[wl][i] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
	wav = np.array([wllist[wl][0] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
	vals *= 1e-17*const.c.value/wav
	errs *= 1e-17*const.c.value/wav
	vals = np.log10(vals)
	errs = np.log10(errs)
	wav = np.log10(wav)
	fit,cov = np.polyfit(wav,vals,1,w=errs,cov=True)
	isolated['alpha'][i] = fit[0]
	df.loc[isolated['SOFIA_name'][i],'alpha'] = fit[0]
	isolated['e_alpha'][i] = np.sqrt(cov[0,0])
	df.loc[isolated['SOFIA_name'][i],'e_alpha'] = np.sqrt(cov[0,0])
	

columnlist_alpha =['ks','i1','i2','i3','i4','m1']
errorbars_alpha = ["e_"+col for col in columnlist_alpha]
wllist = Table(names=columnlist_alpha)
wllist.add_row([2.2,3.6,4.5,5.8,8.,24])

for i in range(len(isolated)):
	if isolated.mask['i4'][i]==False:
		vals = np.array([isolated[wl][i] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
		errs = np.array([isolated[wl][i] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
		wav = np.array([wllist[wl][0] for wl in columnlist_alpha if isolated.mask[wl][i]==False])
		vals *= 1e-17*const.c.value/wav
		errs *= 1e-17*const.c.value/wav
		vals = np.log10(vals)
		errs = np.log10(errs)
		wav = np.log10(wav)
		if len(vals)>2:
			fit,cov = np.polyfit(wav,vals,1,w=errs,cov=True)
			isolated['alpha2'][i] = fit[0]
			isolated['e_alpha2'][i] = np.sqrt(cov[0,0])
			df.loc[isolated['SOFIA_name'][i],'alpha2'] = fit[0]
			df.loc[isolated['SOFIA_name'][i],'e_alpha2'] = np.sqrt(cov[0,0])
	else:
		isolated['alpha2'][i] = -100
		isolated['e_alpha2'][i] = 100
		isolated.mask['alpha2'][i] = True
		isolated.mask['e_alpha2'][i] = True
		
alpha1 = isolated['alpha']
alpha2 = isolated['alpha2']

# fig,(ax,ax2) = plt.subplots(1,2,figsize=figsize,facecolor=facecolor)
# hist,bin_edges = np.histogram(alpha1.compressed(),bins=np.arange(-1,4.5,0.15))  #np.arange(-1,3,0.15)
# w = bin_edges[1] - bin_edges[0]
# ax.bar(bin_edges[:-1],hist,width=w,color=blue)
# ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
# ax.set_ylabel('Number of objects')
# ax.grid(True)
# 
# hist,bin_edges = np.histogram(alpha2.compressed(),bins=np.arange(-1,4.5,0.15))
# w = bin_edges[1] - bin_edges[0]
# ax2.bar(bin_edges[:-1],hist,width=w,color=blue)
# ax2.set_xlabel(r'Spectral index $\alpha_{2.2-24}$')
# ax2.set_ylabel('Number of objects')
# ax2.grid(True)

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
hist,bin_edges = np.histogram(alpha1.compressed(),bins=np.arange(-1,3,0.15))  #np.arange(-1,3,0.15)
w = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1],hist,width=w,color=blue)
ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
ax.set_ylabel('Number of objects')
ax.grid(True)


fig.tight_layout()
fig.savefig('../Figures/SpectralIndex.pdf',dpi=300)
#plt.show()




# plot alpha histogram
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# hist,bin_edges = np.histogram(isolated['alpha'],bins=20)
# w = bin_edges[1] - bin_edges[0]
# ax.bar(bin_edges[:-1],hist,width=w,color=blue)
# ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
# ax.set_ylabel('Number of objects')
# ax.grid(True)
# fig.tight_layout()


### R plot
# First, need to select only the sources from Ophiuchus, NGC1333, NCG2071
# convert to pandas

df = clean_table.to_pandas()
df.index = df['SOFIA_name']
c = SkyCoord(df['RA'],df['DEC'],frame=FK5,unit=u.deg)
df['Coordinates'] = c.to_string('hmsdms',precision=1)

df = df.loc[df['Cluster'].isin(['IRAS20050'])]
print df.columns

columns = ['Coordinates','alpha','R','env_mass','env_mass_std','sLsun','sLsun_std','inc','ext','s']
df[columns].to_csv('Data/IRAS20050_params.csv')


df = isolated.to_pandas()
df.index = df['SOFIA_name']
c = SkyCoord(df['RA'],df['DEC'],frame=FK5,unit=u.deg)
df['Coordinates'] = c.to_string('hmsdms',precision=1)

df = df.loc[df['Cluster'].isin(['Oph','NGC1333'])]
print df.columns

### plot tables

columns = ['Coordinates','R37','alpha','R','env_mass','env_mass_std','calc_mass','sLsun','sLsun_std','Lbol','inc','ext','s']
df[columns].to_csv('Data/Oph_NGC1333_NGC2071.csv')

df = isolated.to_pandas()
df.index = df['SOFIA_name']
c = SkyCoord(df['RA'],df['DEC'],frame=FK5,unit=u.deg)
df['Coordinates'] = c.to_string('hmsdms',precision=1)
df = df.loc[df['Cluster'].isin(['Oph','NGC1333'])]
df = df.loc[df['R'] < 2]
print df[columns]


# plot R histogram
fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
hist,bin_edges = np.histogram(df['R'],bins=20)
w = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1],hist,width=w,color=blue)
ax.set_xlabel(r'$R$')
ax.set_ylabel('Number of objects')
ax.grid(True)

fig.tight_layout()
fig.savefig('../Figures/Rdistr.pdf')
#plt.show()

# plot env_mass histogram
fig,(ax,ax2) = plt.subplots(1,2,figsize=figsize,facecolor=facecolor)
hist,bin_edges = np.histogram(np.log10(df['env_mass']),bins=15)
w = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1],hist,width=w,color=blue)
ax.set_xlabel(r'$\log\ M_\mathrm{env}$ ($M_\odot$)')
ax.set_ylabel('Number of objects')
ax.grid(True)
fig.tight_layout()

# plot lum histogram
hist,bin_edges = np.histogram(np.log10(df['sLsun']),bins=15)
w = bin_edges[1] - bin_edges[0]
ax2.bar(bin_edges[:-1],hist,width=w,color=blue)
ax2.set_xlabel(r'$\log\ L_\mathrm{mod}$ ($L_\odot$)')
ax2.set_ylabel('Number of objects')
ax2.grid(True)
fig.tight_layout()
fig.savefig('../Figures/MassLumHist.pdf',dpi=300)

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(df['alpha'],np.log10(df['env_mass']),color=red)
X = df['alpha']
X = sm.add_constant(X)
model = sm.OLS(np.log10(df['env_mass']),X)
model = sm.WLS(np.log10(df['env_mass']),X,weights=1./(df['R'])**2)
#model = smf.ols('np.log10(env_mass) ~ alpha',data=df)
results=model.fit()
print(results.summary())
#prstd, iv_l, iv_u = wls_prediction_std(results)
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(df['alpha'],np.log10(df['env_mass']),color=blue)
# ax.plot(df['alpha'],results.fittedvalues,color=red)
# ax.plot(df['alpha'],iv_u,'--',color=red)
# ax.plot(df['alpha'],iv_l,'--',color=red)
# ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
# ax.set_ylabel('Fitted envelope mass')
ax.grid(True)
ax.set_xlabel(r'$\alpha_{2.2-37}$')
ax.set_ylabel(r'$\log\ M_\mathrm{env}$ ($M_\odot$)')
fig.tight_layout()
fig.savefig('../Figures/massVSalpha.pdf')

# plot mass vs alpha2
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(df['alpha2'],np.log10(df['env_mass']),color=blue)
# ax.set_xlabel(r'Spectral index $\alpha_{2.2-24}$')
# ax.set_ylabel('Fitted envelope mass')
# ax.grid(True)
# fig.tight_layout()
# #fig.savefig('../Figures/SpectralIndex.pdf')
# sns.regplot(df['alpha2'],np.log10(df['env_mass']),color=red)

# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# sns.regplot(df['alpha'],np.log10(df['env_mass']),color=red)
# sns.regplot(df['alpha2'],np.log10(df['env_mass']),color=green)


fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(df['alpha'],np.log10(df['sLsun']),color=red)
X = df['alpha']
X = sm.add_constant(X)
model = sm.OLS(np.log10(df['env_mass']),X)
model = sm.WLS(np.log10(df['sLsun']),X,weights=1./(df['sLsun_std']))
#model = smf.ols('np.log10(env_mass) ~ alpha',data=df)
results=model.fit()
print(results.summary())
# prstd, iv_l, iv_u = wls_prediction_std(results)
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(df['alpha'],np.log10(df['env_mass']),color=blue)
# ax.plot(df['alpha'],results.fittedvalues,color=red)
# ax.plot(df['alpha'],iv_u,'--',color=red)
# ax.plot(df['alpha'],iv_l,'--',color=red)
# ax.set_xlabel(r'Spectral index $\alpha_{2.2-37}$')
# ax.set_ylabel('Fitted envelope mass')
ax.grid(True)
ax.set_xlabel(r'$\alpha_{2.2-37}$')
ax.set_ylabel(r'$\log\ L_\mathrm{mod}$ ($L_\odot$)')
fig.tight_layout()
fig.savefig('../Figures/slumVSalpha.pdf')


# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# sns.regplot(df['alpha'],np.log10(df['sLsun']),color=red)
# sns.regplot(df['alpha2'],np.log10(df['sLsun']),color=green)

# plot Lbol vs fitted lum
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(np.log10(df['sLsun']),np.log10(df['Lbol']),color=blue)
# ax.plot(np.log10(df['sLsun']),np.log10(df['sLsun']),'--',color=grey)
# ax.set_xlim([np.log10(df['sLsun']).min()*0.95,np.log10(df['sLsun']).max()*1.05])
# ax.set_xlabel(r'Fitted luminosity')
# ax.set_ylabel('Bolometric luminosity')
# ax.grid(True)
# fig.tight_layout()
#fig.savefig('../Figures/SpectralIndex.pdf')

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(np.log10(df['Lbol']),np.log10(df['sLsun']),color=red)
vals = np.arange(min(np.log10(df['Lbol'])),2*max(np.log10(df['Lbol'])))
ax.plot(vals,vals,'--',color=grey)
ax.set_xlim([np.log10(df['Lbol']).min()*0.95,np.log10(df['Lbol']).max()*1.05])
fig.tight_layout()
ax.set_ylabel(r'$\log\ L_\mathrm{mod}$ ($L_\odot$)')
ax.set_xlabel(r'$\log\ L_\mathrm{bol}$ ($L_\odot$)')
X = np.log10(df['Lbol'])
X = sm.add_constant(X)
model = sm.WLS(np.log10(df['sLsun']),X,weights=1./(df['sLsun_std']))
#model = smf.ols('np.log10(env_mass) ~ alpha',data=df)
results=model.fit()
print(results.summary())

fig.savefig('../Figures/LbolVsLest.pdf')


fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(np.log10(df['env_mass']),-np.log10(df['Lbol'])+np.log10(df['sLsun']),color=red)
#vals = np.arange(min(np.log10(df['Lbol'])),2*max(np.log10(df['Lbol'])))
#ax.plot(vals,vals,'--',color=grey)
ax.set_xlim([np.log10(df['env_mass']).min()*0.95,np.log10(df['env_mass']).max()*1.05])
fig.tight_layout()
ax.set_ylabel(r'$\log\ L_\mathrm{mod}-\log\ L_\mathrm{bol}$')
ax.set_xlabel(r'$\log\ M_\mathrm{env}$ ($M_\odot$)')

fig.savefig('../Figures/LbolMinusLestVSMass.pdf')



# plot SOFIA color vs fitted lum
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(np.log10(df['F37'])-np.log10(df['F19']),np.log10(df['sLsun']),color=blue)
# ax.set_xlabel(r'[37]-[19]')
# ax.set_ylabel('Luminosity')
# ax.grid(True)
# fig.tight_layout()
# #fig.savefig('../Figures/SpectralIndex.pdf')

# plot SOFIA color vs fitted mass
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# ax.scatter(np.log10(df['F37'])-np.log10(df['F19']),np.log10(df['env_mass']),color=blue)
# ax.set_xlabel(r'[37]-[19]')
# ax.set_ylabel('Mass')
# ax.grid(True)
# fig.tight_layout()
# #fig.savefig('../Figures/SpectralIndex.pdf')
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)

#sns.regplot(np.log10(df['F37'])-np.log10(df['F19']),np.log10(df['env_mass']),color=red)
# sns.regplot(np.log10(df['F37'])-np.log10(df['i2']),np.log10(df['sLsun']),color=red)
# 
# X = df['alpha']
# X = sm.add_constant(X)
# #model = sm.OLS(np.log10(df['env_mass']),X)
# #model = sm.WLS(np.log10(df['env_mass']),X,weights=1./(df['env_masslog_std']))
# model = smf.ols('np.log10(env_mass) ~ np.log10(F37)+np.log10(F19)+np.log10(i4)+np.log10(i2)',data=df)
# results=model.fit()
# print(results.summary())
# 
# fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
# sns.regplot(np.log10(df['F37'])-np.log10(df['F19']),np.log10(df['i4'])-np.log10(df['i2']),color=red)

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(np.log10(df['env_mass']),df['inc'],color=red)
ax.grid(True)
ax.set_xlabel(r'$\log\ M_\mathrm{env}$ ($M_\odot$)')
ax.set_ylabel(r'Inclination angle (degrees)')
fig.tight_layout()
fig.savefig('../Figures/incVSmass.pdf')

fig,ax = plt.subplots(figsize=figsize,facecolor=facecolor)
sns.regplot(np.log10(df['calc_mass']),np.log10(df['env_mass']),color=red)
vals = np.arange(min(np.log10(df['env_mass'])-1),2*max(np.log10(df['env_mass'])))
ax.plot(vals,vals,'--',color='grey')
ax.grid(True)
ax.set_ylabel(r'$\log\ M_\mathrm{env,fitted}$ ($M_\odot$)')
ax.set_xlabel(r'$\log\ M_\mathrm{env,calc}$ ($M_\odot$)')
fig.tight_layout()
fig.savefig('../Figures/massEstvsCalc.pdf')

#columnlist = ['i1','i2','i3','i4','F11','F19','F31','F37']
# [3.6,4.5,5.8,8.,11.1,19.7,31.5,37.1]
#	ax.errorbar(wllist,nptable(sourcetable[columnlist][0])*1e-17*const.c.value/wllist,nptable(sourcetable[errorlist][0])*1e-17*const.c.value/wllist, 
#		c=color,alpha=alpha,linestyle="None",marker=marker,label = label,markersize=msize)
#

plt.show()


