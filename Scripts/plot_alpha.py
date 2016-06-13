import numpy as np
import matplotlib.pyplot as plt
from plot_library import *
from astropy.table import Table
import pickle
import pandas as pd
from astropy.coordinates import SkyCoord


clean_table = pickle.load(open('data/clean_table.data','r'))

isolated = clean_table.group_by('Property').groups[2]
#clean_table['SOFIA_name','Property','F11','F19','F31','F37'].more()
alpha1 = isolated['alpha']
alpha2 = isolated['alpha2']

clean_table['R19'] = clean_table['R50_19']/clean_table['R50_cal_19']
clean_table['R31'] = clean_table['R50_31']/clean_table['R50_cal_31']
clean_table['R37'] = clean_table['R50_37']/clean_table['R50_cal_37']

df = clean_table.to_pandas()
df.index = df['SOFIA_name']
print df.columns
IRASdf = df[['F11','F19','F31','F37']].loc[['IRAS20050.1','IRAS20050.2','IRAS20050.3','IRAS20050.4','IRAS20050.5','IRAS20050.8']]
print IRASdf
IRASdf.to_csv('data/IRASdf.csv')
#print IRASdf[['F11','F19','F31','F37']].sum(axis=0)
#print df[['F11','F19','F31','F37']].loc['IRAS20050.8']

IRASdf = df.loc[['IRAS20050.1','IRAS20050.2','IRAS20050.3','IRAS20050.4','IRAS20050.5','IRAS20050.6','IRAS20050.7']]
IRASdf = Table.from_pandas(IRASdf)
fieldlist = ['SOFIA_name','RA','DEC','R19','R31','R37',
'j','e_j','h','e_h','ks','e_ks',
'i1','e_i1','i2','e_i2','i3','e_i3','i4','e_i4','m1','e_m1',
'F11','e_F11','F19','e_F19','F31','e_F31','F37','e_F37',
'H70','e_H70','H250','e_H250','H350','e_H350','H500','e_H500'
]
print IRASdf[fieldlist]


NGC2071df = df[['F11','F19','F31','F37']].loc[['NGC2071.1','NGC2071.2','NGC2071.3','NGC2071.7']]
print NGC2071df
NGC2071df.to_csv('data/NGC2071df.csv')
df2071sum = {'sum':NGC2071df[['F11','F19','F31','F37']].sum(axis=0)}
df2071sum = pd.DataFrame(df2071sum,index=['F11','F19','F31','F37'])
df2071tot = {'tot':df[['F11','F19','F31','F37']].loc['NGC2071.7']}
df2071tot = pd.DataFrame(df2071tot,index=['F11','F19','F31','F37'])

#df2071 = pd.merge(df2071sum,df2071tot)
#print df2071sum,df2071tot,df2071

S140df = df[['F11','F19','F31','F37']].loc[['S140.1','S140.2','S140.3','S140.4','S140.5','S140.6','S140.7','S140.8']]
print S140df
S140df.to_csv('data/S140df.csv')

table = pickle.load(open('data/totsourcetable_fits.data','r'))
df = table.to_pandas()
df.index = df['SOFIA_name']

bands =['i1','i2','i3','i4']
megbands = [val+"_megeath" for val in bands]
guthbands = [val+"_guth" for val in bands]
IRASlist = ['IRAS20050.1','IRAS20050.3','IRAS20050.6','IRAS20050.7']
print df[bands+guthbands].loc[IRASlist]

NGClist = ['NGC2071.1','NGC2071.3','NGC2071.4','NGC2071.5']

#print df.dropna(axis =1,subset = 'i4_megeath')
print df[guthbands].dropna()
d = {'i1': (df['i1'].loc[NGClist] - df['i1'+"_megeath"].loc[NGClist])/df['i1'].loc[NGClist],
	'i2': (df['i2'].loc[NGClist] - df['i2'+"_megeath"].loc[NGClist])/df['i2'].loc[NGClist],
	'i3': (df['i3'].loc[NGClist] - df['i3'+"_megeath"].loc[NGClist])/df['i3'].loc[NGClist],
	'i4': (df['i4'].loc[NGClist] - df['i4'+"_megeath"].loc[NGClist])/df['i4'].loc[NGClist],}
ddf = pd.DataFrame(d,index=df['i1'].loc[NGClist].index)
dguth = {'i1': (df['i1'].loc[IRASlist] - df['i1'+"_guth"].loc[IRASlist])/df['i1'].loc[IRASlist],
	'i2': (df['i2'].loc[IRASlist] - df['i2'+"_guth"].loc[IRASlist])/df['i2'].loc[IRASlist],
	'i3': (df['i3'].loc[IRASlist] - df['i3'+"_guth"].loc[IRASlist])/df['i3'].loc[IRASlist],
	'i4': (df['i4'].loc[IRASlist] - df['i4'+"_guth"].loc[IRASlist])/df['i4'].loc[IRASlist],}
ddfguth = pd.DataFrame(dguth,index=df['i1'].loc[IRASlist].index)

new_df = ddf.append(ddfguth)
new_df.to_csv('data/compare_guth_megeath.csv')
print new_df

df = isolated.to_pandas()
descrAlpha= df[['alpha','alpha2']].describe()
descrAlpha.to_csv('data/descrAlpha.csv')

print df[['F37','e_F37']]

fig,(ax,ax2) = plt.subplots(1,2,figsize=figsize,facecolor=facecolor)
hist,bin_edges = np.histogram(alpha1,bins=np.arange(0,0.5,0.05))
w = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1],hist,width=w,color=blue)
ax.set_xlabel('Spectral index')
ax.set_ylabel('Number of objects')
ax.grid(True)
ax.set_xticks([0.0,0.1,0.2,0.3,0.4])

hist,bin_edges = np.histogram(alpha2,bins=np.arange(0,0.5,0.05))
ax2.bar(bin_edges[:-1],hist,width=w,color=blue)
ax2.set_xlabel('Spectral index')
ax2.set_ylabel('Number of objects')
ax2.grid(True)
ax2.set_xticks([0.0,0.1,0.2,0.3,0.4])

fig.tight_layout()
fig.savefig('../Figures/SpectralIndex.pdf')
plt.show()
