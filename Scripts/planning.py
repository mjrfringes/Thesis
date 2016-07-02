import astropy.units as u
from astropy.coordinates import EarthLocation,SkyCoord
from pytz import timezone
from astroplan import Observer,FixedTarget,AltitudeConstraint,AtNightConstraint
from astroplan import observability_table,time_grid_from_range
from astroplan.plots import plot_sky
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from plot_library import *

longitude = '-104.2455'
latitude = '+34.4714'
elevation = 37000 * u.m
location = EarthLocation.from_geodetic(longitude, latitude, elevation)

observer = Observer(name='Fort Sumner',
               location=location,
               pressure=0.615 * u.bar,
               relative_humidity=0.11,
               temperature=0 * u.deg_C,
               timezone=timezone('US/Mountain'),
               description="Launch")
               
time_range = Time(["2016-09-15 20:00","2016-09-16 06:00"])
time_grid = time_grid_from_range(time_range)


clean_table = pickle.load(open('data/clean_table.data','r'))

# add ratios
clean_table['R19'] = clean_table['R50_19']/clean_table['R50_cal_19']
clean_table['R31'] = clean_table['R50_31']/clean_table['R50_cal_31']
clean_table['R37'] = clean_table['R50_37']/clean_table['R50_cal_37']

clean_table.sort(['F37','R37'])
clean_table.reverse()
clean_table = clean_table[clean_table['F37']>20]
print clean_table['SOFIA_name','RA','DEC']

#targets = [FixedTarget(coord=SkyCoord(ra=334.82557*u.deg,dec=63.313065*u.deg),name='S140.5')]
#targets = [FixedTarget(coord=SkyCoord(ra=ra*u.deg,dec=dec*u.deg),name=name) for name,ra,dec in clean_table['SOFIA_name','RA','DEC']]


target_table_string = """# name ra_degrees dec_degrees
S140 334.82654 63.313836
CepheusA 344.07913 62.031778
NGC7129 325.77663 66.115386
NGC2264 100.29251 9.4925956
NGC2071 86.770544 0.36287845
NGC1333 52.292875 31.365424
Ophiuchus 246.78931 -24.621749
IRAS20050+2720 301.77718 27.481741"""
# Read in the table of targets
from astropy.io import ascii
target_table = ascii.read(target_table_string)
targets = [FixedTarget(coord=SkyCoord(ra=ra*u.deg,dec=dec*u.deg),name=name) for name,ra,dec in target_table]

constraints = [AltitudeConstraint(10*u.deg,75*u.deg)]#[AltitudeConstraint(20*u.deg,75*u.deg),AtNightConstraint.twilight_civil()]

tab = observability_table(constraints,observer,targets,time_range=time_range)

print tab

fig= plt.figure(figsize=(textwidth,0.7*textwidth),dpi=120)
cmap = cm.Set1             # Cycle through this colormap

for i, target in enumerate(targets):
    ax = plot_sky(target, observer, time_grid, 
                  style_kwargs=dict(color=cmap(float(i)/len(targets)),
                                    label=target.name,s=2))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

legend=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
legend.get_frame().set_facecolor('w')
#fig.tight_layout()
fig.savefig('../Figures/TargetPlot.pdf',dpi=300)
plt.show()
