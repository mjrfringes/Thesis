import os

plotlist = ['BETTII_transmission',
'Atmo_transmission',
'plot_dust_types']

for plot in plotlist:
	print "Executing script %s.py..." % plot
	os.system('python %s.py' % plot)