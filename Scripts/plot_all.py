import os

plotlist = ['BETTII_transmission',
'Atmo_transmission',
'plot_dust_types',
'plot_visibilities',
'plot_QE_PtGrey',
'plot_SOFIA_bands',
'plot_alpha']

for plot in plotlist:
	print "Executing script %s.py..." % plot
	os.system('python %s.py' % plot)