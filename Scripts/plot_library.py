import seaborn.apionly as sns


textwidth = 5.901 # in inches
figsize = (textwidth,textwidth/2.)
facecolor = 'white'


colors = sns.xkcd_palette(['pale red','medium green','denim blue','amber','dusty purple',
	'faded green','windows blue','greyish'])
blue = sns.xkcd_rgb['denim blue']
red = sns.xkcd_rgb['pale red']
green = sns.xkcd_rgb['medium green']
amber = sns.xkcd_rgb['amber']
purple = sns.xkcd_rgb['dusty purple']
fadedgreen = sns.xkcd_rgb['faded green']
windowsblue = sns.xkcd_rgb['windows blue']
grey = sns.xkcd_rgb['greyish']