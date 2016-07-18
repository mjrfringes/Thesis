from astropy import constants as const
import astropy.units as u
import numpy as np
from numpy import pi


# free-fall time
n = 1000*u.cm**(-3)
rho = 2.33*u.M_p*n
tff = np.sqrt(3.*pi/(32.*const.G*rho))
print n
print "Free fall time for rho = ",rho.to(u.g/u.cm**3),":",tff.to(u.yr)

# Jean's length
cs = 0.2*u.km/u.s
lambdaJ = cs*tff
print "Jean's length:",lambdaJ.to(u.pc)

# Jean's mass
MJ = 4.*pi*rho*(lambdaJ/2.)**3/3.
print "Jean's mass:",MJ.to(u.M_sun)

# BE sphere
BE = 1.182*cs**3/np.sqrt(const.G**3*rho)
print "Bonnor-Ebert mass (thermal):",BE.to(u.M_sun)

# BE sphere
Pext = 3e5*const.k_B*u.K/u.cm**3
BE2 = 1.182*cs**4/np.sqrt(const.G**3*Pext)
print "Bonnor-Ebert mass (turbulent):",BE2.to(u.M_sun), "= ", BE2.to(u.M_sun)/MJ.to(u.M_sun)," MJ"

# BE radius
BR = 0.486*cs/np.sqrt(const.G*rho)
print "Bonnor-Ebert radius:",BR.to(u.pc)

val = 4.*pi/(3.*8.)* ((3.*pi)/(32.))**1.5
print val

# Rtheta
lam = 70*u.um
D = 1*u.m
Rtheta = 0.43*2.44*(lam.to(u.m)/D)/4.848e-6
print "Rtheta:",Rtheta

# Rlinear
d = 500*u.pc
Rlinear = 0.43*2.44*(lam.to(u.m)/D)*500*u.pc
print "Rlinear:",Rlinear
