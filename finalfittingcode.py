import numpy as np
from lightcurve_emcee import lightcurveminimiser_MCMC
from radialvelocity_emcee import radvelminimiser_MCMC

def calcdensity(mass, mass_err, radius, radius_err):
    k = 3./(4.*np.pi)
    p = k*mass*(radius**-3.)
    p_err = k*np.sqrt( (mass_err*radius**-3.)**2. + (3.*mass*radius_err*(radius**-4.))**2.)
    return p, p_err

# this takes everything in SI units remember
def calcefftemp(startemp, startemp_err, starrad, starrad_err, orbitalradius, orbitalradius_err):
    teq = startemp*(.7**.25)*(.5*starrad/orbitalradius)**.5
    radoverorbrad = starrad/orbitalradius
    radoverorbrad_err = (radoverorbrad)*np.sqrt((starrad_err/starrad)**2. + (orbitalradius_err/orbitalradius)**2.)
    teq_err = (.7**.25)*np.sqrt( (.5*radoverorbrad*startemp_err**2.) + ((startemp**2.)/(8*radoverorbrad) * (radoverorbrad_err**2.)) )
    return teq, teq_err

radvel_x = np.loadtxt('radvel.txt',comments='#', usecols=0)
radvel_y = np.loadtxt('radvel.txt',comments='#', usecols=1)
radvel_err = np.loadtxt('radvel.txt',comments='#', usecols=2)

lightcurve_x = np.loadtxt('lightcurvedata.txt',comments='#', usecols=0)
starttime = -lightcurve_x[0]
lightcurve_x += starttime
print("Starttime = " + str(starttime))
lightcurve_y =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=1)
lightcurve_err =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=2)

starmass = 2.5e30
starmass_err = 0
starrad = 1.2*695700e3
starrad_err = 0
#startemp = ########
#startemp_err = #########

radvelminimiser = radvelminimiser_MCMC(radvel_x,radvel_y,radvel_err)
radvelminimiser.run()
radvelminimiser.getsamples()
radvelminimiser.printvals()
radvelminimiser.plotcurve()
radvelminimiser.plotcorner()
#radvelminimiser.plotwalkers()
radvelminimiser.calculateperiod()
radvelminimiser.calculateorbrad(starmass, starmass_err)

lightcurveminimiser = lightcurveminimiser_MCMC(lightcurve_x, lightcurve_y, lightcurve_err)
period = radvelminimiser.period/(24*3600) # converts period from seconds to days
limbdark = 'quadratic'
orbitalradius = radvelminimiser.orbitalradius/starrad
lightcurveminimiser.createmodel(period, limbdark, orbitalradius)
#lightcurveminimiser.initemcee_fixecc()
lightcurveminimiser.initemcee_fitall()
iter = 10000
lightcurveminimiser.run(iter)
lightcurveminimiser.flattensamples(int(.3*iter))
lightcurveminimiser.setfinalparams()
lightcurveminimiser.plotcurve()
#lightcurveminimiser.plotcorner()
#lightcurveminimiser.plotwalkers()
inc, inc_err = lightcurveminimiser.returninc()
rp, rp_err = lightcurveminimiser.returnrp()
k = starrad/69911e3
print("Radius in R is " + str(rp*k) + ' +' + str(k*(rp_err[1])) + ' ' + str(k*(rp_err[0])))
radvelminimiser.calculateplanetmass(starmass, starmass_err, inc, inc_err)
calcdensity(radvelminimiser.planetmass, radvelminimiser.planetmass_err, radvelminimiser.period, radvelminimiser.period_err)
