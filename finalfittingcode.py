import numpy as np

class derivedparams(object):

    def __init__(self, filename):
        self.data = np.loadtxt(filename)

    def calculateplanetmass(self, starmass, starmass_err):
        a = self.data[0,1]
        a_err = np.array([self.data[0,0],self.data[0,2]])
        k = (6.674e-11)**(-0.5)
        i = self.data[7,1]
        i_err = np.array([self.data[7,0],self.data[7,2]])
        i *= np.pi/180
        i_err *= np.pi/180
        self.orbitalradius = np.abs(self.data[4,1])
        self.orbrad_err = np.array([self.data[4,0],self.data[4,2]])
        sini = np.sin(i)
        sini_err = i_err*np.cos(i)
        self.planetmass = k*(starmass**.5)*(self.orbitalradius**.5)*a/sini
        self.planetmass_err = (k/sini)*np.sqrt( (self.orbitalradius*(a*starmass_err)**2./(4*starmass)) \
         + (starmass*(a*self.orbrad_err)**2./(4*self.orbitalradius)) + (self.orbitalradius*starmass*(a_err**2.)) \
          + (self.orbitalradius*starmass*(a**2.)*(sini_err**2.)*(np.tan(i)**-2.)) )
        m = self.planetmass/1.898e27
        m_err = self.planetmass_err/1.898e27
        print('Planet mass in Jupiter masses is ' + str(m) + ' +' + str(m_err[1]) + ' -' + str(m_err[0]))

    def calcdensity(self, starrad, starrad_err):
        mass = self.planetmass
        mass_err = self.planetmass_err
        radius = self.data[6,1]
        radius_err = np.array([self.data[6,0], self.data[6,2]])
        si_radius = starrad*radius
        si_radius_err = si_radius*np.sqrt((radius_err/radius)**2. + (starrad_err/starrad)**2.)
        k = 3./(4.*np.pi)
        p = k*mass*(si_radius**-3.)/1000
        p_err = k*np.sqrt( (mass_err*si_radius**-3.)**2. + (3.*mass*si_radius_err*(si_radius**-4.))**2.)/1000
        print("Density in grams per cubic cm is " + str(p) + ' +' + str(p_err[1]) + ' -' + str(p_err[0]))

    # this takes everything in SI units remember
    def calcefftemp(self, startemp, startemp_err, starrad, starrad_err):
        orbitalradius = self.data[4,1]
        orbitalradius_err = np.array([self.data[4,0], self.data[4,2]])
        teq = startemp*(.7**.25)*(.5*starrad/orbitalradius)**.5
        radoverorbrad = starrad/orbitalradius
        radoverorbrad_err = (radoverorbrad)*np.sqrt((starrad_err/starrad)**2. + (orbitalradius_err/orbitalradius)**2.)
        teq_err = (.7**.25)*np.sqrt( (.5*radoverorbrad*startemp_err**2.) + ((startemp**2.)/(8*radoverorbrad) * (radoverorbrad_err**2.)) )
        print("Equilibrium temp. in K is " + str(teq) + ' +' + str(teq_err[1]) + ' -' + str(teq_err[0]))

    def printvalsinrightunits(self, starrad, starrad_err):
        rv_amp = self.data[0,1]
        rv_amp_error = np.abs(np.array([self.data[0,0], self.data[0,2]]))
        print('RV amplitude in m/s = ' + str(rv_amp) + ' +' + str(rv_amp_error[1]) + ' -' + str(rv_amp_error[0]))

        period = self.data[1,1]
        period_error = np.abs(np.array([self.data[1,0], self.data[1,2]]))
        print('Period in days = ' + str(period) + ' +' + str(period_error[1]) + ' -' + str(period_error[0]))

        phase = self.data[2,1]
        phase_error = np.abs(np.array([self.data[2,0], self.data[2,2]]))
        print('Phase in days = ' + str(phase) + ' +' + str(phase_error[1]) + ' -' + str(phase_error[0]))

        gamma = self.data[3,1]
        gamma_error = np.abs(np.array([self.data[3,0], self.data[3,2]]))
        print('Systematic offset in m/s = ' + str(gamma) + ' +' + str(gamma_error[1]) + ' -' + str(gamma_error[0]))

        orbrad = self.data[4,1]/1.496e11
        orbrad_error = np.abs(np.array([self.data[4,0], self.data[4,2]]))/1.496e11
        print('Orbital radius in = ' + str(orbrad) + ' +' + str(orbrad_error[1]) + ' -' + str(orbrad_error[0]))

        t0 = self.data[5,1]
        t0_error = np.abs(np.array([self.data[5,0], self.data[5,2]]))
        print('t0 in days = ' + str(t0) + ' +' + str(t0_error[1]) + ' -' + str(t0_error[0]))

        rp = self.data[6,1]*starrad/7.1492e7
        rp_error = rp*np.sqrt((np.array([self.data[6,0], self.data[6,2]])/self.data[6,1])**2. + (starrad_err/starrad)**2.)
        print('Planetary radius in Jupiter radii = ' + str(rp) + ' +' + str(rp_error[1]) + ' -' + str(rp_error[0]))

        inc = self.data[7,1]
        inc_error = np.abs(np.array([self.data[7,0], self.data[7,2]]))
        print('Inclination in degrees = ' + str(inc) + ' +' + str(inc_error[1]) + ' -' + str(inc_error[0]))

        ecc = self.data[8,1]
        ecc_error = np.abs(np.array([self.data[8,0], self.data[8,2]]))
        print('Eccentricity in degrees = ' + str(ecc) + ' +' + str(ecc_error[1]) + ' -' + str(ecc_error[0]))

        w = self.data[9,1]
        w_error = np.abs(np.array([self.data[9,0], self.data[9,2]]))
        print('w in degrees = ' + str(w) + ' +' + str(w_error[1]) + ' -' + str(w_error[0]))


def main():
    startemp = 5850
    startemp_err = np.array([50,50])
    starmass = 2.08e30
    starmass_err = 3e28*np.array([1,1])
    starrad = 695700e3
    starrad_err = .03*695700e3*np.array([1,1])
    filename = 'finaldata_uniform.txt'
    dervparam = derivedparams(filename)
    dervparam.printvalsinrightunits(starrad,starrad_err)
    dervparam.calculateplanetmass(starmass, starmass_err)
    dervparam.calcdensity(starrad, starrad_err)
    dervparam.calcefftemp(startemp, startemp_err, starrad, starrad_err)

main()
