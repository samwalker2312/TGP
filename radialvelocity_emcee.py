import emcee
import numpy as np
import matplotlib.pyplot as plt
import corner

class radvelminimiser_MCMC(object):

    def __init__(self, x, y, err):
        self.x = x
        self.y = y
        self.err = err

    def loglike(self, theta):
        a, b, c = theta
        model = a*np.sin(b*self.x+c)
        sigma2 = self.err**2.
        return -.5*np.sum((self.y-model)**2./sigma2)

    def logprior(self, theta):
        a, b, c = theta
        if 0 < a < 100 and 0 < b < 10 and 0 < c < 2*np.pi:
            return 0.0
        return -np.inf

    def logprob(self, theta, x, y, z):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def run(self):
        pos = [[50,1.,np.pi]] + .1*np.random.randn(100,3)
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))
        self.sampler.run_mcmc(pos, 100000, progress=True)

    def getsamples(self):
        self.flat_samples = self.sampler.get_chain(discard=30000, thin=15, flat=True)

    def plotcorner(self):
        fig = corner.corner(self.flat_samples)
        plt.show()

    def printvals(self):
        for i in range(self.ndim):
            val = np.percentile(self.flat_samples[:,i], (16,50,84))
            print(str(val[1]) + ' +' + str(val[2]-val[1]) + " " + str(val[0]-val[1]))

    def plotcurve(self):
        xplot = np.linspace(self.x[0],self.x[-1],5000)
        yplot = np.median(self.flat_samples[:,0])*np.sin(np.median(self.flat_samples[:,1])*xplot + np.median(self.flat_samples[:,2]))
        plt.figure()
        plt.errorbar(self.x,self.y,yerr=self.err, fmt='o')
        plt.plot(xplot, yplot)
        plt.show()

    def plotwalkers(self):
        fig, axes = plt.subplots(3)
        samples = self.sampler.get_chain()
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
        plt.show()

    def returnfinalvals(self):
        values = np.zeros((self.ndim))
        for i in range(self.ndim):
            val = np.percentile(self.flat_samples[:,i], (16,50,84))
            values[i] = val
        return values

    def calculateperiod(self):
        b = np.percentile(self.flat_samples[:,1], (16,50,84))
        b_err = np.array([b[0], b[2]])
        b = b[1]
        b_err -= b
        period = 2*np.pi/b
        period_err = period*b_err/b
        print('The period in days is ' + str(period) + " +" + str(period_err[1]) + ' ' + str(period_err[0]))
        self.period = period*24*3600
        self.period_err = period_err*24*3600

    def calculateorbrad(self, starmass, starmass_err):
        k = 6.674e-11/(4*np.pi**2.)
        k **= (1/3)
        self.orbitalradius = k*(starmass*self.period**2.)**(1/3)
        self.orbrad_err = k/3*np.sqrt(((starmass_err**2.)*(self.period/starmass)**(4/3)) + (4*(self.period_err**2.)*(starmass/self.period)**(2/3)))
        o = self.orbitalradius/1.496e11
        o_err = self.orbrad_err/1.496e11
        print('Orbital radius in AU is ' + str(o) + ' +' + str(o_err[1]) + ' -' + str(o_err[0]))

    def calculateplanetmass(self, starmass, starmass_err, i, i_err):
        a = np.percentile(self.flat_samples[:,0], (16,50,84))
        a_err = np.array([a[0], a[2]])
        a = a[1]
        a_err -= a
        k = (6.674e-11)**(-.5)
        i *= np.pi/180
        i_err *= np.pi/180
        sini = np.sin(i)
        sini_err = i_err*np.cos(i)
        self.planetmass = k*(starmass**.5)*(self.orbitalradius**.5)*a/sini
        self.planetmass_err = (k/sini)*np.sqrt( (self.orbitalradius*(a*starmass_err)**2./(4*starmass)) \
         + (starmass*(a*self.orbrad_err)**2./(4*self.orbitalradius)) + (self.orbitalradius*starmass*(a_err**2.)) \
          + (self.orbitalradius*starmass*(a**2.)*(sini_err**2.)*(np.tan(i)**-2.)) )
        m = self.planetmass/1.898e27
        m_err = self.planetmass_err/1.898e27
        print('Planet mass in Jupiter masses is ' + str(m) + ' +' + str(m_err[1]) + ' -' + str(m_err[0]))


def main():
    x = np.loadtxt('radvel.txt',comments='#', usecols=0)
    y =  np.loadtxt('radvel.txt',comments='#', usecols=1)
    err =  np.loadtxt('radvel.txt',comments='#', usecols=2)
    minimiser = radvelminimiser_MCMC(x,y,err)
    minimiser.run()
    minimiser.getsamples()
    minimiser.printvals()
    b = np.percentile(minimiser.flat_samples[:,1], (16,50,84))
    bpluserror = b[2] - b[1]
    bnegerror = b[0] - b[1]
    b = b[1]
    print("b = " +str(b))
    period = 2*np.pi/b
    periodpluserror = period*bpluserror/b
    periodnegerror = (period*bnegerror/b)
    print("Period in days is " + str(period) + " +" + str(periodpluserror) + " " + str(periodnegerror))
    period *= 24*3600
    periodpluserror *= 24*3600
    periodnegerror *= 24*3600
    starmass = 2.5e30 #float(input("Input the mass of the host star: "))
    G = 6.674e-11
    orbitalradius = (starmass*G*period**2./(4.*np.pi**2.))**(1/3)
    orbradpluserror = 2/3*periodpluserror*orbitalradius/period
    orbradnegerror = 2/3*periodnegerror*orbitalradius/period
    print("Orbital radius in AU is " + str(orbitalradius/1.496e11) + " +" + str(orbradpluserror/1.496e11) + " " + str(orbradnegerror/1.496e11))
    planetvel = (G*starmass/orbitalradius)**.5
    #print("Orbital velocity is " + str(planetvel))
    planetmass = starmass*np.median(minimiser.flat_samples[:,0])/planetvel
    print("Planet mass assuming inc=90 in Mj is " + str(np.abs(planetmass/(1.898e27))))
    minimiser.plotcurve()
