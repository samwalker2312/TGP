import emcee
import numpy as np
import matplotlib.pyplot as plt

class MCradvelminim(object):

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
        if 0 < a < 500 and 0. < b < 5000. and 0 < c < 1000:
            return 0.0
        return -np.inf

    def logprob(self, theta, x, y, z):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def run(self):
        pos = [[50,10,0]] + [[1,1,1]]*np.random.randn(50,3)
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))
        self.sampler.run_mcmc(pos, 100000, progress=True)

    def getsamples(self):
        self.flat_samples = self.sampler.get_chain(discard=200, thin=15, flat=True)

    def printvals(self):
        for i in range(self.ndim):
            print(np.median(self.flat_samples[:,i]))

    def plotcurve(self):
        xplot = np.linspace(self.x[0],self.x[-1],500)
        yplot = np.median(self.flat_samples[:,0])*np.sin(np.median(self.flat_samples[:,1])*xplot + np.median(self.flat_samples[:,2]))
        plt.figure()
        plt.errorbar(self.x,self.y,yerr=self.err, fmt='o')
        plt.plot(xplot, yplot)
        plt.show()

def main():
    x = np.loadtxt('radvel.txt',comments='#', usecols=0)
    y =  np.loadtxt('radvel.txt',comments='#', usecols=1)
    err =  np.loadtxt('radvel.txt',comments='#', usecols=2)
    minimiser = MCradvelminim(x,y,err)
    minimiser.run()
    minimiser.getsamples()
    #minimiser.printvals()
    period = 2*np.pi/np.median(minimiser.flat_samples[:,1])
    print("Period in days is " + str(period))
    starmass = float(input("Input the mass of the host star: "))
    G = 6.674e-11
    planetradius = (starmass*G*period**2./(4.*np.pi**2.))**(1/3)
    print("Radius in Rj is " + str(planetradius/(69911e3)))
    planetvel = (G*starmass/planetradius)**.5
    #print("Orbital velocity is " + str(planetvel))
    planetmass = starmass*np.median(minimiser.flat_samples[:,0])/planetvel
    print("Planet mass assuming zero inclination in Mj is " + str(np.abs(planetmass/(1.898e27))))
    minimiser.plotcurve()

main()
