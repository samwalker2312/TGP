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
        if 0 < a < 150 and 0 < b < 50 and 0 < c < 2*np.pi:
            return 0.0
        return -np.inf

    def logprob(self, theta, x, y, z):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def run(self):
        pos = [[50,1.,np.pi]] + np.random.randn(100,3)
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))
        self.sampler.run_mcmc(pos, 100000, progress=True)

    def getsamples(self):
        self.flat_samples = self.sampler.get_chain(discard=2000, thin=15, flat=True)

    def printvals(self):
        for i in range(self.ndim):
            print(np.median(self.flat_samples[:,i]))

    def plotcurve(self):
        xplot = np.linspace(self.x[0],self.x[-1],5000)
        yplot = np.median(self.flat_samples[:,0])*np.sin(np.median(self.flat_samples[:,1])*xplot + np.median(self.flat_samples[:,2]))
        plt.figure()
        plt.errorbar(self.x,self.y,yerr=self.err, fmt='o')
        plt.plot(xplot, yplot)
        plt.show()
        fig = corner.corner(self.flat_samples)
        plt.show()
        fig, axes = plt.subplots(3)
        samples = self.sampler.get_chain()
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
        plt.show()

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
    bnegerror = b[1] - b[0]
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
    print("Orbital radius in AU is " + str(orbitalradius/(1.496e11)))
    planetvel = (G*starmass/orbitalradius)**.5
    #print("Orbital velocity is " + str(planetvel))
    planetmass = starmass*np.median(minimiser.flat_samples[:,0])/planetvel
    print("Planet mass assuming inc=90 in Mj is " + str(np.abs(planetmass/(1.898e27))))
    minimiser.plotcurve()

main()
