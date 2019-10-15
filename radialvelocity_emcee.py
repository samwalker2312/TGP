import emcee
import numpy as np
import matplotlib.pyplot as plt

class MCradvelminim(object):

    def __init__(self, x, y, err):
        self.x = x
        self.y = y
        self.err = err

    def loglike(self, theta):
        a, b, c, d, log_f = theta
        model = a*np.sin(b*self.x+c)
        sigma2 = self.err**2. + model**2.*np.exp(2*log_f)
        return -.5*np.sum((self.y-model)**2./sigma2 + np.log(sigma2))

    def logprior(self, theta):
        a, b, c, d, log_f = theta
        if -200 < a < 200 and 0. < b < 5000. and 0 < c < 1000 and -1000 < d < 1000 and -10. <       log_f < 1.:
            return 0.0
        return -np.inf

    def logprob(self, theta, x, y, z):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def run(self):
        pos = [[0,100,0,0,0]] + [[.1,1,1,.1,.01]]*np.random.randn(50,5)
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))
        self.sampler.run_mcmc(pos, 100000, progress=True)

    def getsamples(self):
        self.flat_samples = self.sampler.get_chain(discard=200, thin=15, flat=True)

    def printvals(self):
        for i in range(self.ndim-1):
            print(np.median(self.flat_samples[:,i]))

    def plotcurve(self):
        xplot = np.linspace(self.x[0],self.x[-1],500)
        yplot = np.median(self.flat_samples[:,0])*np.sin(np.median(self.flat_samples[:,1])*xplot + np.median(self.flat_samples[:,2])) + np.median(self.flat_samples[:,3])
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
    minimiser.printvals()
    minimiser.plotcurve()

main()
