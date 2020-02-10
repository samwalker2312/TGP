import emcee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner
from scipy.stats.distributions import chi2

class radvelminimiser_MCMC(object):

    def __init__(self, x, y, err):
        self.x = x
        self.y = y
        self.err = err

    def loglike(self, theta):
        a, b, c, d = theta
        factor = 2*np.pi/b
        model = a*np.sin(factor*(self.x + c)) + d
        sigma2 = self.err**2.
        return -.5*np.sum(((self.y-model)**2.)/sigma2)

    def logprior(self, theta):
        a, b, c, d = theta
        if 0 < a < 300 and 0. < b < 7 and 0 < c < 7 and -200 < d < 200:
            return 0.0
        return -np.inf

    def logprob(self, theta, x, y, z):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

    def loglike_exofast(self, theta):
        A, B, Tp, P, y, ydot = theta
        t0 = np.average(self.x)
        x = 2*np.pi*(self.x-Tp)/P
        model = A*np.cos(x) + B*np.sin(x) + y + ydot*(self.x - t0)
        sigma2 = self.err**2.
        return -.5*np.sum(((self.y-model)**2.)/sigma2)

    def logprior_exofast(self, theta):
        A, B, Tp, P, y, ydot = theta
        if 0<A<300 and 0<B<300 and self.x[0]<Tp<self.x[-1] and 0<P<4 and -1e8<y<1e8 and -1e8<ydot<1e8:
            return 0.
        return -np.inf

    def logprob_exofast(self, theta, x, y, z):
        lp = self.logprior_exofast(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike_exofast(theta)

    def run(self, iter):
        pos = [[150.,3.5, 3.5, 0.]] + np.random.randn(50,4)
        self.nwalkers, self.ndim = pos.shape

        filename = "sampler.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(self.nwalkers, self.ndim)

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err), backend=backend)
        self.sampler.run_mcmc(pos, iter, progress=True)

    def run_exofast(self, iter):
        pos = [[200.,200.,self.x[int(len(self.x)/2)], 2, 0, 0]] + np.random.randn(100,6)
        self.nwalkers, self.ndim = pos.shape

        filename = "sampler_exofast.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob_exofast, args = (self.x,self.y,self.err))
        self.sampler.run_mcmc(pos, iter, progress=True)

    def getsamples(self, cutoff):
        tau = self.sampler.get_autocorr_time()
        print(tau)
        taumax = np.nanmax(tau)
        print(taumax)
        if cutoff < 5*taumax:
            cutoff = int(5*taumax)
        if int(np.nanmin(tau)/2) == 0:
            taumin = 15
        else:
            taumin = np.nanmin(tau)
        print(taumin)
        self.flat_samples = self.sampler.get_chain(discard=cutoff, thin=int(taumin/2), flat=True)

    def plotcorner(self):
        labels = [r'$\textrm{Amplitude}$', r'$\textrm{Period}$', r'$\Delta\textrm{Phase}$', r'$\gamma$']
        fig = corner.corner(self.flat_samples, labels=labels, quantiles = [0.16, .5, .84], show_titles = True, use_math_text = True, smooth = True, title_kwargs={"fontsize": 20}, smooth1d = True, label_kwargs={'fontsize':20})
        plt.savefig('radvel_corner.png')

    def printvals(self):
        for i in range(self.ndim):
            val = np.percentile(self.flat_samples[:,i], (16,50,84))
            print(str(val[1]) + ' +' + str(val[2]-val[1]) + " " + str(val[0]-val[1]))

    def plotcurve(self):
        factor = 2*np.pi/np.median(self.flat_samples[:,1])
        xplot = np.linspace(-.5,2.5)
        yplot = np.median(self.flat_samples[:,0])*np.sin(factor*(xplot + np.median(self.flat_samples[:,2]))) + np.median(self.flat_samples[:,3])

        model = np.median(self.flat_samples[:,0])*np.sin(factor*(self.x + np.median(self.flat_samples[:,2]))) + np.median(self.flat_samples[:,3])
        chisq = np.sum(((self.y-model)**2.)/self.err)
        print(chisq)
        print(chi2.sf(chisq, 7))

        fig = plt.figure()
        ax1 = fig.add_axes((.1,.3,.8,.6))
        ax2 = fig.add_axes((.1,.1,.8,.2))
        ax2.set_xlabel(r"\textrm{Time (days)}")
        ax1.set_ylabel(r"$\textrm{Velocity (ms}^{-1})$")
        ax2.set_ylabel(r"$\textrm{Residuals}$")
        ax1.set_xlim(-.5,2.5)
        ax2.set_xlim(-.5,2.5)
        ax1.errorbar(self.x,self.y,yerr=self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax2.errorbar(self.x, (model-self.y), yerr = self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax1.plot(xplot, yplot, color='black')
        ax2.plot(xplot, np.zeros_like(xplot), ':',color='black')
        plt.savefig('radvel_curve.png')

    def plotcurve_exofast(self):
        xplot = np.linspace(self.x[0],self.x[-1],5000)
        A = np.median(self.flat_samples[:,0])
        B = np.median(self.flat_samples[:,1])
        Tp = np.median(self.flat_samples[:,2])
        P = np.median(self.flat_samples[:,3])
        y = np.median(self.flat_samples[:,4])
        ydot = np.median(self.flat_samples[:,5])
        t0 = np.average(self.x)
        x = 2*np.pi*(xplot - Tp)/P
        yplot = A*np.cos(x) + B*np.sin(x) + y + ydot*(xplot - t0)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel(r"\textrm{Time (days)}")
        ax1.set_ylabel(r"$\textrm{Velocity (ms}^{-1})$")
        ax1.errorbar(self.x,self.y,yerr=self.err, fmt='o')
        ax1.plot(xplot, yplot)
        plt.savefig('radvel_curve_exofast.png')

    def plotwalkers(self):
        fig, axes = plt.subplots(self.ndim)
        samples = self.sampler.get_chain()
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
        plt.savefig('radvel_walkers.png')

    def calculateamplitude_exofast(self):
        A = np.median(self.flat_samples[:,0])
        B = np.median(self.flat_samples[:,1])
        err_A = np.percentile(self.flat_samples[:,0], [16,84])
        err_B = np.percentile(self.flat_samples[:,1], [16,84])
        k = np.sqrt(A**2. + B**2.)
        err = np.sqrt( (A*err_A/k)**2. + (B*err_B/k)**2.)
        print("K = " + str(k))
        print(err)

    def printperiod_exofast(self):
        b = np.percentile(self.flat_samples[:,3], (16,50,84))
        b_err = np.array([b[0], b[2]])
        b = b[1]
        b_err -= b
        print('The period in days is ' + str(b) + " +" + str(b_err[1]) + ' ' + str(b_err[0]))

    def returnfinalvals(self):
        values = []
        for i in range(self.ndim):
            val = np.percentile(self.flat_samples[:,i], (16,50,84))
            val[0] = val[0] - val[1]
            val[2] = val[2] - val[1]
            values.append(val)
        return np.array(values)

    def calculateperiod(self):
        b = np.percentile(self.flat_samples[:,1], (16,50,84))
        b_err = np.array([b[0], b[2]])
        b = b[1]
        b_err -= b
        print('The period in days is ' + str(b) + " +" + str(b_err[1]) + ' ' + str(b_err[0]))
        self.period = b*24*3600
        self.period_err = b_err*24*3600

    def calculateorbrad(self, starmass, starmass_err):
        k = 6.674e-11/(4*np.pi**2.)
        k **= (1/3)
        self.orbitalradius = k*(starmass*self.period**2.)**(1/3)
        self.orbrad_err = k/3*np.sqrt(((starmass_err**2.)*(self.period/starmass)**(4/3)) + (4*(self.period_err**2.)*(starmass/self.period)**(2/3)))
        o = self.orbitalradius/149597870700
        o_err = self.orbrad_err/149597870700
        print('Orbital radius in AU is ' + str(o) + ' +' + str(o_err[1]) + ' -' + str(o_err[0]))
        orbrad = np.array([self.orbrad_err[0],self.orbitalradius,self.orbrad_err[1]])
        return orbrad

    def setperiod(self, period, period_err):
        self.period = period*24*3600
        self.period_err = period_err*24*3600

    def plotfromreadin(self):
        values = np.loadtxt('cuillin/radveloutput.txt')
        a = values[0,1]
        b = values[1,1]
        c = values[2,1]
        d = values[3,1]
        factor = 2*np.pi/b
        xplot = np.linspace(-.5,2.5)
        yplot = a*np.sin(factor*(xplot + c)) + d
        model = a*np.sin(factor*(self.x + c)) + d
        chisq = np.sum(((self.y-model)**2.)/self.err)
        print(chisq)
        print(chi2.sf(chisq, 7))

        fig = plt.figure()
        ax1 = fig.add_axes((.1,.3,.8,.6))
        ax2 = fig.add_axes((.1,.1,.8,.2))
        ax2.set_xlabel(r"\textrm{Time (days)}")
        ax1.set_ylabel(r"$\textrm{Velocity (ms}^{-1})$")
        ax2.set_ylabel(r"$\textrm{Residuals}$")
        ax1.get_xaxis().set_ticks([])
        ax1.set_xlim(-.5,2.5)
        ax2.set_xlim(-.5,2.5)
        ax1.errorbar(self.x,self.y,yerr=self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax2.errorbar(self.x, (model-self.y), yerr = self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax1.plot(xplot, yplot, color='black')
        ax2.plot(xplot, np.zeros_like(xplot), ':',color='black')
        plt.savefig('cuillin/radvel_curve.png')

        print('Saved!')

        reader = emcee.backends.HDFBackend('cuillin/cuillin_sampler.h5')
        tau = reader.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        sampler = reader.get_chain(discard = burnin, flat=True, thin=thin)

        labels = [r'$K$', r'$P$', r'$C$', r'$\gamma$']
        fig = corner.corner(sampler, labels=labels, quantiles = [0.16, .5, .84], show_titles = True, use_math_text = True, smooth = True, title_kwargs={"fontsize": 28}, label_kwargs={"fontsize": 28}, smooth1d = True)
        plt.savefig('cuillin/radvel_corner.png')

def main():
    x = np.loadtxt('tres2b_rv.dat',comments='#', usecols=0)
    y =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=1)
    err =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=2)
    x -= x[0]
    print(x)

    minimiser = radvelminimiser_MCMC(x,y,err)
    iter = 1500000
    minimiser.run(iter)
    cutoff = int(.1*iter)
    minimiser.getsamples(cutoff)
    minimiser.printvals()
    minimiser.plotcurve()
    minimiser.plotcorner()
    #minimiser.plotwalkers()
    values = minimiser.returnfinalvals()
    print(values)
    minimiser.calculateperiod()

    #using James 8/2/20 vals
    starmass = 0.9 * 1.98847e30
    starmass_err = 0.2 * 1.98847e30*np.array([1,1])

    orbrad = minimiser.calculateorbrad(starmass, starmass_err)
    orbrad = np.array([orbrad])
    print(orbrad)
    values = np.concatenate((values, orbrad))
    print(values)
    np.savetxt('radveloutput.txt', values)


#main()

def printchisq():
    x = np.loadtxt('tres2b_rv.dat',comments='#', usecols=0)
    y =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=1)
    err =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=2)
    x -= x[0]
    minimiser = radvelminimiser_MCMC(x,y,err)
    minimiser.plotfromreadin()

printchisq()

def exofastmain():
    x = np.loadtxt('tres2b_rv.dat',comments='#', usecols=0)
    y =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=1)
    err =  np.loadtxt('tres2b_rv.dat',comments='#', usecols=2)
    #x -= x[0]
    print(x)
    minimiser = radvelminimiser_MCMC(x,y,err)
    iter = 100000
    minimiser.run_exofast(iter)
    minimiser.getsamples(int(.1*iter))
    minimiser.printvals()
    minimiser.plotcurve_exofast()
    minimiser.plotcorner()
    minimiser.plotwalkers()
    minimiser.calculateamplitude_exofast()

#exofastmain()

def rerunmain():
    #using James 8/2/20 vals
    starmass = 0.9 * 1.98847e30
    starmass_err = 0.2 * 1.98847e30*np.array([1,1])

    values = np.loadtxt('radveloutput.txt')
    period = values[1,1]
    period_err = np.array([values[1,0], values[1,2]])
    print(period)
    print(period_err)
    #input()
    filename = 'tres2b_rv.dat'
    x = np.loadtxt(filename,comments='#', usecols=0)
    y =  np.loadtxt(filename,comments='#', usecols=1)
    err =  np.loadtxt(filename,comments='#', usecols=2)
    minimiser = radvelminimiser_MCMC(x,y,err)
    minimiser.setperiod(period, period_err)
    orbrad = minimiser.calculateorbrad(starmass, starmass_err)
    values[3,:] = orbrad
    print(values)
    np.savetxt('radveloutput.txt', values)

#rerunmain()
