import numpy as np
import emcee
import matplotlib.pyplot as plt
import batman
import corner

class lightcurveminimiser_MCMC(object):

    def __init__(self, x, y, err):
        self.x = x
        self.y = y
        self.err = err

    def createmodel(self, t0, period, limbdark):
        self.params = batman.TransitParams()
        self.params.t0 = t0
        self.params.per = period
        self.params.limb_dark = limbdark
        self.params.rp = .1
        self.params.a = 15.
        self.params.inc = 90.
        self.params.ecc = 0.
        self.initialecc = self.params.ecc
        self.params.w = 90.
        if self.params.limb_dark == 'uniform':
            self.params.u = []
        elif self.params.limb_dark == 'linear':
            self.params.u = [.1]
        elif self.params.limb_dark == 'nonlinear':
            self.params.u = [.1,.1,.1,.1]
        else:
            self.params.u = [.1,.1]
        self.m = batman.TransitModel(self.params, self.x)

    def loglike(self,theta, x, y, err):
        if self.params.limb_dark == 'uniform':
            rp, a, inc, ecc, w = theta
        elif self.params.limb_dark == 'linear':
            rp, a, inc, ecc, w, u1 = theta
            self.params.u = [u1]
        elif self.params.limb_dark == 'nonlinear':
            rp, a, inc, ecc, w, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
        else:
            rp, a, inc, ecc, w, u1, u2 = theta
            self.params.u = [u1,u2]

        self.params.rp = rp
        self.params.a=a
        self.params.inc = inc
        self.params.ecc = ecc
        self.params.w = w
        model = self.m.light_curve(self.params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

    def logprior(self,theta):
        if self.params.limb_dark == 'uniform':
            rp, a, inc, ecc, w = theta
            if 0<rp<1 and 1<a<100 and 0<inc<90 and 0<ecc<1 and -180<w<180:
                return 0.0
        elif self.params.limb_dark == 'linear':
            rp, a, inc, ecc, w, u1 = theta
            self.params.u = [u1]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and 0<ecc<1 and -180<w<180 and -1<u1<1:
                return 0.0
        elif self.params.limb_dark == 'nonlinear':
            rp, a, inc, ecc, w, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and 0<ecc<1 and -180<w<180 and -1<u1<1 and -1<u2<1 and -1<u3<1 and -1<u4<1:
                return 0.0
        else:
            rp, a, inc, ecc, w, u1, u2 = theta
            self.params.u = [u1,u2]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and 0<ecc<1 and -180<w<180 and -1<u1<1 and -1<u2<1 and u1 + u2 < 1:
                return 0.0
        return -np.inf

    def logprob(self, theta, x, y, err):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta,x,y,err)

    def loglike_fixecc(self, theta, x, y, err):
        if self.params.limb_dark == 'uniform':
            rp, a, inc = theta
        elif self.params.limb_dark == 'linear':
            rp, a, inc,u1 = theta
            self.params.u = [u1]
        elif self.params.limb_dark == 'nonlinear':
            rp, a, inc,u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
        else:
            rp, a, inc, u1, u2 = theta
            self.params.u = [u1,u2]
        params.rp = rp
        params.a=a
        params.inc = inc
        model = m.light_curve(params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

    def logprior_fixecc(self, theta):
        if self.params.limb_dark == 'uniform':
            rp, a, inc = theta
            if 0<rp<1 and 1<a<100 and 0<inc<90:
                return 0.0
        elif self.params.limb_dark == 'linear':
            rp, a, inc, u1 = theta
            self.params.u = [u1]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and -1<u1<1:
                return 0.0
        elif self.params.limb_dark == 'nonlinear':
            rp, a, inc, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and -1<u1<1 and -1<u2<1 and -1<u3<1 and -1<u4<1:
                return 0.0
        else:
            rp, a, inc, u1, u2 = theta
            self.params.u = [u1,u2]
            if 0<rp<1 and 1<a<100 and 0<inc<90 and -1<u1<1 and -1<u2<1 and u1 + u2 < 1:
                return 0.0
        return -np.inf

    def logprob_fixecc(self, theta):
        lp = self.logprior_fixecc(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike_fixecc(theta,self.x,self.y,self.err)

    def initemcee_fitall(self):
        #pos = [[paramlist[0],paramlist[1],paramlist[2],paramlist[3],paramlist[4],paramlist[5],paramlist[6]]] + 1e-3*np.random.randn(50,7)
        if self.params.limb_dark == 'uniform':
            self.pos = [[.1, 15., 90., 0., 90.]] + 1e-3*np.random.randn(50,5)
        elif self.params.limb_dark == 'linear':
            self.pos = [[.1, 15., 90., 0., 90., .1]] + 1e-3*np.random.randn(50,6)
        elif self.params.limb_dark == 'nonlinear':
            self.pos = [[.1, 15., 90., 0., 90., .1, .1, .1, .1]] + 1e-3*np.random.randn(50,9)
        else:
            self.pos = [[.1, 15., 90., 0., 90., .1, .1]] + 1e-3*np.random.randn(50,7)
        self.nwalkers, self.ndim = self.pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))

    def initemcee_fixecc(self):
        if self.params.limb_dark == 'uniform':
            self.pos = [[.1, 15., 90.]] + 1e-3*np.random.randn(50,3)
        elif self.params.limb_dark == 'linear':
            self.pos = [[.1, 15., 90., .1]] + 1e-3*np.random.randn(50,4)
        elif self.params.limb_dark == 'nonlinear':
            self.pos = [[.1, 15., 90., .1, .1, .1, .1]] + 1e-3*np.random.randn(50,7)
        else:
            self.pos = [[.1, 15., 90., .1, .1]] + 1e-3*np.random.randn(50,5)
        self.nwalkers, self.ndim = self.pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob_fixecc, args = (self.x,self.y,self.err))

    def run(self):
        self.sampler.run_mcmc(self.pos, 10000, progress = True)

    def plotwalkers(self):
        fig, axes = plt.subplots(self.ndim)
        samples = self.sampler.get_chain()
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
        plt.show()

    def flattensamples(self):
        self.flat_samples = self.sampler.get_chain(discard=2000, thin=15, flat=True)

    def plotcorner(self):
        fig = corner.corner(self.flat_samples)
        plt.show()

    def plotcurve(self):
        plt.figure()
        plt.errorbar(self.x,self.y,yerr=self.err, fmt='o')
        plt.plot(self.x, self.m.light_curve(self.params))
        plt.show()

    def setfinalparams(self):
        self.params.rp = np.median(self.flat_samples[:,0])
        print(self.params.rp)
        self.params.a = np.median(self.flat_samples[:,1])
        print(self.params.a)
        self.params.inc = np.median(self.flat_samples[:,2])
        print(self.params.inc)
        if self.params.ecc == self.initialecc:
            if self.params.limb_dark == 'uniform':
                self.params.u = []
            elif self.params.limb_dark == 'linear':
                self.params.u = [np.median(self.flat_samples[:,3])]
            elif self.params.limb_dark == 'nonlinear':
                self.params.u = [np.median(self.flat_samples[:,3]),np.median(self.flat_samples[:,4]),np.median(self.flat_samples[:,5]),np.median(self.flat_samples[:,6])]
            else:
                self.params.u = [np.median(self.flat_samples[:,3]),np.median(self.flat_samples[:,4])]
        else:
            self.params.ecc = np.median(self.flat_samples[:,3])
            print(self.params.ecc)
            self.params.w = np.median(self.flat_samples[:,4])
            print(self.params.w)
            if self.params.limb_dark == 'uniform':
                self.params.u = []
            elif self.params.limb_dark == 'linear':
                self.params.u = [np.median(self.flat_samples[:,5])]
            elif self.params.limb_dark == 'nonlinear':
                self.params.u = [np.median(self.flat_samples[:,5]),np.median(self.flat_samples[:,6]),np.median(self.flat_samples[:,7]),np.median(self.flat_samples[:,8])]
            else:
                self.params.u = [np.median(self.flat_samples[:,5]),np.median(self.flat_samples[:,6])]
        print(self.params.u)

x = np.loadtxt('lightcurvedata.txt',comments='#', usecols=0)
starttime = -x[0]
x += starttime
y =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=1)
err =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=2)
valuearray = np.zeros((8,5))
for i in range(0,8):
    model = lightcurveminimiser_MCMC(x, y, err)
    if i == 0:
        model.createmodel(starttime, 3.52, 'uniform')
    if i == 1:
        model.createmodel(starttime, 3.52, 'linear')
    if i == 2:
        model.createmodel(starttime, 3.52, 'nonlinear')
    if i == 3:
        model.createmodel(starttime, 3.52, 'quadratic')
    if i == 4:
        model.createmodel(starttime, 3.52, 'power2')
    if i == 5:
        model.createmodel(starttime, 3.52, 'exponential')
    if i == 6:
        model.createmodel(starttime, 3.52, 'logarithmic')
    if i == 7:
        model.createmodel(starttime, 3.52, 'squareroot')
    model.initemcee_fitall()
    model.run()
    model.flattensamples()
    model.setfinalparams()
    if i == 0:
        model.plotcurve()
        model.plotcorner()
        model.plotwalkers()
    print("\n-----------------------------------------------------\n")
    valuearray[i,0] = model.params.rp
    valuearray[i,1] = model.params.a
    valuearray[i,2] = model.params.inc
    valuearray[i,3] = model.params.ecc
    valuearray[i,4] = model.params.w
print(valuearray)
print("\n-----------------------------------------------------\n")
newarray = valuearray
newarray[:] /= newarray[3,:]
print(newarray)
