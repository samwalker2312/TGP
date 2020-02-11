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
        self.i = 0

    def createmodel(self, period, limbdark, a):
        self.params = batman.TransitParams()
        self.params.t0 = 0
        self.params.per = period
        self.params.limb_dark = limbdark
        self.params.rp = .1
        self.params.a = a
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
            t0, rp, inc, ecc, w = theta
        elif self.params.limb_dark == 'linear':
            t0, rp, inc, ecc, w, u1 = theta
            self.params.u = [u1]
        elif self.params.limb_dark == 'nonlinear':
            t0, rp, inc, ecc, w, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
        else:
            t0, rp, inc, ecc, w, u1, u2 = theta
            self.params.u = [u1,u2]

        self.params.t0 = t0
        self.params.rp = rp
        self.params.inc = inc
        self.params.ecc = ecc
        self.params.w = w
        model = self.m.light_curve(self.params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

    def logprior(self,theta):
        if self.params.limb_dark == 'uniform':
            t0, rp, inc, ecc, w = theta
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and 0.<=ecc<.95 and 0<w<360:
                return 0.0

        #edited for t0 == 0
        elif self.params.limb_dark == 'linear':
            t0, rp, inc, ecc, w, u1 = theta
            self.params.u = [u1]
            #step = self.x[1]-self.x[0]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and 0.<=ecc<.95 and 0<w<360 and -1<u1<1:
                return 0.

        elif self.params.limb_dark == 'nonlinear':
            t0, rp, inc, ecc, w, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and 0.<=ecc<.95 and 0<w<360 and -1<u1<1 and -1<u2<1 and -1<u3<1 and -1<u4<1:
                return 0.0
        else:
            t0, rp, inc, ecc, w, u1, u2 = theta
            self.params.u = [u1,u2]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and 0.<=ecc<.95 and 0<w<360 and -1<u1<1 and -1<u2<1 and u1 + u2 < 1:
                return 0.0
        return -np.inf

    def logprob(self, theta, x, y, err):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta,x,y,err)

    def loglike_fixecc(self, theta, x, y, err):
        if self.params.limb_dark == 'uniform':
            t0, rp, inc = theta
        elif self.params.limb_dark == 'linear':
            t0, rp, inc,u1 = theta
            self.params.u = [u1]
        elif self.params.limb_dark == 'nonlinear':
            t0, rp, inc,u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
        else:
            t0, rp, inc, u1, u2 = theta
            self.params.u = [u1,u2]
        self.params.t0 = t0
        self.params.rp = rp
        self.params.inc = inc
        model = self.m.light_curve(self.params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

    def logprior_fixecc(self, theta):
        if self.params.limb_dark == 'uniform':
            t0, rp, inc = theta
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90:
                return 0.0
        elif self.params.limb_dark == 'linear':
            t0, rp, inc, u1 = theta
            self.params.u = [u1]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and -1<u1<1:
                return 0.0
        elif self.params.limb_dark == 'nonlinear':
            t0, rp, inc, u1, u2, u3, u4 = theta
            self.params.u = [u1, u2, u3, u4]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and -1<u1<1 and -1<u2<1 and -1<u3<1 and -1<u4<1:
                return 0.0
        else:
            t0, rp, inc, u1, u2 = theta
            self.params.u = [u1,u2]
            if self.x[0]<t0<self.x[-1] and 0<rp<1 and 0<inc<90 and -1<u1<1 and -1<u2<1 and u1 + u2 < 1:
                return 0.0
        return -np.inf

    def logprob_fixecc(self, theta, x, y, err):
        lp = self.logprior_fixecc(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike_fixecc(theta,x,y,err)

    def initemcee_fitall(self):

        ##changed t0_init for mirrored transit

        t0_init = 0
        if self.params.limb_dark == 'uniform':
            self.pos = [[t0_init, .1, 85., .01, 90.]] + 1e-3*np.random.randn(100,5)
        elif self.params.limb_dark == 'linear':
            self.pos = [[t0_init, .1, 85., .01, 90., .1]] + [[.00001,.01,.1,.0001,.1,.01]]*np.random.randn(100,6)
        elif self.params.limb_dark == 'nonlinear':
            self.pos = [[t0_init, .1, 85., .01, 90., .1, .1, .1, .1]] + 1e-3*np.random.randn(100,9)
        else:
            self.pos = [[t0_init, .1, 85., .01, 90., .1, .1]] + 1e-3*np.random.randn(100,7)
        self.nwalkers, self.ndim = self.pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, args = (self.x,self.y,self.err))

    def initemcee_fixecc(self):
        t0_init = .5*(self.x[-1] - self.x[0])
        if self.params.limb_dark == 'uniform':
            self.pos = [[t0_init, .1, 90.]] + 1e-3*np.random.randn(50,3)
        elif self.params.limb_dark == 'linear':
            self.pos = [[t0_init, .1, 90., .1]] + 1e-3*np.random.randn(50,4)
        elif self.params.limb_dark == 'nonlinear':
            self.pos = [[t0_init, .1, 90., .1, .1, .1, .1]] + 1e-3*np.random.randn(50,7)
        else:
            self.pos = [[t0_init, .1, 90., .1, .1]] + 1e-3*np.random.randn(50,5)
        self.nwalkers, self.ndim = self.pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob_fixecc, args = (self.x,self.y,self.err))

    def run(self, iter):
        self.sampler.run_mcmc(self.pos, iter, progress = True)

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

    def flattensamples(self, discardnumb):
        self.flat_samples = self.sampler.get_chain(discard=discardnumb, thin=15, flat=True)

    def plotcorner(self):
        labels = [r'$t_0$', r'$\textrm{R}_{\textrm{p}}$', r'$i$']
        if self.params.ecc == self.initialecc:
            pass
        else:
            extralabels = [r'$e$', r'$\omega$']
            labels.extend(extralabels)
        if self.params.limb_dark == 'uniform':
            pass
        elif self.params.limb_dark == 'linear':
            labels.append(r'$\textrm{c}_1$')
        elif self.params.limb_dark == 'nonlinear':
            extralabels = [r'$\textrm{c}_1$',r'$\textrm{c}_2$',r'$\textrm{c}_3$',r'$\textrm{c}_4$']
            labels.extend(extralabels)
        else:
            extralabels = [r'$\textrm{c}_1$',r'$\textrm{c}_2$']
            labels.extend(extralabels)

        fig = corner.corner(self.flat_samples, labels=labels, quantiles = [0.16, .5, .84], show_titles = True, use_math_text = True, smooth = True, title_kwargs={"fontsize": 20}, smooth1d = True, label_kwargs={'fontsize':20})
        plt.savefig('lightcurve_corner.eps')

    def plotcurve(self):
        plotx = np.linspace(self.x[0], self.x[-1], 500)
        plotmodel = batman.TransitModel(self.params, plotx)
        model = batman.TransitModel(self.params, self.x)

        fig = plt.figure()
        ax1 = fig.add_axes((.1,.3,.8,.6))
        ax2 = fig.add_axes((.1,.1,.8,.2))
        ax2.set_xlabel(r'$\textrm{Days from}~t_0$')
        ax1.set_ylabel(r'$\textrm{Relative flux}$')
        ax2.set_ylabel(r"$\textrm{Residuals}$")
        ax1.get_xaxis().set_ticks([])
        ax1.errorbar(self.x,self.y,yerr=self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax2.errorbar(self.x, (model.light_curve(self.params)-self.y), yerr = self.err, fmt='o', mfc='black', mec='black', ecolor='black')
        ax1.plot(plotx, plotmodel.light_curve(self.params), color='black')
        ax2.plot(plotx, np.zeros_like(plotmodel.light_curve(self.params)), ':',color='black')
        plt.savefig('lightcurve.eps')

    def setfinalparams(self):
        self.params.t0 = np.median(self.flat_samples[:,0])
        print(self.params.t0)
        self.params.rp = np.median(self.flat_samples[:,1])
        print(self.params.rp)
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

    def returninc(self):
        inc = np.percentile(self.flat_samples[:,2], (16,50,84))
        inc_err = np.array([inc[0], inc[2]])
        inc = inc[1]
        inc_err -= inc
        return inc, inc_err

    def returnrp(self):
        rp = np.percentile(self.flat_samples[:,1], (16,50,84))
        rp_err = np.array([rp[0], rp[2]])
        rp = rp[1]
        rp_err -= rp
        return rp, rp_err

    def returnvalues(self):
        values = []
        for i in range(self.ndim):
            val = np.percentile(self.flat_samples[:,i], (16, 50, 84))
            val[0] = val[0] - val[1]
            val[2] = val[2] - val[1]
            values.append(val)
        return np.array(values)

def main():
    readfile = 'LightCurveFinal.txt'
    lightcurve_x = np.loadtxt(readfile,comments='#', usecols=0)
    #starttime = -lightcurve_x[0]
    #lightcurve_x += starttime
    lightcurve_y =  np.loadtxt(readfile,comments='#', usecols=1)
    lightcurve_err =  np.loadtxt(readfile,comments='#', usecols=2)

    data_x = lightcurve_x[34:] - lightcurve_x[33]
    data_y = lightcurve_y[34:]
    data_err = lightcurve_err[34:]

    start_x = -data_x[::-1]
    start_y = data_y[::-1]
    start_err = data_err[::-1]

    final_x = np.concatenate((start_x, data_x))/(24*3600)
    final_y = np.concatenate((start_y, data_y))
    final_err = np.concatenate((start_err, data_err))

    #using James values from 8/2/20
    starrad = 1.06*695700e3
    starrad_err = .15*695700e3*np.array([1,1])

    radveldata = np.loadtxt('radveloutput.txt')
    period = radveldata[1,1]
    orbrad = radveldata[-1,1]/starrad
    limbdark = 'linear'
    lightcurveminimiser = lightcurveminimiser_MCMC(final_x, final_y, final_err)

    lightcurveminimiser.createmodel(period, limbdark, orbrad)
    #lightcurveminimiser.initemcee_fixecc()
    lightcurveminimiser.initemcee_fitall()
    iter = 10000
    lightcurveminimiser.run(iter)
    lightcurveminimiser.flattensamples(int(.3*iter))
    lightcurveminimiser.setfinalparams()

    lightcurveminimiser.plotcurve()
    lightcurveminimiser.plotwalkers()
    lightcurveminimiser.plotcorner()

    values = lightcurveminimiser.returnvalues()
    #print(radveldata)
    print(values)
    finaldata = np.concatenate((radveldata, values))
    #print(finaldata)
    filename = 'finaldata_' + str(lightcurveminimiser.params.limb_dark) + '.txt'
    np.savetxt(filename, finaldata)

main()
