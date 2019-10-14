from iminuit import Minuit as minuit
import numpy as np
import matplotlib.pyplot as plt
import batman

def main():
    x = np.loadtxt('lightcurvedata.txt',comments='#', usecols=0)
    starttime = -x[0]
    x += starttime
    y =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=1)
    err =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=2)
    minim = minimiser(x,y,err,starttime)
    minim.minchisq()
    minim.plot()

class minimiser(object):

    def __init__(self, x, y, err, starttime):
        self.x = x
        self.y = y
        self.err = err
        self.params = batman.TransitParams()
        self.params.t0 = starttime
        self.params.per = 3.52
        self.params.rp = 0.3
        self.params.a = 8.
        self.params.inc = 87.
        self.params.ecc = 0.
        self.params.w = 90.
        self.params.u = [0.1, 0.3]
        self.params.limb_dark = 'power2'
        self.m = batman.TransitModel(self.params, self.x)

    def chisq(self, rp, a, inc, ecc, w, u1, u2):
        self.params.rp = rp
        self.params.a = a
        self.params.inc = inc
        self.params.ecc = ecc
        self.params.w = w
        self.params.u = [u1, u2]
        model = self.m.light_curve(self.params)
        chisq = np.sum(((self.y-model)/self.err)**2.)
        return chisq

    def minchisq(self):
        self.minimise = minuit(self.chisq, rp = .3, a = 8., inc=87., ecc=0.,fix_ecc=True,w=90.,\
        u1=0.1, u2=0.3, error_rp = .0001, error_a = .01, error_inc = .01,\
        error_ecc = .01, error_w = .01, error_u1 = .0001, error_u2 = .0001, errordef=1., limit_u1 = (-1,1), limit_u2 = (-1,1))
        fmin, param = self.minimise.migrad()

    def plot(self):
        plotparams = batman.TransitParams()
        plotparams.t0 = self.params.t0
        plotparams.per = self.params.per
        plotparams.limb_dark = self.params.limb_dark
        plotx = np.linspace(self.x[0], self.x[-1], 500)
        plotparams.rp = self.minimise.values[0]
        plotparams.a = self.minimise.values[1]
        plotparams.inc = self.minimise.values[2]
        plotparams.ecc = self.minimise.values[3]
        plotparams.w = self.minimise.values[4]
        plotparams.u = [self.minimise.values[5], self.minimise.values[6]]
        plotmodel = batman.TransitModel(plotparams, plotx)
        flux = plotmodel.light_curve(plotparams)
        plt.figure()
        plt.errorbar(self.x, self.y, yerr=self.err, fmt='o', color='black')
        plt.plot(plotx, flux)
        plt.show()

main()
