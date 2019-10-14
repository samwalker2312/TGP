import numpy as np
import emcee
import matplotlib.pyplot as plt
import batman
from lightcurve_minuit import minimiser
import corner

x = np.loadtxt('lightcurvedata.txt',comments='#', usecols=0)
starttime = -x[0]
x += starttime
y =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=1)
err =  np.loadtxt('lightcurvedata.txt',comments='#', usecols=2)

minim = minimiser(x,y,err,starttime)
minim.minchisq()
paramlist = minim.minimise.values

params = batman.TransitParams()
params.t0 = minim.params.t0
params.per = minim.params.per
params.rp = paramlist[0]
params.a = paramlist[1]
params.inc = paramlist[2]
params.ecc = paramlist[3]
params.w = paramlist[4]
params.u = [paramlist[5], paramlist[6]]
params.limb_dark = minim.params.limb_dark
m = batman.TransitModel(params, x)

def loglike(theta, x, y, err):
    rp, a, inc, ecc, w, u1, u2 = theta
    if u1 + u2 > 1.:
        return -np.inf
    else:
        params.rp = rp
        params.a=a
        params.inc = inc
        params.ecc = ecc
        params.w = w
        params.u = [u1,u2]
        model = m.light_curve(params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

def logprior(theta):
    rp, a, inc, ecc, w, u1, u2= theta
    if 0<rp<1 and 1<a<100 and 0<inc<90 and 0<ecc<1 and 0<w<180 and -1<u1<1 and -1<u2<1:
        return 0.0
    return -np.inf

def logprob(theta, x, y, err):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta,x,y,err)

def loglike_fixecc(theta, x, y, err):
    rp, a, inc, u1, u2 = theta
    if u1 + u2 > 1.:
        return -np.inf
    else:
        params.rp = rp
        params.a=a
        params.inc = inc
        params.u = [u1,u2]
        model = m.light_curve(params)
        sigma2 = err**2.
        return -.5*np.sum((y-model)**2./sigma2)

def logprior_fixecc(theta):
    rp, a, inc, u1, u2= theta
    if 0<rp<1 and 1<a<100 and 0<inc<90 and -1<u1<1 and -1<u2<1:
        return 0.0
    return -np.inf

def logprob_fixecc(theta, x, y, err):
    lp = logprior_fixecc(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_fixecc(theta,x,y,err)

def initemcee_fitall():
    pos = [[paramlist[0],paramlist[1],paramlist[2],paramlist[3],paramlist[4],paramlist[5],paramlist[6]]] + 1e-3*np.random.randn(50,7)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args = (x,y,err))
    return sampler, pos, ndim
def initemcee_fixecc():
    pos = [[paramlist[0],paramlist[1],paramlist[2],paramlist[5],paramlist[6]]] + 1e-3*np.random.randn(50,5)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fixecc, args = (x,y,err))
    return sampler, pos, ndim
sampler, pos, ndim = initemcee_fixecc()

sampler.run_mcmc(pos, 10000, progress=True)
def plot_fitall():
    fig, axes = plt.subplots(7)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")

def plot_fixecc():
    fig, axes = plt.subplots(5)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")

plot_fixecc()

flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)

fig = corner.corner(flat_samples)

plt.figure()
params.rp = np.median(flat_samples[:,0])
print(params.rp)
params.a = np.median(flat_samples[:,1])
print(params.a)
params.inc = np.median(flat_samples[:,2])
print(params.inc)
if params.ecc == paramlist[3]:
    params.u = [np.median(flat_samples[:,3]),np.median(flat_samples[:,4])]
    print(params.u)
else:
    params.ecc = np.median(flat_samples[:,3])
    print(params.ecc)
    params.w = np.median(flat_samples[:,4])
    print(params.w)
    params.u = [np.median(flat_samples[:,5]),np.median(flat_samples[:,6])]
    print(params.u)
plt.errorbar(x,y,yerr=err, fmt='o')
plt.plot(x, m.light_curve(params))
plt.show()
