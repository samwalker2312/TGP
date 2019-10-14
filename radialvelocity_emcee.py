import emcee
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('radvel.txt',comments='#', usecols=0)
y =  np.loadtxt('radvel.txt',comments='#', usecols=1)
err =  np.loadtxt('radvel.txt',comments='#', usecols=2)

def loglike(theta, x, y, err):
    a, b, c, d, log_f = theta
    model = a*np.sin(b*x+c)
    sigma2 = err**2. + model**2.*np.exp(2*log_f)
    return -.5*np.sum((y-model)**2./sigma2 + np.log(sigma2))

def logprior(theta):
    a, b, c, d, log_f = theta
    if -200 < a < 200 and 0. < b < 5000. and 0 < c < 1000 and -1000 < d < 1000 and -10. < log_f < 1.:
        return 0.0
    return -np.inf

def logprob(theta, x, y, err):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta,x,y,err)

pos = [[0,100,0,0,0]] + [[.1,1,1,.1,.01]]*np.random.randn(50,5)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args = (x,y,err))
sampler.run_mcmc(pos, 100000, progress=True)

flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)
for i in range(ndim-1):
    print(np.median(flat_samples[:,i]))

xplot = np.linspace(x[0],x[-1],500)
yplot = np.median(flat_samples[:,0])*np.sin(np.median(flat_samples[:,1])*xplot + np.median(flat_samples[:,2]))
plt.figure()
plt.errorbar(x,y,yerr=err, fmt='o')
plt.plot(xplot, yplot)
plt.show()
