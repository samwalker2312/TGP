import numpy as np
from pprint import pprint
from iminuit import Minuit as minuit
import matplotlib.pyplot as plt

time = np.loadtxt('radvel.txt',comments='#', usecols=0)
vel =  np.loadtxt('radvel.txt',comments='#', usecols=1)
err =  np.loadtxt('radvel.txt',comments='#', usecols=2)

class minimiser(object):
    def __init__(self, time, vel, err):
        self.time = time - time[0]
        self.vel = vel
        self.err = err

    def chisq(self, a, b, c):
        self.y = a*np.sin(b*self.time + c)
        chisq = np.sum(((self.y - self.vel)/self.err)**2.)
        return chisq

    def minchisq(self):
        self.m = minuit(self.chisq, a = 100, b = 100, c = 10, error_a = .01,\
                        error_b = .00001, error_c = 1, errordef = 1)
        fmin, param = self.m.migrad()
        self.mina = self.m.values[0]
        self.minb = self.m.values[1]
        self.minc = self.m.values[2]

    def plot(self):
        plt.figure()
        ax1 = plt.subplot(211)
        plt.errorbar(self.time, self.vel, yerr = self.err, fmt = 'o', \
                    color = 'black', capsize=5)
        x = np.linspace(self.time[0], self.time[-1], 500)
        ycalc = self.mina*np.sin(self.minb*x + self.minc)
        plt.plot(x, ycalc, color='black')
        ax2 = plt.subplot(212)
        ycalc = self.mina*np.sin(self.minb*self.time + self.minc)
        plt.errorbar(self.time, self.vel-ycalc, yerr = self.err, fmt = 'o',\
                     color = 'black', capsize=5)
        plt.axhline(0,color='black')
        plt.show()

def main():
    minim = minimiser(time, vel, err)
    minim.minchisq()
    minim.plot()

main()
