#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

def get_chi_squared(mu,sigma,data,samples,N_bins):
    '''
    In order to use the in-build chisquare function, we need 2 inputs.
    ------
    f_obs: Observed frequencies, this comes from our histogram.
    f_exp: Expected frequencies, this comes from our gaussian distribution.
    ------
    '''
    ### computing f_exp for our samples ###
    f_exp = stats.norm.pdf(samples, mu, sigma)

    ### formatting frequencies and bins to use in loop ###
    freq, bins = np.histogram(data,bins=N_bins,density=1)
    freq = [0] + list(freq) + [0]
    bins = list(bins) + [samples.max()]

    ### computing f_obs for our samples ###
    f_obs = []
    for i, b in enumerate(bins):
        cond = (samples > b)
        samples = samples[np.where(cond)]
        f_obs += [freq[i]]*sum(~cond)

    ### computing chi-square value using f_obs and f_exp ###
    chi, p = stats.chisquare(f_obs, f_exp)

    return chi

N_bins = 5
N_samples = 50

### TEST 1: STRAIGHT ###

data = pd.read_excel('Results.xlsx',sheetname=3,stylesheet="abc").as_matrix()
x = data[2:,0]
y = data[2:,1]
theta = data[2:,4]
data_forward = np.vstack(((x,y),theta))
data = np.sort(data_forward, axis=1)
mu = np.mean(data,axis=1)

# Standard Deviation
s = np.var(data,axis=1)
sigma_x = np.sqrt(s[0])
sigma_y = np.sqrt(s[1])
sigma_theta = np.sqrt(s[2])
sigma = np.hstack(((sigma_x,sigma_y),sigma_theta))

# x
samples = np.linspace(mu[0]-3*sigma[0], mu[0]+3*sigma[0], N_samples)
chi = get_chi_squared(mu[0],sigma[0],data[0,:],samples,N_bins)
plt.title(('Straight Motion Test - X PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[0], sigma[0]))
plt.hist(data[0,:][np.newaxis].T,bins=N_bins, density = True)
plt.xlabel('X (mm)')
plt.grid()
plt.show()
# y
samples = np.linspace(mu[1]-3*sigma[1], mu[1]+3*sigma[1], N_samples)
chi = get_chi_squared(mu[1],sigma[1],data[1,:],samples,N_bins)
plt.title(('Straight Motion Test - Y PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[1], sigma[1]))
plt.hist(data[1,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Y (mm)')
plt.grid()
plt.show()
# theta
samples = np.linspace(mu[2]-3*sigma[2], mu[2]+3*sigma[2], N_samples)
chi = get_chi_squared(mu[2],sigma[2],data[2,:],samples,N_bins)
plt.title(('Straight Motion Test - Theta PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[2], sigma[2]))
plt.hist(data[2,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Theta (degrees)')
plt.grid()
plt.show()


### TEST 2: LEFT ###

data = pd.read_excel('Results.xlsx',sheetname=1,stylesheet="abc").as_matrix()
x = data[2:,0]
y = data[2:,1]
theta = data[2:,4]
data_forward = np.vstack(((x,y),theta))
data = np.sort(data_forward, axis=1)
mu = np.mean(data,axis=1)

# Standard Deviation
s = np.var(data,axis=1)
sigma_x = np.sqrt(s[0])
sigma_y = np.sqrt(s[1])
sigma_theta = np.sqrt(s[2])
sigma = np.hstack(((sigma_x,sigma_y),sigma_theta))

# x
samples = np.linspace(mu[0]-3*sigma[0], mu[0]+3*sigma[0], N_samples)
chi = get_chi_squared(mu[0],sigma[0],data[0,:],samples,N_bins)
plt.title(('Left Motion Test - X PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[0], sigma[0]))
plt.hist(data[0,:][np.newaxis].T,bins=N_bins, density = True)
plt.xlabel('X (mm)')
plt.grid()
plt.show()
# y
samples = np.linspace(mu[1]-3*sigma[1], mu[1]+3*sigma[1], N_samples)
chi = get_chi_squared(mu[1],sigma[1],data[1,:],samples,N_bins)
plt.title(('Left Motion Test - Y PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[1], sigma[1]))
plt.hist(data[1,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Y (mm)')
plt.grid()
plt.show()
# theta
samples = np.linspace(mu[2]-3*sigma[2], mu[2]+3*sigma[2], N_samples)
chi = get_chi_squared(mu[2],sigma[2],data[2,:],samples,N_bins)
plt.title(('Left Motion Test - Theta PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[2], sigma[2]))
plt.hist(data[2,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Theta (degrees)')
plt.grid()
plt.show()


### TEST 3: RIGHT ###

data = pd.read_excel('Results.xlsx',sheetname=0,stylesheet="abc").as_matrix()
x = data[2:,0]
y = data[2:,1]
theta = data[2:,4]
data_forward = np.vstack(((x,y),theta))
data = np.sort(data_forward, axis=1)
mu = np.mean(data,axis=1)

# Standard Deviation
s = np.var(data,axis=1)
sigma_x = np.sqrt(s[0])
sigma_y = np.sqrt(s[1])
sigma_theta = np.sqrt(s[2])
sigma = np.hstack(((sigma_x,sigma_y),sigma_theta))

# x
samples = np.linspace(mu[0]-3*sigma[0], mu[0]+3*sigma[0], N_samples)
chi = get_chi_squared(mu[0],sigma[0],data[0,:],samples,N_bins)
plt.title(('Right Motion Test - X PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[0], sigma[0]))
plt.hist(data[0,:][np.newaxis].T,bins=N_bins, density = True)
plt.xlabel('X (mm)')
plt.grid()
plt.show()
# y
samples = np.linspace(mu[1]-3*sigma[1], mu[1]+3*sigma[1], N_samples)
chi = get_chi_squared(mu[1],sigma[1],data[1,:],samples,N_bins)
plt.title(('Right Motion Test - Y PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[1], sigma[1]))
plt.hist(data[1,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Y (mm)')
plt.grid()
plt.show()
# theta
samples = np.linspace(mu[2]-3*sigma[2], mu[2]+3*sigma[2], N_samples)
chi = get_chi_squared(mu[2],sigma[2],data[2,:],samples,N_bins)
plt.title(('Right Motion Test - Theta PDF [chi-squared = {0:.3f}]').format(chi))
plt.plot(samples, stats.norm.pdf(samples, mu[2], sigma[2]))
plt.hist(data[2,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Theta (degrees)')
plt.grid()
plt.show()

### START POSITION ###

data = pd.read_excel('Results.xlsx',sheetname=2,stylesheet="abc").as_matrix()
x = data[2:,0]
y = data[2:,1]
theta = data[2:,4]
data_forward = np.vstack(((x,y),theta))
data = np.sort(data_forward, axis=1)
mu = np.mean(data,axis=1)

# Standard Deviation
s = np.var(data,axis=1)
sigma_x = np.sqrt(s[0])
sigma_y = np.sqrt(s[1])
sigma_theta = np.sqrt(s[2])
sigma = np.hstack(((sigma_x,sigma_y),sigma_theta))

# x
samples = np.linspace(mu[0]-3*sigma[0], mu[0]+3*sigma[0], N_samples)
plt.title('Start Position - X PDF')
plt.plot(samples, stats.norm.pdf(samples, mu[0], sigma[0]))
plt.hist(data[0,:][np.newaxis].T,bins=N_bins, density = True)
plt.xlabel('X (mm)')
plt.grid()
plt.show()
# y
samples = np.linspace(mu[1]-3*sigma[1], mu[1]+3*sigma[1], N_samples)
plt.title('Start Position - Y PDF')
plt.plot(samples, stats.norm.pdf(samples, mu[1], sigma[1]))
plt.hist(data[1,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Y (mm)')
plt.grid()
plt.show()
# theta
samples = np.linspace(mu[2]-3*sigma[2], mu[2]+3*sigma[2], N_samples)
plt.title('Start Position - Theta PDF ')
plt.plot(samples, stats.norm.pdf(samples, mu[2], sigma[2]))
plt.hist(data[2,:][np.newaxis].T,bins=N_bins, density = 1)
plt.xlabel('Theta (degrees)')
plt.grid()
plt.show()
