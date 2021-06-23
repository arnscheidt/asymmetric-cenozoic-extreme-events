#!/usr/bin/env python3

# This file produces Figures S1 and S2 of Arnscheidt and Rothman - Asymmetry of extreme Cenozoic climate-carbon cycle events (2021)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# IMPORT DATA FROM FILE 
# these data were generated using the online interface provided by Laskar et al. (2004)
la04_lat0 = np.loadtxt('la04_lat0.txt',dtype=float,delimiter=',')
la04_lat45 = np.loadtxt('la04_lat45.txt',dtype=float,delimiter=',')
la04_mean = np.loadtxt('la04_mean.txt',dtype=float,delimiter=',')

# Figure S1
plt.figure()
plt.subplot(121)
plt.title('La04, equator, 100 Ma - present')
histcolor = (0.8,0.4,0.4)
gaussiancolor = (0,0,0)
gaussianwidth = 3
x = np.linspace(300,500,200)
binwidth = 5

lat0_hist,lat0_bins = np.histogram(la04_lat0[:,1],bins=np.arange(380,500,binwidth))
lat0_hist = lat0_hist/np.sum(lat0_hist) 
plt.bar(lat0_bins[:-1],lat0_hist,width=binwidth,align='edge',edgecolor=(0,0,0),color=histcolor)

plt.xlabel(r'Insolation (W/m$^2$)')
plt.ylabel('Probability (N='+str(len(la04_lat0[:,1]))+')')

flucmean = np.nanmean(la04_lat0[:,1])
flucstd = np.sqrt(np.nanmean((la04_lat0[:,1]-flucmean)**2))
flucskew = np.nanmean(((la04_lat0[:,1]-flucmean)/flucstd)**3)
fluckurt = np.nanmean(((la04_lat0[:,1]-flucmean)/flucstd)**4)-3
plt.text(400,0.08,'S = '+format(flucskew,".2f")+'\n K = '+format(fluckurt,".2f"),horizontalalignment='center')

plt.subplot(122)
plt.title(r'La04, 45$^{\circ}$ N, 100 Ma - present')

lat45_hist,lat45_bins = np.histogram(la04_lat45[:,1],bins=np.arange(270,350,binwidth))
lat45_hist = lat45_hist/np.sum(lat45_hist)
plt.bar(lat45_bins[:-1],lat45_hist,width=binwidth,align='edge',edgecolor=(0,0,0),color=histcolor)

plt.xlabel(r'Insolation (W/m$^2$)')
plt.ylabel('Probability (N='+str(len(la04_lat45[:,1]))+')')

flucmean = np.nanmean(la04_lat45[:,1])
flucstd = np.sqrt(np.nanmean((la04_lat45[:,1]-flucmean)**2))
flucskew = np.nanmean(((la04_lat45[:,1]-flucmean)/flucstd)**3)
fluckurt = np.nanmean(((la04_lat45[:,1]-flucmean)/flucstd)**4)-3
plt.text(280,0.10,'S = '+format(flucskew,".2f")+'\n K = '+format(fluckurt,".2f"),horizontalalignment='center')

# Figure S2
fig = plt.figure()
ax = plt.subplot(111)
binwidth = 0.05
plt.title(r'La04, mean annual insolation, 100 Ma - present')
mean_hist,mean_bins = np.histogram(la04_mean[:,1],bins=np.arange(342,342.8,binwidth))
mean_hist = mean_hist/np.sum(mean_hist)
plt.bar(mean_bins[:-1],mean_hist,width=binwidth,align='edge',edgecolor=(0,0,0),color=histcolor)

flucmean = np.nanmean(la04_mean[:,1])
flucstd = np.sqrt(np.nanmean((la04_mean[:,1]-flucmean)**2))
flucskew = np.nanmean(((la04_mean[:,1]-flucmean)/flucstd)**3)
fluckurt = np.nanmean(((la04_mean[:,1]-flucmean)/flucstd)**4)-3

plt.xlabel(r'Insolation (W/m$^2$)')
plt.ylabel('Probability (N='+str(len(la04_mean[:,1]))+')')
plt.text(342.6,0.04,'S = '+format(flucskew,".2f")+'\n K = '+format(fluckurt,".2f"),horizontalalignment='center')

ax2 = fig.add_axes([0.55, 0.52, 0.3, 0.3])
plt.plot(la04_mean[:,0],la04_mean[:,1])
plt.xlabel('Time (ka)')
plt.ylabel(r'Insolation (W/m$^2$)')
plt.xlim(-2000,0)

plt.show()
