#!/usr/bin/env python3

# This file produces Figure 1, Figure 2, Table 1, Figure S3 of Arnscheidt and Rothman - Asymmetry of extreme Cenozoic climate-carbon cycle events (2021)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft
from scipy.special import gammainc

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# IMPORT CENOGRID DATA (Westerhold et al. 2020) 
# The following code allows you to import the data directly from Table S33 of Westerhold et al. (2020)
#rawdata = np.loadtxt('TableS33.tab',dtype=str,delimiter='\t',skiprows=92)
#cenogrid = rawdata[:,[4,14,15]]
#cenogrid[cenogrid == ''] = 'NaN'
#cenogrid = np.flip(cenogrid.astype(float),axis=0)
#cenogrid[:,0] = -cenogrid[:,0]
#np.save('cenogrid_data',cenogrid)

# Here, we provide the relevant data as a .npy file. cenogrid[:,0] is time, [:,1] is d13C, and [:,2] is d18O 
cenogrid = np.load('cenogrid_data.npy')

# DEFINE HELPER FUNCTIONS  

def detrend(data, interval):
    # return detrended data, moving average, and moving standard deviation
    # nontrivial because of non-constant time interval
    # the one-sided standard deviation takes into account positive fluctuations and is used
    # for the analysis in Table 1
    # data[:,1] is the data, data[:,0] is the time
   
    data_detrended = []

    data_values = data[:,1]
    data_time = data[:,0]

    for i in range(0,len(data[:,1])):
        window = [data[i,0]-interval/2,data[i,0]+interval/2]
        if window[0]>np.amin(data[:,0]) and window[1]<np.amax(data[:,0]):
            vals = data_values[(data_time>window[0])*(data_time<window[1])]
            mean = np.nanmean(vals)
            vals2 = vals[vals>mean]
            std = np.sqrt(np.nanmean((vals-mean)**2))
            std_one_sided = np.sqrt(np.nanmean((vals2-mean)**2))

            data_detrended.append([data_time[i],data_values[i]-mean,mean,std,std_one_sided])
    return np.array(data_detrended)

def skewkurt(x):
    # returns skewness and kurtosis of x
    mean = np.nanmean(x)
    sd = np.sqrt(np.nanmean((x-mean)**2))

    skew = np.nanmean(((x-mean)/sd)**3)
    kurt = np.nanmean(((x-mean)/sd)**4)-3

    return skew,kurt

def sk_bootstrap(x,n,val,conf):
    # returns bootstrap confidence intervals of conf, for x, using n resamplings

    sk = np.zeros((n,2))

    # generate bootstrap error distribution
    for i in range(0,n):
        sk[i,:] = skewkurt(np.random.choice(x,size=len(x),replace=True)) - val

    quantiles = [(1-conf)/2,1-(1-conf)/2]
    return np.abs([np.quantile(sk[:,0],quantiles[0]),np.quantile(sk[:,0],quantiles[1])]), np.abs([np.quantile(sk[:,1],quantiles[0]),np.quantile(sk[:,1],quantiles[1])])

def bin_mean(data,interval):
    # returns means of the data across bins of width interval

    data_values = data[:,1]
    data_time = data[:,0]

    data_binned = []

    nbins = int((np.amax(data_time)-np.amin(data_time))/interval)

    for i in range(0,nbins):
        window = [np.amin(data_time)+i*interval,np.amin(data_time)+(i+1)*interval]
        vals = data_values[(data_time>window[0])*(data_time<window[1])]
        data_binned.append([np.amin(data_time)+(i+0.5)*interval,np.nanmean(vals)])

    return np.array(data_binned)

def rma(x,y):
    # linear reduced major axis regression according to Rayner (1985)

    xmean = np.nanmean(x)
    ymean = np.nanmean(y)

    xvar = np.nanmean((x-xmean)**2)
    yvar = np.nanmean((y-ymean)**2)
    cov = np.nanmean((x-xmean)*(y-ymean))

    slope = np.sign(cov)*np.sqrt(yvar/xvar)

    # the line has to pass through (xmean, ymean)
    yintercept = ymean - xmean*slope

    return slope,yintercept

def corr(x,y):
    # calculates rank correlation coefficient

    # create rank arrays
    temp = x.argsort()
    xranks = np.empty_like(temp)
    xranks[temp] = np.arange(len(x))

    temp = y.argsort()
    yranks = np.empty_like(temp)
    yranks[temp] = np.arange(len(y))

    x = xranks
    y = yranks

    # calculate correlation coefficient
    xmean = np.mean(x)
    ymean = np.mean(y)
    xvar = np.mean((x-xmean)**2)
    yvar = np.mean((y-ymean)**2)
    cov = np.mean((x-xmean)*(y-ymean))

    return cov/np.sqrt(xvar*yvar)

def permutation(x,y,N,val):
    # permutation test
    # randomly re-orders the y-vector N times and 
    # returns the error bars for the given confidence interval

    corr_dist = np.zeros(N)
    y2 = np.copy(y)

    for i in range(0,N):
        np.random.shuffle(y2)
        corr_dist[i] = corr(x,y2)

    # we are trying to assess whether or not fluctuations increase with temperature
    p = np.sum(corr_dist<=val)/N
    return p


# END OF FUNCTION DEFINITIONS

# DETREND data with 1 Myr moving average

moving_average_interval = 1 # Myr

dc = detrend(cenogrid[:,[0,1]],moving_average_interval) # detrended d13C
do = detrend(cenogrid[:,[0,2]],moving_average_interval) # detrended d18O

division_names = ['Pal','Eo','Ol','Mio','Plio','Plei']
division_names_full = ['Paleocene','Eocene','Oligocene','Miocene','Pliocene','Pleistocene']
divisions = [-66,-56,-33.9,-23,-5.33,-2.58,-0.0118]

#############################################################
# Figure 1 (time series/histograms)
#############################################################

# select Eocene snapshot
dc_index = (dc[:,0]<-46)*(dc[:,0]>-54)
dc_lim = [-1.2,0.7]
dc_binwidth = 0.15

do_index = (do[:,0]<-46)*(do[:,0]>-54)
do_lim = [-1.1,0.6]
do_binwidth = 0.15

histcolor = (0.8,0.4,0.4)
gaussiancolor = (0,0,0)
gaussianwidth = 3
x = np.linspace(dc_lim[0],dc_lim[1],200)

fig = plt.figure()
ax_ts1 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)

plt.plot(dc[dc_index,0],dc[dc_index,1],color=(0,0,0))
plt.ylim(dc_lim[1],dc_lim[0])
plt.ylabel(r'$\delta^{13}$C (per mil)')
ax_ts1.set_xticks([])

ax_hist1 = plt.subplot2grid((4,4),(0,3),colspan=1,rowspan=2)
plt.ylim(dc_lim[1],dc_lim[0])
dc_hist,dc_bins = np.histogram(dc[dc_index,1],bins=np.arange(dc_lim[0],dc_lim[1],dc_binwidth))
dc_hist = dc_hist/np.sum(dc_hist)
plt.barh(dc_bins[:-1],dc_hist,height=dc_binwidth,align='edge',edgecolor=(0,0,0),color=histcolor)
ax_hist1.set_yticks([])
ax_hist1.set_xticks([])
plt.text(0.03,-0.7,r'\textbf{*}',fontsize=20)

plt.xlim(0,0.35)

ax_ts2 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=2)

plt.plot(do[do_index,0],do[do_index,1],color=(0,0,0))
plt.ylim(do_lim[1],do_lim[0])
plt.xlabel('Time (Ma)')
plt.ylabel(r'$\delta^{18}$O (per mil)')

ax_hist2 = plt.subplot2grid((4,4),(2,3),colspan=1,rowspan=2)
plt.ylim(do_lim[1],do_lim[0])
do_hist,do_bins = np.histogram(do[do_index,1],bins=np.arange(do_lim[0],do_lim[1],do_binwidth))
do_hist = do_hist/np.sum(do_hist)
plt.barh(do_bins[:-1],do_hist,height=do_binwidth,align='edge',edgecolor=(0,0,0),color=histcolor)
ax_hist2.set_yticks([])
plt.xlabel('Probability')
plt.xlim(0,0.35)
plt.text(0.03,-0.57,r'\textbf{*}',fontsize=20)

fig.text(0.01,0.95,r'$\textbf{A}$')
fig.text(0.01,0.52,r'$\textbf{B}$')
fig.text(0.77,0.95,r'$\textbf{C}$')
fig.text(0.77,0.52,r'$\textbf{D}$')

plt.subplots_adjust(top=0.945,bottom=0.146,left=0.086,right=0.987,hspace=0.473,wspace=0.182)

#############################################################
# Figure 2 (skewness/kurtosis)
#############################################################

division_names = ['Pal','Eo','Ol','Mio','Plio','Plei']
divisions = [-65,-56,-33.9,-23,-5.33,-2.59,0.0118]

# initialize means
dc_div_sk = np.zeros((len(division_names),2))
do_div_sk = np.zeros((len(division_names),2))

# initialize confidence intervals
dc_div_sk_conf = np.zeros((len(division_names),2,2))
do_div_sk_conf = np.zeros((len(division_names),2,2))

for idiv in range(0,len(division_names)):

    # inelegant removal of the PETM
    if idiv==1:
        divisions[idiv]=-55.5

    dc_temp = dc[(dc[:,0]>divisions[idiv])*(dc[:,0]<divisions[idiv+1]),1] 
    do_temp = do[(do[:,0]>divisions[idiv])*(do[:,0]<divisions[idiv+1]),1] 

    dc_div_sk[idiv,:] = skewkurt(dc_temp)
    do_div_sk[idiv,:] = skewkurt(do_temp)
    #print(divisions[idiv:idiv+2],skewkurt(dc_temp),skewkurt(do_temp),len(dc_temp),len(do_temp))
    dc_div_sk_conf[idiv,:,:] = sk_bootstrap(dc_temp,1000,dc_div_sk[idiv,:],0.95)
    do_div_sk_conf[idiv,:,:] = sk_bootstrap(do_temp,1000,do_div_sk[idiv,:],0.95)

# plot figure
fig = plt.figure()
plt.subplot2grid((15,1),(7,0),rowspan=8)
dc_col = (0,0,0.8)
do_col = (0.8,0,0)
axline_col = (0.7,0.7,0.7)

plt.axvline(0,color=axline_col,linestyle='--',zorder=-19)
plt.axhline(0,color=axline_col,linestyle='--',zorder=-19)

label_offset_general = [0.02,0.04,0.02,0.04]

label_size = 14
cap_size = 3
eb_linewidth = 0.7 
eb_color=(0.8,0.8,0.8)
cam_col = (0.9,0.9,0.9)
cam_label_col = (0.6,0.6,0.6)

eb_color=(0.65,0.65,0.65)

dc_col = (0,0,0.7)
do_col = (0.7,0,0)

ln_col = (0,0,0)

# points/errorbars
plt.errorbar(dc_div_sk[:,0],dc_div_sk[:,1],xerr=np.squeeze(dc_div_sk_conf[:,0,:]).T,yerr=np.squeeze(dc_div_sk_conf[:,1,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='o',ecolor=eb_color,color=(dc_col),label=r'$\delta^{13}$C')
plt.errorbar(do_div_sk[:,0],do_div_sk[:,1],xerr=np.squeeze(do_div_sk_conf[:,0,:]).T,yerr=np.squeeze(do_div_sk_conf[:,1,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='s',ecolor=eb_color,color=(do_col),label=r'$\delta^{18}$O')

# plot text
for idiv in range(0,len(division_names)):
    
    label_offset = label_offset_general
    # unique rule for some epochs to avoid the labels intersecting Lognormal
    if division_names[idiv] == 'Ol':
        label_offset = [-0.1,-0.21,0.02,0.03]
    if division_names[idiv] == 'Pal':
        label_offset = [-0.12,-0.21,0.02,0.03]
                
    plt.text(dc_div_sk[idiv,0]+label_offset[0],dc_div_sk[idiv,1]+label_offset[1],division_names[idiv],fontsize=label_size,color=dc_col,zorder=100)
    plt.text(do_div_sk[idiv,0]+label_offset[2],do_div_sk[idiv,1]+label_offset[3],division_names[idiv],fontsize=label_size,color=do_col,zorder=100)

plt.scatter(0,0,color=(0,0,0),label='Normal')

# calculate lognormal skewness-kurtosis relationship
sigvec = np.linspace(0,1,100)
ln_s = (np.exp(sigvec**2)+2)*np.sqrt(np.exp(sigvec**2)-1)
ln_k = np.exp(4*sigvec**2)+2*np.exp(3*sigvec**2)+3*np.exp(2*sigvec**2)-6

cam_label_col=(1,1,1)
unimodal_label_col=(0.88,0.88,0.88)

x = np.arange(-1.5,1.5,0.02)
plt.fill_between(x,1.5*x**2,5*np.ones(len(x)),zorder=-20,color=cam_label_col)
plt.fill_between(x,1.5*x**2,-2*np.ones(len(x)),zorder=-20,color=unimodal_label_col)
plt.fill_between(x,1*x**2-186/125,-2*np.ones(len(x)),zorder=-20,color=(0.5,0.5,0.5))

plt.plot(ln_s,ln_k,linewidth=1,color=ln_col,zorder=-15,label='Lognormal')
plt.plot(-ln_s,ln_k,linewidth=1,color=ln_col,zorder=-15)

plt.text(0.01,1.5,r'\textit{CAM}',horizontalalignment='center',rotation=0,color=(0,0,0),fontsize=20)
plt.text(-0.75,-0.95,r'\textit{unimodal}',horizontalalignment='center',rotation=-33,color=(0,0,0),fontsize=16)

handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,1,0]
plt.legend([handles[iord] for iord in order],[labels[iord] for iord in order])

plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
plt.xlim(-1.4,0.9)
plt.ylim(-1.1,4.5)

plt.subplot2grid((15,1),(0,0),rowspan=3)
plt.ylabel('Skewness')
plt.errorbar(range(0,len(division_names)),dc_div_sk[:,0],yerr=np.squeeze(dc_div_sk_conf[:,0,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='o-',ecolor=dc_col,color=(dc_col),label=r'$\delta^{13}$C')
plt.errorbar(range(0,len(division_names)),do_div_sk[:,0],yerr=np.squeeze(do_div_sk_conf[:,0,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='s-',ecolor=do_col,color=(do_col),label=r'$\delta^{18}$O')
plt.axhline(0,linestyle='--',color=cam_col,linewidth=2)
plt.ylim(-1.5,1.3)

plt.legend(framealpha=0.4)

ax3=plt.subplot2grid((15,1),(3,0),rowspan=3)
plt.ylabel('Kurtosis')
plt.errorbar(range(0,len(division_names)),dc_div_sk[:,1],yerr=np.squeeze(dc_div_sk_conf[:,1,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='o-',ecolor=dc_col,color=(dc_col),label=r'$\delta^{13}$C')
plt.errorbar(range(0,len(division_names)),do_div_sk[:,1],yerr=np.squeeze(do_div_sk_conf[:,1,:]).T,capsize=cap_size,elinewidth = eb_linewidth,capthick=eb_linewidth,fmt='s-',ecolor=do_col,color=(do_col),label=r'$\delta^{18}$O')
plt.axhline(0,linestyle='--',color=cam_col)
plt.ylim(-2,4.5)
ax3.set_yticks([-2,0,2,4])
ax3.set_xticks(np.arange(0,len(division_names),1))
ax3.set_xticklabels(division_names)

plt.subplots_adjust(top=0.945,bottom=0.08,left=0.115,right=0.975,hspace=0.0,wspace=0.2)
fig.text(0.015,0.97,r'$\textbf{A}$')
fig.text(0.015,0.55,r'$\textbf{B}$')

#############################################################
# Table 1 (fluctuation amplitude-mean d180 correlation analysis)
#############################################################

noise_corr_data = []
noise_corr = np.zeros((4))
noise_corr_p = np.zeros((4))

binwidth = 0.5 # Myr

for idiv in range(0,4):
    # inelegant removal of the PETM
    if idiv==1:
        divisions[idiv]=-55.5
    
    cind = (dc[:,0]>divisions[idiv])*(dc[:,0]<divisions[idiv+1])
    oind = (do[:,0]>divisions[idiv])*(do[:,0]<divisions[idiv+1])

    # do_bin is do + do_mean (i.e. the original values),
    # but indexed only at the indices of do_mean
    do_bin = do[:,[0,2]]
    do_bin[:,1] = do_bin[:,1] + do[:,1] 
    do_bin = do_bin[oind,:]

    # one-sided standard deviation
    do_sd = do[:,[0,4]]
    do_sd = do_sd[oind,:]

    # bin mean and sd
    binned_mean = bin_mean(do_bin,binwidth)
    binned_sd = bin_mean(do_sd,binwidth)

    # calculate correlations and p-values
    noise_corr_data.append([binned_mean[:,1],binned_sd[:,1]])
    noise_corr[idiv] = corr(noise_corr_data[idiv][0],noise_corr_data[idiv][1])
    noise_corr_p[idiv] = permutation(noise_corr_data[idiv][0],noise_corr_data[idiv][1],10000,noise_corr[idiv])

# print correlations and p-values by epoch
print(noise_corr,noise_corr_p)

# combined p-values
# note that the analysis above is inherently stochastic and provides very slightly different
# p-values each time. Below, we use the p-values shown in the paper.

# calculate Fisher's test statistic
fisher_stat = -2*(np.log(10**(-5))+np.log(0.247)+np.log(0.045)+np.log(0.06))
# obtain Chi-squared CDF from definition via Gamma function
fisher_p = 1 - gammainc(4,fisher_stat/2)

# harmonic mean p-value
hm_p = 1/(0.25*(1/(10**(-5))+ 1/0.247 + 1/0.045 + 1/0.06 ))

print(fisher_p,hm_p)

#############################################################
# Figure S3 (scatter plots by epoch and regression lines) 
#############################################################

fig, axes = plt.subplots(3,2)
colors = [(1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5)]

i = 0
for ax in axes.flat:
    dc_trim = dc[(dc[:,0]<divisions[i+1])*(dc[:,0]>divisions[i]),:]
    do_trim = do[(do[:,0]<divisions[i+1])*(do[:,0]>divisions[i]),:]
    ax.scatter(dc_trim[:,1],do_trim[:,1],alpha=1,s=5,color=colors[i])
    slope,intercept = rma(dc_trim[:,1],do_trim[:,1])
    x = np.linspace(np.nanmin(dc_trim[:,1]),np.nanmax(dc_trim[:,1]),10)
    ax.plot(x,slope*x+intercept,color=(0,0,0))
    # print(slope,intercept) 
    ax.set_xlabel(r'$\Delta \delta^{13}$C')
    ax.set_ylabel(r'$\Delta \delta^{18}$O')
    ax.set_title(division_names_full[i]+', slope = %3.2f' %(slope))
    i = i+1

plt.subplots_adjust(top=0.96,bottom=0.075,left=0.14,right=0.97,hspace=0.545,wspace=0.44)

plt.show()
