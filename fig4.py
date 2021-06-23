#!/usr/bin/env python3

# This file produces Figure 4 of Arnscheidt and Rothman - Asymmetry of extreme Cenozoic climate-carbon cycle events (2021)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,ticker

rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')

# DEFINE HELPER FUNCTIONS
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

# LOAD DATA FROM ENSEMBLE RUN
# the data are produced by model.jl
# the runs shown in the paper are in the below .csv files
ensemble_T = np.loadtxt("ens_400_T.csv",delimiter=',') 
ensemble_dc = np.loadtxt("ens_400_dc.csv",delimiter=',')-1 

t = np.arange(0,4.01,.01)

# calculate S, K
n_ens = len(ensemble_T[1,:])
do_sk = np.zeros((n_ens,2)) 
dc_sk = np.zeros((n_ens,2)) 

for i_ens in range(0,n_ens):
    mean = np.mean(ensemble_T[:,i_ens])
    sd = np.sqrt(np.mean((ensemble_T[:,i_ens]-mean)**2))
    do_sk[i_ens,0] = -np.mean((((ensemble_T[:,i_ens]-mean)/sd)**3)) # because T ~ -d18o
    do_sk[i_ens,1] = np.mean((((ensemble_T[:,i_ens]-mean)/sd)**4))-3

    mean = np.mean(ensemble_dc[:,i_ens])
    sd = np.sqrt(np.mean((ensemble_dc[:,i_ens]-mean)**2))
    dc_sk[i_ens,0] = np.mean((((ensemble_dc[:,i_ens]-mean)/sd)**3))
    dc_sk[i_ens,1] = np.mean((((ensemble_dc[:,i_ens]-mean)/sd)**4))-3

# highlighted trajectory
i =10 

fig = plt.figure()

dc_col = (0,0,0.8)
do_col = (0.8,0,0)
dc_line_col = (0,0,0)
do_line_col = (0,0,0)

axline_col = (0.8,0.8,0.8)

ax1 = plt.subplot2grid((9,6),(0,0),colspan=7,rowspan=2)

# note it is here that the temperature time series is converted to d18O
# using the conversion coefficient from Bemis et al. (1998)
plt.plot(t,ensemble_T/-4.8,linewidth=0.5,alpha = 0.2)
plt.plot(t,ensemble_T[:,i]/-4.8,linewidth=1.5,color=do_line_col)

ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
plt.xlabel('Time (Myr)')

plt.ylim(0.49,-1)
plt.ylabel(r'$\Delta \delta^{18}$O')
plt.xlim(0,4)

ax2 = ax1.twinx()
plt.ylim(-0.49*4.8,1*4.8)
plt.ylabel(r'$\Delta T$ (K)')

ax3 = plt.subplot2grid((9,6),(2,0),colspan=7,rowspan=2)
plt.ylim(0.49,-1)
plt.plot(t,ensemble_dc,linewidth=0.5,alpha = 0.2)
plt.plot(t,ensemble_dc[:,i],linewidth=1.5,color=dc_line_col)
plt.xlim(0,4)
plt.ylabel(r'$\Delta \delta^{13}$C')
ax3.set_xticks([])

plt.subplots_adjust(top=0.92,bottom=0.075,left=0.075,right=0.925,hspace=0.0,wspace=0.8)

ax5 = plt.subplot2grid((9,6),(5,0),colspan=3,rowspan=4)
plt.scatter(dc_sk[:,0],dc_sk[:,1],label=r'$\delta^{13}$C',color=dc_col,alpha=0.7,zorder=-5)
plt.scatter(do_sk[:,0],do_sk[:,1],label=r'$\delta^{18}$O',color=do_col,marker='s',alpha=0.7,zorder=-5)
plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
s = np.linspace(-3,3,100)

cam_label_col=(1,1,1)
unimodal_label_col=(0.88,0.88,0.88)

plt.fill_between(s,1.5*s**2-0.9,10*np.ones(len(s)),zorder=-20,color=cam_label_col)
plt.fill_between(s,1.5*s**2-0.9,-2*np.ones(len(s)),zorder=-20,color=unimodal_label_col)
plt.fill_between(s,1*s**2-2,-2*np.ones(len(s)),zorder=-20,color=(0.5,0.5,0.5))
plt.text(0.01,3,r'\textit{CAM}',horizontalalignment='center',rotation=0,color=(0,0,0),fontsize=20)
plt.text(-1.92,1.15,r'\textit{unimodal}',horizontalalignment='center',rotation=-53,color=(0,0,0),fontsize=14)

# calculate lognormal skewness-kurtosis relationship
sigvec = np.linspace(0,1,100)
ln_s = (np.exp(sigvec**2)+2)*np.sqrt(np.exp(sigvec**2)-1)
ln_k = np.exp(4*sigvec**2)+2*np.exp(3*sigvec**2)+3*np.exp(2*sigvec**2)-6
plt.plot(ln_s,ln_k,linewidth=2,color=(0,0,0),zorder=0,label='Lognormal')
plt.plot(-ln_s,ln_k,linewidth=2,color=(0,0,0),zorder=0)

handles, labels = ax5.get_legend_handles_labels()
order = [1,2,0]
plt.legend([handles[iord] for iord in order],[labels[iord] for iord in order],loc='upper right',framealpha=1,fontsize=14)

plt.xlim(-2.5,2.5)
plt.ylim(-2,8)

ax6 = plt.subplot2grid((9,6),(5,3),colspan=3,rowspan=4)
scatter_color = (0.8,0.8,0.8)
plt.scatter(ensemble_dc,ensemble_T/(-4.8),color=scatter_color,s=10,alpha=0.4)

slope,intercept = rma(ensemble_dc.flatten(),ensemble_T.flatten()/(-4.8))
x = np.linspace(-1.5,0.5,10)
plt.plot(x,x*slope+intercept,zorder=10,color=(0,0,0),linewidth=3,label='Model')

# the below data are obtained from the output of fig1_fig2_tab1_figS3.py
slope = [0.85,0.77,1.06,0.93]
intercept = [-0.00016,0.00026,0.0039,-0.0033]
epochs = ['Paleocene','Eocene','Oligocene','Miocene']
color = [(0.8,0,0),(0,0,1),(0,0.8,0),(0,0,0)]
linestyle=['--','-.',':',':']
for i in range(0,4):
    plt.plot(x,x*slope[i]+intercept[i],zorder=5-i,color=color[i],linestyle=linestyle[i],linewidth=3,label=epochs[i])
plt.xlabel(r'$\Delta \delta^{13}$C')
plt.ylabel(r'$\Delta \delta^{18}$O')

plt.legend(loc='lower right',fontsize=14)
ax6.yaxis.tick_right()
ax6.yaxis.set_label_position('right')

fig.text(0.01,0.97,r'$\textbf{A}$')
fig.text(0.01,0.47,r'$\textbf{B}$')
fig.text(0.5,0.47,r'$\textbf{C}$')

plt.show()

