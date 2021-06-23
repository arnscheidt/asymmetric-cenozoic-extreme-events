#!/usr/bin/env python3

# This file produces Figure S4 of Arnscheidt and Rothman - Asymmetry of extreme Cenozoic climate-carbon cycle events (2021)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,ticker

rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')

# unit conventions: numbers are in moles, pressure is in atm, temperature in C, S in per mil, everything else is SI

#####################################################
# Define various constants
#####################################################

# Millero et al (2008)
bt = 0.0004151          # total boron

# carbonate dissociation  
# DOE (1994)
def k1(T,S):
    T = T + 273.15
    return np.exp(2.83855 - 2307.1266/T - 1.5529413*np.log(T)-(0.207608410+4.0484/T)*np.sqrt(S) + 0.0846834*S - 0.00654208*S**(1.5) + np.log(1-0.001005*S))

def k2(T,S):
    T = T + 273.15
    return np.exp(-9.226508 -3351.6106/T - 0.2005743*np.log(T)-(0.106901773+23.9722/T)*np.sqrt(S) + 0.1130822*S - 0.00846934*S**(1.5) + np.log(1-0.001005*S))

# henry's law constant for CO2
# Weiss (1974)
def k0(T,S):
    T = T + 273.15
    return np.exp(9345.17/T - 60.2409 + 23.3585*np.log(T/100) + S*(0.023517-0.00023656*T+0.0047036*(T/100)**2))

# ion product of water
# DOE (1994)
def kw(T,S):
    T = T + 273.15
    return np.exp(148.96502 - 13847.26/T - 23.6521*np.log(T)+(118.67/T-5.977+1.0495*np.log(T))*np.sqrt(S)-0.01615*S)

# boric acid dissociation constant
# DOE (1994)
def kb(T,S):
    T = T + 273.15
    return np.exp((-8966.90 - 2890.53*np.sqrt(S)-77.942*S+1.728*S**(3/2)-0.0996*S**2)/T+148.0248+137.1942*np.sqrt(S) + 1.62142*S - (24.4344+25.085*np.sqrt(S)+0.2474*S)*np.log(T)+0.053105*np.sqrt(S)*T)

#####################################################
# MAIN CODE
#####################################################

# calculate P(I,h) and Alk(I,h)
def Ih(I,h):

    # local constants
    T = 10
    S = 35
    mc = 12/1000
    rho = 1027

    # from Sarmiento and Gruber (2006)
    V = 1.34*10**18 
    Ma = 5.13*10**18 

    K0 = k0(T,S)
    K1 = k1(T,S)
    K2 = k2(T,S)
    Kw = kw(T,S)
    Kb = kb(T,S)
   
    p = I/(mc*V*rho*K0*(1+K1/h+K1*K2/(h**2))+Ma)
    alk = ((K0*I)/(mc*V*rho*K0*(1+K1/h+K1*K2/(h**2))+Ma))*(K1/h+2*K1*K2/(h**2)) + bt*Kb/(Kb+h)+Kw/h - h 
    return p,alk

# generate arrays of I and h
Iarr = np.linspace(0,70000,200)*10**12 # in kg
harr = np.flipud(10**(-np.linspace(1,12,200)))
print(harr)

I2,h2 = np.meshgrid(Iarr,harr,indexing = 'ij')

# calculate p and alk on the grid of (I,h)
p2 = np.empty_like(I2)
alk2 = np.empty_like(I2)
for iI in range(0,len(Iarr)):
    for ih in range(0,len(harr)):
        p2[iI,ih] = Ih(Iarr[iI],harr[ih])[0]
        alk2[iI,ih] = Ih(Iarr[iI],harr[ih])[1]


# calculate p(I) for fixed alkalinity
alk = 2400*10**(-6)
p_alk = np.empty_like(Iarr)
for iI in range(0,len(Iarr)):
    ih = np.argmin((alk2[iI,:]-alk)**2)
    p_alk[iI] = p2[iI,ih]

# plot p(I,h) and alk(I,h)
#plt.figure()
#cplt = plt.contourf(I2*10**(-12),-np.log10(h2),p2*10**6)
#cbar = plt.colorbar(cplt)
#
#plt.figure()
#cplt = plt.contourf(I2*10**(-12),-np.log10(h2),alk2)
#cbar = plt.colorbar(cplt)

# PLOT FIGURE S4
plt.figure()
plt.plot(Iarr*10**(-12),p_alk*10**(6),linewidth=2,label='Numerical solution')
Iarr_pg = Iarr*10**(-12)
plt.plot(Iarr_pg, 7000*((Iarr_pg)**6.5)/((Iarr_pg)**6.5+58000**6.5),label=r'$\chi\frac{I^{\gamma}}{I^{\gamma}+I_T^{\gamma}}$',linewidth=2) 

plt.xlabel('I (Pg)')
plt.ylabel(r'$p$CO$_2$ ($\mu$atm)')
plt.legend()

plt.show()

