#!/usr/bin/env python

# Testing the stationary phase approximation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def eta(t,H0,w):
    """
    Return conformal time at given cosmic time

    Input:
      H0 - Initial Hubble (in units of source frequency omega)
      w  - Equation of state parameter (assumed constant)
    """
    return (2./H0)/(3.*w+1) * ( (1.+1.5*(1.+w)*H0*t)**((3.*w+1)/(3.*(1.+w))) - 1. )

def a(t,H0,w):
    """
    Return the scale factor in cosmic time for give H_0/omega and EOS w

    Input:
      H0 - Initial Hubble (in units of source frequency omega)
      w  - Equation of state parameter (assumed constant)
    """
    return (1. + 1.5*(1+w)*H0*t)**(2./(3.*(1+w)))

def a_eta(tau,H0,w):
    return

def t_eta(tau,H0,w):
    """
    Return cosmic time as function of conformal time
    """
    return

def tt(t,k,H0,w):
    acur = a(t,H0,w)
    return np.exp(-0.25*k**2/acur**2)/acur

def plotPhaseFunc(H0,w,k):
    tv = np.linspace(0.,50.*np.pi,1500)
    plt.plot(a(tv,H0,w),np.cos(k*eta(tv,H0,w))*np.cos(tv)**2)
    plt.plot(a(tv,H0,w),np.cos(k*eta(tv,H0,w)-2.*tv))
    plt.plot(a(tv,H0,w),np.cos(k*eta(tv,H0,w)+2.*tv))
    plt.plot(a(tv,H0,w),np.cos(k*eta(tv,H0,w)))

# Make some super temporary plots to demonstrate stationary phase
#
# To Do: Add an xaxis to the top
def paper_plots_tmp():
    tv = np.linspace(0.,50.*np.pi,1500); H0=0.1; w=0.; k=5.
    fig,ax = plt.subplots()
    ax.plot(eta(tv,H0,w),0.25*np.cos(k*eta(tv,H0,0.)-2.*tv),'r--',label=r'Diff')
    ax.plot(eta(tv,H0,w),np.cos(k*eta(tv,H0,0.))*np.cos(tv)**2,'r',label=r'Full',alpha=0.3)

    ax.set_xlabel(r'$\omega(\eta - \eta_0)$')
    ax.set_ylabel(r'Integrand')
    return fig,ax

full_c = lambda y,t : np.cos(t)**2*np.cos(5.*eta(t,0.1,0.))
full_s = lambda y,t : np.cos(t)**2*np.sin(5.*eta(t,0.1,0.))
part_c = lambda y,t : 0.25*np.cos(5.*eta(t,0.1,0.)-2.*t)
part_s = lambda y,t : 0.25*np.sin(5.*eta(t,0.1,0.)-2.*t)

def integral_plots_tmp():
    tv = np.linspace(0.,50.*np.pi,1500); H0=0.1; w=0.; k=5.

    sol_f_c = odeint(full_c,0.,tv); sol_f_s = odeint(full_s,0.,tv)
    sol_p_c = odeint(part_c,0.,tv); sol_p_s = odeint(part_s,0.,tv)
    
    fig,ax = plt.subplots()
    tau = eta(tv,H0,w)
    ax.plot(tau,sol_f_c); ax.plot(tau,sol_f_s)
    ax.plot(tau,sol_p_c); ax.plot(tau,sol_p_s)

    ax.plot(tau,sol_p_c**2+sol_p_s**2)
    ax.plot(tau,sol_f_c**2+sol_f_s**2)
    return fig,ax
    
if __name__=="__main__":
    pass
