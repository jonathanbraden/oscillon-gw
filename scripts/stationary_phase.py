#!/usr/bin/env python
# Testing the stationary phase approximation

# To Do: Filter the post production curve to get the mean value, plot this as a function of k for fixed H0 and w.  This seems to have some nontrivial error induced by boundary corrections.
# Extend this for various time-dependence in the source, both as powers in a and in an exponent, check how this changes amplitude calculation

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Cosmology(object):
    def __init__(self,H0,w):
        self.H0 = H0; self.w = w
        
    def eta_t(self,t):
        return eta(t,self.H0,self.w)

    def a_t(self,t):
        return a(t,self.H0,self.w)
    
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

full_c = lambda y,t,k,H0,w : np.cos(t)**2*np.cos(k*eta(t,H0,w)) #/ a(t,H0,w)
full_s = lambda y,t,k,H0,w : np.cos(t)**2*np.sin(k*eta(t,H0,w)) #/ a(t,H0,w)
part_c = lambda y,t,k,H0,w : 0.25*np.cos(k*eta(t,H0,w)-2.*t) #/ a(t,H0,w)
part_s = lambda y,t,k,H0,w : 0.25*np.sin(k*eta(t,H0,w)-2.*t) #/ a(t,H0,w)

early_c_o1 = lambda t,k,H0,w : 0.25*(np.sin(k*eta(t,H0,w)-2.*t)-np.sin(0.))/(k/a(t,H0,w)-2.)
early_s_o1 = lambda t,k,H0,w : -0.25*(np.cos(k*eta(t,H0,w)-2.*t)-np.cos(0.))/(k/a(t,H0,w)-2.)
early_c_o2 = lambda t,k,H0,w : 0.25*(np.sin(k*eta(t,H0,w)-2.*t)-np.sin(0.))/(k/a(t,H0,w)-2.) + 0.25*(np.cos(k*eta(t,H0,w)-2.*t)-np.cos(0.))*(k/a(t,H0,w))*H0*a(t,H0,w)**(-1.5-1.5*w)/(k/a(t,H0,w)-2.)**3
early_s_o2 = lambda t,k,H0,w : -0.25*(np.cos(k*eta(t,H0,w)-2.*t)-np.cos(0.))/(k/a(t,H0,w)-2.) + 0.25*(np.sin(k*eta(t,H0,w)-2.*t)-np.sin(0.))*(k/a(t,H0,w)*H0*a(t,H0,w)**(-1.5-1.5*w))/(k/a(t,H0,w)-2.)**3

# Squared amplitude
amp_k = lambda k,H0,w : ( np.pi/16./H0*(k/2.)**(1.5*(1+w)) )**0.5 #/ (0.5*k)
t_k = lambda k,H0,w   : (2./3./(1+w))/H0 * ( (0.5*k)**(1.5*(1+w)) - 1. )
eta_k = lambda k,H0,w : eta(t_k(k,H0,w),H0,w)
phase_k = lambda k,H0,w : k*eta_k(k,H0,w) - 2.*t_k(k,H0,w) - 0.25*np.pi

def integral_plots_tmp(tv = np.linspace(0.,50.*np.pi,1500),k=5.,H0=0.1,w=0.):
#    tv = np.linspace(0.,50.*np.pi,1500)

    arg = (k,H0,w)
    sol_f_c = odeint(full_c,0.,tv,args=arg); sol_f_s = odeint(full_s,0.,tv,args=arg)
    sol_p_c = odeint(part_c,0.,tv,args=arg); sol_p_s = odeint(part_s,0.,tv,args=arg)
    
    fig,ax = plt.subplots()
    tau = eta(tv,H0,w)
    col_c = 'r'; col_s = 'b'
    ax.plot(tau,sol_f_c,alpha=0.3,color=col_c); ax.plot(tau,sol_f_s,alpha=0.3,color=col_s)
    ax.plot(tau,sol_p_c,color=col_c); ax.plot(tau,sol_p_s,color=col_s)

    step = np.where(tv<t_k(k,H0,w),0.,1.)
    ax.plot(tau,amp_k(k,H0,w)*step*np.cos(phase_k(k,H0,w)),'--',color=col_c)
    ax.plot(tau,amp_k(k,H0,w)*step*np.sin(phase_k(k,H0,w)),'--',color=col_s)
    ax.plot(tau,amp_k(k,H0,w)*step*np.cos(phase_k(k,H0,w))+early_c_o1(tv,k,H0,w),'-.',color=col_c)
    ax.plot(tau,amp_k(k,H0,w)*step*np.cos(phase_k(k,H0,w))+early_c_o2(tv,k,H0,w),'--',color=col_c)
    ax.plot(tau,amp_k(k,H0,w)*step*np.sin(phase_k(k,H0,w))+early_s_o2(tv,k,H0,w),'--',color=col_s)
    ax.plot(tau,amp_k(k,H0,w)*step*np.sin(phase_k(k,H0,w))+early_s_o1(tv,k,H0,w),'-.',color=col_s)
    

    col_sum = 'k'
    ax.plot(tau,sol_p_c**2+sol_p_s**2,color=col_sum)
    ax.plot(tau,sol_f_c**2+sol_f_s**2,alpha=0.3,color=col_sum)
    ax.plot(tau,amp_k(k,H0,w)**2*step,'--',color=col_sum)

    ax.set_xlabel(r'$\tau$')
    
    return fig,ax
    
if __name__=="__main__":
    pass
