#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def F1(x,kv):
    """
    Return 1D form factor (continuum version) for oscillons at positions x_i

    Input:
      - x  - Positions of oscillons
      - kv - k-values to evaluate the form factor at
    """
    x = np.array(x)
    xg,kg = np.meshgrid(x,kv)
    return np.abs(np.sum(np.exp(1j*kg*xg),axis=-1))**2

def F1_sample(x,L,n):
    """
    Return 1D form factor for oscillons at positions x_i sampled at lattice momenta

    Input:
       x - Positions of oscillons
       L - Side length of box
       n - Number of lattice sites
    """
    kv = (2.*np.pi/L)*np.arange(n//2+1)
    return kv,F1(x,kv)

F3_exact = lambda dr, k : 2.*(1.+np.sin(k*dr)/(k*dr))

def F3_integrals(x,n,dn,ns=101):
    """
    Return the form factor by performing the angular integrals numerically instead of
    sampling off a lattice.

    Input:
      x - Positions of the oscillons
      n - Number of lattice sites
      dn - Bin width (in units of lattice sites)

    Output:
      form - Angle averaged form factor
      dn   - Sample n-values (in units of grid spacing)
      area - Area of unit sphere normalized to 4pi (using the approximate integration here)

    # TO DO: Use a quadrature integrator here instead of discrete sampling so area is actually 1
    """
    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*nn)
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.empty(nb)

    # Speed at expense of memory
    tv,pv = np.meshgrid( np.linspace(0.,np.pi,ns,endpoint=False),np.linspace(0.,2.*np.pi,ns,endpoint=False) )
    norm = (tv[0,1]-tv[0,0])*(pv[1,0]-pv[0,0]) / (4.*np.pi)
    k_x = np.cos(tv); k_y = np.sin(tv)*np.cos(pv); k_z = np.sin(tv)*np.sin(pv); st = np.sin(tv)
    area = norm*np.sum(st)
    
    for i in range(nb):
        ii = i*dn
        form[i] = np.sum(st*np.abs(np.sum(np.exp(1j*2.*np.pi*ii*(k_x[...,np.newaxis]*x[:,0]+k_y[...,np.newaxis]*x[:,1]+k_z[...,np.newaxis]*x[:,2])),axis=-1))**2 )
    return norm*form, dn*np.arange(nb), area

def F3_sample(x,n,dn):
    """
    Return form factor sampled on lattice using a binning approach

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - Bin width to use

    TO DO: Add windowing with various windows.  Adjust bin sizes
    """
    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*nn)
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.zeros(nb); w = np.zeros(nb)
    
    # I have to kill the zero mode in here (doing the array conversion is a terrible idea in python, it's horrifically slow)
    for nx,ny,nz in product(nv,nv,nv):
        i = np.floor(np.sqrt(nx**2+ny**2+nz**2)/dn); i = np.int(i)
        form[i] += np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2 # Think about this
        w[i] += 1.
    return form, w, dn*np.arange(nb)

def F3_window(x,n,dn):
    """
    Return form factor sampled on lattice using a Welch window

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - bin width (not yet implemented)
    """
    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*nn)
    nb = np.int(np.ceil(nmax/dn))
    form = np.zeros(nb); w = np.zeros(nb)
    wtmp = np.empty(2)
    
    for nx,ny,nz in product(nv,nv,nv):
        r = np.sqrt(nx**2+ny**2+nz**2)/dn; i = np.int(np.floor(r))
        wtmp[0] = (1.-(r-i)**2)**2; wtmp[1] = (1.-(r+1-i)**2)**2 
        form[i:i+2] += wtmp * np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
        w[i:i+2] += wtmp
    return form,w,dn*np.arange(nb)
        
def plot_uniform_x(nx,nl,kv=None):
    """
    Plot form factors in 1D sampled at lattice momenta for oscillons uniformly separated from each other (and thus highly correlated).  Plots the form factors up to the specified maximum number of oscillons, normalized to n_osc

    Input:
       nx - Maximum number of oscillons to use
       nl - Number of lattice sites
    """
    fig,ax = plt.subplots()
    for i in range(1,nx+1):
        xv = np.linspace(0.,1.,i,endpoint=False)
        k,f = F1_sample(xv,1.,nl)
        ax.plot(k[1:],f[1:]/i)
        if kv is not None:
            f = F1(xv,kv)
            ax.plot(kv[1:],f[1:]/i,'--')
    return fig,ax

def plot_uniform_random_pert(nx,nl,ns=1000,frac=0.01):
    """
    Plot some form factors in 1D, assuming they are uniformly spaced with random perturbations from the equally space locations

    Input:
      - frac - Fraction of the spacing to fluctuate the oscillons around (should be less than 1/2)
      - nx   - Number of oscillons
      - nl   - Number of lattice sites
      - ns   - Number of samples
    """
    fig,ax = plt.subplots()
    dx = 1./nx; dr = frac*dx
    for i in range(ns):
        pert = np.random.uniform(low=-dr,high=dr,size=nx)
        k,f = F1_sample(np.linspace(0.,1.,nx,endpoint=False)+pert,1.,nl)
        ax.plot(k[1:],f[1:]/nx)
    
    return fig,ax

def plot_uniform_random(nx,nl,ns=1000):
    """
    Plot some form factors in 1D sampled at the lattice momenta for oscillons placed a randomly selected locations on the interval [0:L]

    Input:
       nx - Number of oscillons
       nl - Number of lattice sites
       ns - Number of form factor sample functions to plot
    """
    fig,ax = plt.subplots()
    for i in range(ns):
        k,f = F1_sample(np.random.uniform(size=nx),1.,nl)
        ax.plot(k[1:],f[1:]/nx)
    return fig,ax

def main():
    return
    
if __name__=="__main__":
    main()
