#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def _nyq1D(n):
    return n//2

def _nyq3D(n):
    return

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

def F3_integrals(x,n,dn,ns=128):
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

def F3_integrals_adapt(x,n,dn,ns=16):
    """
    Compute the angular integrals using an adaptive number of points depending on the radius in k.  This speeds up the computation since the key is to resolve wavelengths, which requires a variable number of points as a function of radius.  As implemented, this at least converges without having to tweak inputs, but isn't necessarily blazing fast.

    Input:
      ns  - Number of samples per wavelength
    """
    # To Do: Estimate scale from the positions

    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*nn)
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.empty(nb)

    # Test for speed, just compute the arrays for maximal size, and do slicings in the loop
    
    for i in range(nb):
        ii = i*dn
        sz = np.max([32,ns*np.int(np.ceil(ii))])
        tv,pv = np.meshgrid(np.linspace(0.,np.pi,sz,endpoint=False),np.linspace(0.,2.*np.pi,sz,endpoint=False))  # Can probably adapt this to theta to improve speed as well
        norm = (tv[0,1]-tv[0,0])*(pv[1,0]-pv[0,0]) / (4.*np.pi)
        k_x = np.cos(tv); k_y = np.sin(tv)*np.cos(pv); k_z = np.sin(tv)*np.sin(pv); st = np.sin(tv)
        form[i] = norm*np.sum(st*np.abs(np.sum(np.exp(1j*2.*np.pi*ii*(k_x[...,np.newaxis]*x[:,0]+k_y[...,np.newaxis]*x[:,1]+k_z[...,np.newaxis]*x[:,2])),axis=-1))**2 )
    return form, dn*np.arange(nb)

def F3_sample_reduced(x,n,dn):
    """
    Return form factor sampled on lattice using a binning approach and only summing over the positive k-values.

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - Bin width to use

    TO DO: Add windowing with various windows.  Adjust bin sizes
    TO DO: Fix treatment of the zero modes?
    """
    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*nn)
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.zeros(nb); w = np.zeros(nb)
    
    # I have to kill the zero mode in here (doing the array conversion is a terrible idea in python, it's horrifically slow)
    for nx,ny,nz in product(nv,nv,nv):  # Oops, this is wrong, but maybe it doesn't matter?
        i = np.floor(np.sqrt(nx**2+ny**2+nz**2)/dn); i = np.int(i)
        form[i] += np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2 # Think about this
        w[i] += 1.
    return form, w, dn*np.arange(nb)

def F3_sample(x,n,dn):
    """
    Return form factor sampled on lattice using a binning approach

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - Bin width to use

    TO DO: Add windowing with various windows.  Adjust bin sizes
    TO DO: Fix this to work properly for odd grids by weighting bins with Nyquist values properly
    """
    nn = n//2+1; nv = np.arange(nn); npos = np.arange(1,nn); nmax = np.int(np.sqrt(3.)*(n//2))+2
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.zeros(nb); w = np.zeros(nb)

    # Fix Nyquist counting for odd grids
    for nx,ny,nz in product(nv,nv,nv):
        i = np.int( np.floor(np.sqrt(nx**2+ny**2+nz**2)/dn) )
        nzeros = 3 - np.nonzero([nx,ny,nz])[0].size
        nnyq = 3 - np.nonzero([nn-1-nx,nn-1-ny,nn-1-nz])[0].size
        norm = 1./2.**(nzeros+nnyq)  
        dform  = ( np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                   )
        # Weight modes properly accounting for zeros so that negative isn't part of grid
        form[i] += norm*dform
        w[i] += norm*8.  
    return form, w, dn*np.arange(nb)    

def F3_center_bin(x,n,dn):
    """
    Return form factor sampled on lattice using a binning approach with centered bins

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - Bin width to use

    TO DO: Add windowing with various windows.  Adjust bin sizes
    TO DO: Fix this to work properly for odd grids by weighting bins with Nyquist values properly
    """
    nn = n//2+1; nv = np.arange(nn); nmax = np.int(np.sqrt(3.)*(n//2))+2
    nb = np.int(np.ceil(nmax/dn)) 
    form = np.zeros(nb); w = np.zeros(nb)

    # Fix Nyquist counting for odd grids
    for nx,ny,nz in product(nv,nv,nv):
        i = np.int( np.floor(np.sqrt(nx**2+ny**2+nz**2)/dn + 0.5) )  # Fix this, not correct for arbitrary dn 
        nzeros = 3 - np.nonzero([nx,ny,nz])[0].size
        nnyq = 3 - np.nonzero([nn-1-nx,nn-1-ny,nn-1-nz])[0].size
        norm = 1./2.**(nzeros+nnyq)  
        dform  = ( np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                   )
        form[i] += norm*dform
        w[i] += norm*8.  
    return form, w, dn*np.arange(nb)    

def F3_window_reduced(x,n,dn):
    """
    Return form factor sampled on lattice using a Welch window

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - bin width (not yet implemented)
    """

    # This is all broken for dn != 1.  Have to compute max differently then
    nn = n//2+1; nv = np.arange(nn)
    nmax = np.floor(np.sqrt(3.)*(n//2)/(1.*dn))+1+1 # 1 for 0-mode, 1 to get next k above max
    nb = np.int(nmax)
    form = np.zeros(nb); w = np.zeros(nb)
    wtmp = np.empty(2)
    
    for nx,ny,nz in product(nv,nv,nv):
        r = np.sqrt(nx**2+ny**2+nz**2)/dn; i = np.int(np.floor(r))
        wtmp[0] = (1.-(i-r)**2)**2; wtmp[1] = (1.-(i+1-r)**2)**2 
        form[i:i+2] += wtmp * np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
        w[i:i+2] += wtmp
    return form,w,dn*np.arange(nb)

def F3_window(x,n,dn):
    """
    Return form factor sampled on lattice using a Welch window

    Input:
      x  - Positions of oscillons
      n  - Number of lattice sites (assumed same in all directions)
      dn - bin width (not yet implemented)
    """

    # This is all broken for dn != 1.  Have to compute max differently then
    nn = n//2+1; nv = np.arange(nn)
    nmax = np.floor(np.sqrt(3.)*(n//2)/(1.*dn))+1+1 # 1 for 0-mode, 1 to get next k above max
    nb = np.int(nmax)
    form = np.zeros(nb); w = np.zeros(nb)
    wtmp = np.empty(2)

    # Fix Nyquist counting for odd grids
    for nx,ny,nz in product(nv,nv,nv):
        r = np.sqrt(nx**2+ny**2+nz**2)/dn; i = np.int(np.floor(r))
        i = np.int( np.floor(np.sqrt(nx**2+ny**2+nz**2)/dn) )
        nzeros = 3 - np.nonzero([nx,ny,nz])[0].size
        nnyq = 3 - np.nonzero([nn-1-nx,nn-1-ny,nn-1-nz])[0].size
        norm = 1./2.**(nzeros+nnyq)
        wtmp[0] = (1.-(i-r)**2)**2; wtmp[1] = (1.-(i+1-r)**2)**2 

        dform  = ( np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]+nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]+ny*x[:,1]-nz*x[:,2]))))**2
                 + np.abs(np.sum(np.exp(1j*2.*np.pi*(-nx*x[:,0]-ny*x[:,1]-nz*x[:,2]))))**2
                   )
        # Weight modes properly accounting for zeros so that negative isn't part of grid
        form[i:i+2] += norm*dform * wtmp
        w[i:i+2] += norm*8. * wtmp
    return form,w,dn*np.arange(nb)

def F3_dens(n,dn):
    nn = n//2+1; nn0 = n//2; nv = np.arange(nn); nf = np.arange(n); nmax = np.int(np.sqrt(3.)*(n//2))+2
    nb = np.int(np.ceil(nmax/dn))

    w = np.zeros((nb,2)); wtmp = np.empty(2)
    for nz in nf:
        nnz = nz
        if (nz > nn0):
            nnz = n-nz
        for ny in nf:
            nny = ny
            if (ny > nn0):
                nny = n-ny
            for nx in nf:
                nnx = nx
                if (nx > nn0):
                    nnx = n-nx
                r = np.sqrt(nnx**2+nny**2+nnz**2); i = np.int(np.floor(r))
                wtmp[0] = (1.-(r-i)**2)**2; wtmp[1] = (1.-(r+1-i)**2)**2
                w[i,0] += 1.
                w[i:i+2,1] += wtmp
    return w, dn*np.arange(nb)
    
def _get_nb(n):
    nn = n//2+1
    nmax = np.floor(np.sqrt(3.)*(n//2)/(1.*dn))+1+1
    
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

def formFactorSamples_uniform(nOsc,nl,ns):
    """
    Produce samples of form factors for oscillons randomly distributed in a given volume

    Input:
       nOsc - Number of oscillons
       nl   - Number of lattice sites per dimension
       ns   - Number of oscillon distributions to sample

    Output:
       f    - ns X nb array of samples (with nb number of k-bin samples)
       w    - weights to normalize f by
       k    - k-values (in units of fundamental grid spacing)
    """
    for i in range(ns):
        a,w,k = F3_sample(np.random.uniform(size=(nOsc,3)),nl,1)
    nb = k.size
        
    f = np.empty((ns,nb))
    f[0,:] = a
    for i in range(1,ns):
        a,w,k = F3_sample(np.random.uniform(size=(nOsc,3)),nl,1)
        f[i,:] = a
    return f,w,k

def uniform_random(nOsc):
    return np.random.uniform(size=(nOsc,3))

def uniform_pert(nOsc):
    x1 = np.linspace(0.,1.,nOsc,endpoint=False)
    xm = np.array(list(product(x1,x1,x1)))
    dx = np.random.uniform(size=(xm.shape[0],3))  # fix to be the right size 
    return xm + dx

def random_cluster(nCluster,cSize,dr,nd=3):
    """
    Randomly sample clusters of given numbers of oscillons uniformly distributed in volume.
    Currenlty, the cluster centers are drawn randomly from the volume, then cSize positions are randomly drawn from a cube of side length dr around these centers

    Input:
       nCluster - Number of clusters
       cSize    - Number of oscillons per cluster
       dr       - Fractional size of a cluster (in units of side length

    Output:


    To Do : Sample in some sphere not a cube, add Gaussian correlations, etc.  Allow for randomly number of oscillons per cluster
    """
    xm = np.random.uniform(size=(nCluster,nd))
    x2 = np.random.uniform(low=-0.5*dr,high=0.5*dr,size=(nCluster,cSize,nd))
    return np.reshape(xm[:nCluster,np.newaxis,:nd]+x2,(-1,nd))

def main():
    return
    
if __name__=="__main__":
    main()
