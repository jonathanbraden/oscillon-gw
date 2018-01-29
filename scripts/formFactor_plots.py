import numpy as np
import matplotlib.pyplot as plt
from formFactor import *

def plot_two_oscillon_convergence_len(dr):
    """
    Show how the convergence to the two oscillon state in the infinite volume limit is approached with 2 oscillons on a periodic lattice.
    Currently assumes that both oscillons lie along the x-axis.

    Input:
      dr - Separation in fraction of smallest box
    
    TO DO: Add vertical line for Nyquist
    """
    fig,ax = plt.subplots()
    nBase = 16
    for i in range(5):
        xv = np.array([[0.,0.,0.],[dr/2**i,0.,0.]])
        fs,ws,ks = F3_sample(xv,nBase*2**i,1)
        ax.plot(ks/2.**i,fs/ws,label=r'$%i$' % (2**i))
    kk = np.linspace(0.,nBase,500)
    ax.plot(kk,F3_exact(dr,2.*np.pi*kk),label=r'$\infty$')
    ax.set_xlabel(r'$L_{%i}k / 2\pi$' % nBase); ax.set_ylabel(r'$F_2$')

    ax.set_ylim(1.,5.)
    plt.legend(loc='upper center',bbox_transform=fig.transFigure,bbox_to_anchor=(0,0,1,1),ncol=2,title=r'$L / L_{\rm %i}$' % nBase)  # fix this
    ax.text(0.5,0.05,r'$L^{-1}_{\rm %i}\Delta r$' % nBase,horizontalalignment='center',verticalalignment='bottom',transform=ax.transAxes)

    # Add ticker fix in here
    return fig,ax

def plot_two_oscillon_convergence_dx(dr):
    """
    Show how the grid spacing influences the form factor for two oscillons in a box of fixed side length.
    Currently assumes that both oscillons lie along the x-axis.

    Input:
      dr - Separation in fraction of smallest box
    
    TO DO: Add vertical line for Nyquist
    """
    fig,ax = plt.subplots()
    nBase = 16
    for i in range(3):
        xv = np.array([[0.,0.,0.],[dr,0.,0.]])
        fs,ws,ks = F3_sample(xv,nBase*2**i,1)
        ax.plot(ks,fs/ws,label=r'$%i$' % (2**i))
    kk = np.linspace(0.,nBase,500)
#    ax.plot(kk,F3_exact(dr,2.*np.pi*kk),label=r'$\infty$')
    ax.set_xlabel(r'$kL / 2\pi$'); ax.set_ylabel(r'$F_2$')

    ax.set_ylim(1.,5.)
    plt.legend(loc='upper center',bbox_transform=fig.transFigure,bbox_to_anchor=(0,0,1,1),ncol=2,title=r'$N / N_{\rm %i}$' % nBase)  # fix this
    ax.text(0.5,0.05,r'$L^{-1}_{\rm %i}\Delta r$' % nBase,horizontalalignment='center',verticalalignment='bottom',transform=ax.transAxes)

    # Add ticker fix in here
    return fig,ax

from itertools import product
def plot_uniform_x_3d(nx,nl,kv=None,fig=None,ax=None,testInt=None):
    """
    Plot form factor for 3D uniform grid of oscillons with 

    Input:
      nx - Number of oscillons in a single direction
      nl - Number of lattice sites in one direction

      fig - Figure to plot
      ax   - Axis to draw on, if not provided create a new one
      testInt - If True, do sampling integral with different bin sizes to test convergence

    Returns
      fig - The figure containing the axis
      ax  - The axis the plot is drawn on
    """
    x1 = np.linspace(0.,1.,nx,endpoint=False)
    xv = np.array(list(product(x1,x1,x1)))
    
    if ax == None:
        fig,ax = plt.subplots()
        
    f,w,k = F3_sample(xv,nl,1)
    ax.plot(k[1:],f[1:]/w[1:]/nx**3)
    f,k,a = F3_integrals(xv,nl,1)
    ax.plot(k[1:],f[1:]/nx**3,'ro')
    f,k,a = F3_integrals(xv,nl,0.1)
    ax.plot(k[1:],f[1:]/nx**3,'r')
    if testInt is not None:
        f,k,a = F3_integrals(xv,nl,0.1,ns=testInt)
        ax.plot(k[1:],f[1:]/nx**3,'r--')

    ax.set_ylabel(r'$F_N / N_{\rm osc}$')
    ax.set_xlabel(r'$kL / 2\pi$')
    return fig,ax

def plot_uniform_random_3d(nOsc,nl,fig=None,ax=None,testInt=None):
    """
    Plot form factor for 3D uniform grid of oscillons with 

    Input:
      nOsc - Number of oscillons
      nl - Number of lattice sites in one direction

      fig - Figure to plot
      ax   - Axis to draw on, if not provided create a new one
      testInt - If True, do sampling integral with different bin sizes to test convergence

    Returns
      fig - The figure containing the axis
      ax  - The axis the plot is drawn on
    """
    xv = np.random.uniform(size=(nOsc,3))
    if ax == None:
        fig,ax = plt.subplots()
        
    f,w,k = F3_sample(xv,nl,1)
    ax.plot(k[1:],f[1:]/w[1:]/xv.shape[0])
    f,k,a = F3_integrals(xv,nl,1)
    ax.plot(k[1:],f[1:]/xv.shape[0],'ro')
    f,k,a = F3_integrals(xv,nl,0.1)
    ax.plot(k[1:],f[1:]/xv.shape[0],'r')
    if testInt is not None:
        f,k,a = F3_integrals(xv,nl,0.1,ns=testInt)
        ax.plot(k[1:],f[1:]/xv.shape[0],'r--')
    return fig,ax

def plot_form_factor_contours(fName):
    """
    Read in a list of form factors and plot various useful statistics.

    To Do: Work out all the implied increments so I plot the correct thing (should really use an odd number of samples to the 50% is on a sample, here can just average)
    """
    data = np.load(fName); k = data['k']; w = data['weight']
    ff = np.sort(data['form'],axis=0)
    ns = ff.shape[0]
    
    fig,ax = plt.subplots()
    cuts = np.array(np.array([0.005,0.025,0.2])*(ns+1),dtype=int)  # Check the indices for off by ones
    for i in cuts:
        ax.fill_between(k,ff[i]/w,ff[-i]/w,alpha=0.2)
    ax.plot(k,ff[(ns+1)/2]/w,'b')
    ax.plot(k,np.mean(ff,axis=0)/w,'m')
    ax.axvline(x=32,color='r')

    # Add a few randomly sampled trajectories (This only works if I haven't already sorted the trajectories ...
    for i in np.random.random_integers(ff.shape[0],size=5):
        plt.plot(k,data['form'][i]/w,'k--',alpha=0.5)

    ax.set_xlabel(r'$kL / 2\pi$'); ax.set_ylabel(r'$F_{\rm N}$')
    ax.set_xlim(0.,np.max(k))
    return fig,ax
    
def main():
    return

if __name__=="__main__":
    main()
