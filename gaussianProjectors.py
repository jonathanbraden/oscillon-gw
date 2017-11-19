#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# Real Space Distributions
Txy_o2 = lambda x,y,dx : np.exp(-dx**2)*np.exp(-(x**2+y**2))*np.sinh(x*dx)*np.sinh(y*dx)/dx**2
Txx_o2 = lambda x,y,dx : np.exp(-dx**2)*np.exp(-(x**2+y**2))*np.sinh(x*dx)**2/dx**2
Txy_exact = lambda x,y,dx : np.exp(-(x**2+y**2))*x*y
Txx_exact = lambda x,y,dx : np.exp(-(x**2+y**2))*x**2

# These are written to reduce roundoff errors
Txy_diff = lambda x,y,dx : np.exp(-(x**2+y**2))*(np.exp(-dx**2)*np.sinh(x*dx)*np.sinh(y*dx)/dx**2-x*y)
Txx_diff = lambda x,y,dx : np.exp(-(x**2+y**2))*(np.exp(-dx**2)*np.sinh(x*dx)**2/dx**2-x**2)

# Fourier Space distributions
#Tkxy_o2 = lambda x,y,dx : 

def error_convergence():
    dxv = np.logspace(-3.,1.,51); err = np.empty((dxv.size,2))
    xv, yv = np.linspace(-2.,2.,201), np.linspace(-2.,2.,201)
    for i,dx in enumerate(dxv):
        err[i,0] = np.max(np.abs(Txx_diff(xv,yv,dx)))
        err[i,1] = np.max(np.abs(Txy_diff(xv,yv,dx)))
    return dxv, err

def plot_Txy_num(dx):
    return plot_Tij_slice(Txy_o2, dx)

def plot_Txx_num(dx):
    return plot_Tij_slice(Txx_o2,dx)

def plot_Txx_analytic():
    return plot_Tij_slice(Txx_exact,0.1)

def plot_Txy_analytic():
    return plot_Tij_slice(Txy_exact,0.1)

def plot_Txy_diff(dx):
    return plot_Tij_slice(lambda x,y,dx : np.exp(-(x**2+y**2))*(np.sinh(x*dx)*np.sinh(y*dx)/dx**2-x*y),dx)    

def plot_Txx_diff(dx):
    return plot_Tij_slice(lambda x,y,dx : np.exp(-(x**2+y**2))*(np.sinh(x*dx)**2/dx**2-x**2),dx)    
def plot_Tij_slice(func,dx):
    x = np.linspace(-3.,3.,201); y = np.linspace(-3.,3.,201)
    xv,yv = np.meshgrid(x,y)

    fig, ax = plt.subplots()
    cnt = ax.contourf(xv,yv,func(xv,yv,dx),51,cmap='coolwarm')

    ax.set_xlabel(r'$x/\sigma$'); ax.set_ylabel(r'$y/\sigma$')
    fig.colorbar(cnt,fraction=0.05,pad=0.02)
    return fig,ax

# Now make plots in Fourier space
#T_xx_o2_k = lambda kx,ky,dx : 
#T_xy_o2_k = lambda kx,ky,dx : -np.exp(0.5*dx**2)*np.exp(-0.25*(kx**2*ky**2)

def plot_Tij_slice_k(func,dx):
    x = np.linspace(-np.pi,np.pi,201); k = np.linspace(-np.pi*np.pi,201)
    kx,ky = np.meshgrid(x,y)

    fig,ax = plt.subplots()
    cnt = ax.contourf(kx,ky,func(kx,ky,dx),51,cmap='coolwarm')

    ax.set_xlabel(r'$k_x dx$'); ax.set_ylabel(r'$k_y dx$')
    fig.colorbar(cnt,fraction=0.05,pad=0.02)
    return fig,ax

def plot_slices_gaussian(dx=0.25):
    fig,ax = plot_Txx_analytic()
    fig.savefig('txx-analytic.pdf')
    fig,ax = plot_Txy_analytic()
    fig.savefig('txy-analytic.pdf')
    fig,ax = plot_Txx_num(dx)
    fig.savefig('txx-numerical-dx%g.pdf' % dx)
    fig,ax = plot_Txy_num(dx)
    fig.savefig('txy-numerical-dx%g.pdf' % dx)
    fig,ax = plot_Txx_diff(dx)
    fig.savefig('txx-diff-dx%g.pdf' % dx)
    fig,ax = plot_Txy_diff(dx)
    fig.savefig('txy-diff-dx%g.pdf' % dx)
    
def main():
    return

if __name__=="__main__":
    pass
