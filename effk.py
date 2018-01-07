#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# Make some plots of effective wavenumbers

k2_1 = lambda k,th,phi : np.sin(0.5*k*np.cos(th))**2+np.sin(0.5*k*np.sin(th)*np.cos(phi))**2+np.sin(0.5*k*np.sin(th)*np.sin(phi))**2
k2_2 = lambda k,th,phi : (np.cos(th)*np.sin(0.5*k*np.cos(th)) + np.sin(th)*np.cos(phi)*np.sin(0.5*k*np.sin(th)*np.cos(phi)) + np.sin(th)*np.sin(phi)*np.sin(0.5*k*np.sin(th)*np.sin(phi)) )**2
k2_3 = lambda k,th,phi : np.cos(th)**2*np.sin(0.5*k*np.cos(th))**2 + np.sin(th)**2*np.cos(phi)**2*np.sin(0.5*k*np.sin(th)*np.cos(phi))**2 + np.sin(th)**2*np.sin(phi)**2*np.sin(0.5*k*np.sin(th)*np.sin(phi))**2

def plot_k2_diff(kv):
    tv = np.linspace(0.,np.pi,201); pv = np.linspace(0.,2.*np.pi,201)
    xv,yv = np.meshgrid(tv,pv)
    sin_v = np.sin(xv)
    
    fig,ax = plt.subplots()
    cnt = ax.contourf(xv,yv,sin_v*(k2_1(kv,xv,yv)-k2_2(kv,xv,yv)))
    return fig,ax,cnt

# Use a better integrator here
def compute_k2_diff_volume(kv,ns=501):
    tv = np.linspace(0.,np.pi,ns); pv = np.linspace(0.,2.*np.pi,ns)
    xv,yv = np.meshgrid(tv,pv)

    k2diff = np.sin(xv)*(k2_1(kv,xv,yv)-k2_2(kv,xv,yv))
    return 2.*np.pi**2*np.mean(k2diff)
    
if __name__=="__main__":
    pass
