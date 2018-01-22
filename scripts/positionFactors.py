#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

gauss = lambda x,x0,s : np.exp(-s*(x-x0)**2)

def make_profile(prof,xv,par,sum=False):
    if sum:
        p = make_background_periodic_sum(prof,xv,par)
    else:
        p = make_background_periodic(prof,xv,par)
    return

def make_background_periodic(prof,xv,par):
    """
    Create a periodically continued profile located at the given position
    """
    return prof(xv,par)

def make_background_periodic_sum(prof,xv,par):
    return

def field_extend(prof,xv,par):
    dx = xv[1]-xv[0]; l = (xv[-1]-xv[0])+dx
    return


# These are monolithic-like routines that should be modularised
def vary_dx(l=2.,s=100.,x0=0.5):
    n = [ 16, 32, 64, 128 ]
    xv = [ np.linspace(-0.5*l,0.5*l,nv,endpoint=False) for nv in n ]
    fv_1 = [ np.exp(-s*x**2) for x in xv ]      # need to periodicize this
    fv_2 = [ np.exp(-s*(x-x0)**2) for x in xv ]  # need to periodicize this

    fk_g = [ np.fft.rfft(f) for f in fv_1 ]
    fk_s = [ np.fft.rfft(f1+f2) for (f1,f2) in zip(fv_1,fv_2) ]
    kv = [ 2.*np.pi / l * np.arange(nc/2+1) for nc in n ]
    return fk_g, fk_s, kv

def vary_l(n,s=100.,x0=0.5):
    print n
    lv = [ 2., 4., 8., 16. ]; nv = [ n, 2*n, 4*n, 8*n ]
    xv = [ np.linspace(-0.5*l,0.5*l,n,endpoint=False) for (l,n) in zip(lv,nv) ]
    fv_1 = [ np.exp(-s*x**2) for x in xv ]      # need to periodicize this
    fv_2 = [ np.exp(-s*(x-x0)**2) for x in xv ]  # need to periodicize this

    fk_g = [ np.fft.rfft(f) for f in fv_1 ]
    fk_s = [ np.fft.rfft(f1+f2) for (f1,f2) in zip(fv_1,fv_2) ]
    kv = [ 2.*np.pi / l * np.arange(n//2+1) for (n,l) in zip(nv,lv) ]
    return fk_g, fk_s, kv

def vary_x0(l=2.,n=64,s=100.):
    return

def prefactor(x0,kv):
    return 2.*np.cos(0.5*kv*x0)
             
if __name__=="__main__":
    pass
