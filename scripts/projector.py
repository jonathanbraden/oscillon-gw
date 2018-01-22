#!/usr/bin/env python
import numpy as np

# explain the assumed storage of the array

def derivDiscrete(p,right=True):
    """
    First order finite difference, uncentered
    """
    dp = np.diff(p)
    if right:
        return np.hstack((dp,p[0]-p[-1]))
    else:
        return np.hstack((p[0]-p[-1],dp))

# This assumes the thing is periodic
def derivDiscreteCenter(p):
    """
    Second order centered finite difference
    """
    dp = p[2:]-p[:-2]
    return np.hstack((p[1]-p[-1],dp,p[0]-p[-2]))
    
def derivSpectral(P):
    fk = np.fft.rfft(P)
    kv = np.fft.rfftfreq(P.size)
    return P.size*np.fft.irfft(1j*kv*fk,P.size)

def laplacianSpectral(P):
    fk = np.fft.rfft(P)
    kv = np.fft.rfftfreq(P.size)
    return P.size**2*np.fft.irfft(-kv**2*fk,P.size)

def derivSpectral2D(P):
    fk = np.fft.rfft2(P)
    kx,ky = np.fft.fftfreq(P.shape[0]), np.fft.rfftfreq(P.shape[1])

    return np.array([ P.shape[0]*np.fft.irfft2(1j*kx[:,np.newaxis]*fk,P.shape), P.shape[1]*np.fft.irfft2(1j*ky[np.newaxis,:]*fk,P.shape) ])

def laplacianSpectral2D(P):
    fk = np.fft.rfft2(P)
    kx,ky = np.fft.fftfreq(P.shape[0]), np.fft.rfftfreq(P.shape[1])
    
def create_profile(n):
    p = np.empty((n,n,n))
    return p

def stressTensor(P):
    """
    Compute the stress-energy tensor from a profile
    """
    return

def trace(M):
    """
    Compute the trace
    """
    return 

def quadForm(M,k):
    """
    Compute the quadratic form k^TMk
    """
    return

def projVec(M,k):
    """
    Compute vector M.dot.k
    """
    return

def traceless(M,d=3):
    """
    Return trace free tensor
    """
    return

if __name__=="__main__":
    pass
