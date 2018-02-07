#!/usr/bin/env python
import numpy as np

# explain the assumed storage of the array

# Fix this subroutine so that the Fourier space array can be complex
# I think I put in the necessary fixes, but should be checked, especially with the squared matrices
def TT_Power(pij,k):
    """
    Compute Tr(P^\dagger P) for the TT projection of the given matrix at the specified wavenumbers

    To Do: use some masking so that I can replace explicit sums with shorter vectorized operations
    To Do: Fix this so that it's using intelligent summations for C-ordering
    """
    # Convert these to some masking arrays
#    kx = [ True, False, False, True, True, False ]
#    ky = [ False, True, False, True, False, True ]
#    kz = [ False, False, True, False, True, True ]
    kx = [0,3,4]; ky = [1,3,5]; kz = [2,4,5]
    d_i = [0,1,2]
    k2d_i = 0  # combine above into a matrix for kSqk
    
    trSq = np.sum(np.abs(pij[0:3])**2,axis=0) + 2.*np.sum(np.abs(pij[3:])**2,axis=0)
    tr   = np.sum(pij[0:3],axis=0)
    kSk  = np.sum(k**2*pij[0:3],axis=0) + 2.*(k[0]*k[1]*pij[3]+k[0]*k[2]*pij[4]+k[1]*k[2]*pij[5])
    KSqk = k[0]**2*np.sum(np.abs(pij[kx])**2,axis=0) + k[1]**2*np.sum(np.abs(pij[ky])**2,axis=0) + k[2]**2*np.sum(np.abs(pij[kz])**2,axis=0)
    # Define some indices to reduce the horrible complexity of this expression
    kSqk += 2.*np.real(k[0]*k[1]*(pij[3]*np.conj(pij[0]) + pij[1]*np.conj(pij[3]) + pij[5]*np.conj(pij[4])) + k[0]*k[2]*(pij[]*np.conj(pij[])+pij[]*np.conj(pij[])) ) + k[1]*k[2]*(0.))
#    kSqk = k[0]**2*(np.abs(pij[0])**2+np.abs(pij[3])**2+np.abs(pij[4])**2) + k[1]**2*(np.abs(pij[1]**2 +  ) + 2.*(k[0]*k[1]*(0.) + k[0]*k[2]*(0.) + k[1]*k[2]*(0.) )
    return trSq - 2.*kSqk  + np.abs(kSk)**2 - 0.5*np.abs(tr - kSk)**2


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
