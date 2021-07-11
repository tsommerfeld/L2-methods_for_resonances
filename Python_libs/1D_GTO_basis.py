#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:41:41 2021

@author: thomas
"""

"""  
GTO functions for the 1D Jolanta potential 
this has not been used much as 1D should not work with RAC 
"""

import numpy as np


def Jolanta_1D_Norm(a, l=1):
    """
    see Analytic integrals notebook in Stab directory for formulas
    integrals of two GTOs: x*exp(-a_j*x**2)
    return the normalization 1/sqrt(S_jj)
    S:     2**(1/4) * a1**(1/4) / pi**(1/4)
    P: 2 * 2**(1/4) * a1**(3/4) / pi**(1/4)
    """
    if l == 0:
        return  (2*a/np.pi)**0.25
    else :
        return 2 * (2/np.pi)**0.25 * a**0.75 


def Jolanta_1D_GTO(a1, a2, param, l=1):
    """
    see Analytic integrals notebook in Stab directory for formulas
    integrals of two GTOs: x*exp(-a_j*x**2)
    computes overlap, kinetic energy, and potential
    """
    a, b, c = param
    sqrt_pi = np.sqrt(np.pi)
    if l == 0:
        S = sqrt_pi /np.sqrt(a1 + a2)
        T = sqrt_pi * a1 * a2 / (a1 + a2)**(3/2)
        V = sqrt_pi*(a - 2*a1*b - 2*a2*b - 2*b*c)/(2*(a1 + a2 + c)**(3/2))
    else:
        S = sqrt_pi / (2*(a1 + a2)**(3/2))
        T = 1.5 * sqrt_pi * a1 * a2 / (a1 + a2)**(5/2)
        V = sqrt_pi*(3*a - 2*a1*b - 2*a2*b - 2*b*c)/(4*(a1 + a2 + c)**(5/2))
    
    return S, T, V


def Jolanta_GTO_H(GTO_fn, alphas, Ns, param, l=1):
    """
    Hamiltonian matrix in a GTO basis set
    
    Parameters
    ----------
    GTO_fn : either Jolanta_1D_GTO()
    alphas : np.array of GTO exponents
    Ns : np.array of normalization constants
    param : (a, b, c): parameters of the Jolanta potential
    l : = 0 (even) or 1 (odd) in the 1D case

    Returns 3 numpy matrices
    -------
    S : overlap matrix
    T : kinetic energy matrix
    V : potential energy matrix

    """
    nbas = len(alphas)
    S=np.zeros((nbas,nbas))
    T=np.zeros((nbas,nbas))
    V=np.zeros((nbas,nbas))
    for i in range(nbas):
        ai, Ni = alphas[i], Ns[i]
        S[i,i], T[i,i], V[i,i] = GTO_fn(ai, ai, param, l=l)
        S[i,i] *= Ni*Ni
        T[i,i] *= Ni*Ni
        V[i,i] *= Ni*Ni
        for j in range(i):
            aj, Nj = alphas[j], Ns[j]
            Sij, Tij, Vij = GTO_fn(ai, aj, param, l=l)
            S[i,j] = S[j,i] = Ni*Nj * Sij
            T[i,j] = T[j,i] = Ni*Nj * Tij
            V[i,j] = V[j,i] = Ni*Nj * Vij
    return S, T, V



def Eval_GTO_wf(alphas, Ns, cs, xs, l=1):
    """
    This is the 1D function
    input:
        alphas, norms = a basis set 
        cs = GTO coefficient vector
        alphas, Ns, and cs define a wavefunction
        xs = positions at which the wf is to be evaluated 
    """
    nx = len(xs)
    nbas = len(cs)
    ys = np.zeros(nx)
    for i in range(nx):
        y=0
        #xfactor = xs[i]**l
        xsq = xs[i]**2
        for k in range(nbas):
            y += cs[k] * Ns[k] * np.exp(-alphas[k]*xsq)
        ys[i] = y*xs[i]**l
    return ys 