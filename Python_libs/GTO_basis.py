#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 08:54:27 2021

@author: Thomas Sommerfeld
"""

"""
A class to manage a Gaussian basis for the Jolanta potential with l=1

Lots of helper functions not in the class to keep the legacy notebooks working.
"""

import numpy as np
import scipy.special
from scipy.linalg import eigh


class GBR:

    def __init__(self, alphas, param, contract=(0, 0), diffuse=(0, 1.4)):
        """
        alphas: exponents alphas
        aparam: Jolanta-3D parameters (a,b,c)
        contract(nc, nu): 
            uses the lowest nc eigenvectors of H (only nc==1 makes sense)
            and adds nu uncontracted GTOs starting from the smallest alpha
        diffuse(n_diff, scale):
            after contraction, add n_diff even-tempered diffuse functions
            alphas[-1]/s, alphas[-1]/s**2, ... 
        """

        self.n_val = len(alphas)
        self.n_diff, scale = diffuse
        self.param = param
        self.nc, self.nu = contract
        
        self.alphas = np.zeros(self.n_val + self.n_diff)
        self.alphas[:self.n_val] = np.sort(alphas)[::-1]
        for j in range(self.n_diff):
            self.alphas[self.n_val+j] = self.alphas[self.n_val+j-1]/scale

        self.Ns = np.zeros(self.n_val + self.n_diff)        
        for j, a in enumerate(self.alphas):
            self.Ns[j] = Jolanta_3D_PNorm(a)

        """  uncontracted S, T, and V_Jolanta """
        self.Sun, self.Tun, self.Vun = Jolanta_GTO_H(self.alphas, self.Ns, self.param)

        """ contraction matrix C with shape (n_primitive, n)contracted) """
        self.C = 0
        if self.nc == 0:
            """ no contraction = all basis function """
            self.S, self.T, self.V = self.Sun, self.Tun, self.Vun
            return
        """ else: contract with valence eigenstates """
        nv = self.n_val
        Sval, Tval, Vval = self.Sun[:nv,:nv], self.Tun[:nv,:nv], self.Vun[:nv,:nv]
        Es, cs = eigh(Tval + Vval, b=Sval)
        n_prim = self.n_val + self.n_diff
        n_cont = self.nc + self.nu + self.n_diff
        self.C = np.zeros((n_prim, n_cont))
        self.C[:nv,:self.nc] = cs[:,:self.nc]
        for j in range(n_cont - self.nc):
            self.C[-1-j,-1-j] = 1.0
        
        self.S = np.linalg.multi_dot([self.C.T, self.Sun, self.C])
        self.T = np.linalg.multi_dot([self.C.T, self.Tun, self.C])
        self.V = np.linalg.multi_dot([self.C.T, self.Vun, self.C])


    def exponents(self):
        return self.alphas
    
    def normalization_constants(self):
        return self.Ns
        
    def print_exp(self):
        print("      alpha         r0=1/sqrt(alpha)     Norm")
        for j, a in enumerate(self.alphas):
            print(f"  {a:15.8e}   {np.sqrt(1/a):15.8e}   {self.Ns[j]:11.4e}")

    def contraction_matrix(self):
        return self.C

    def STV(self):
        return self.S, self.T, self.V
    
    def V_jolanta(self, params):
        """ 
        returns the Jolanta(l=1) potential with different parameters 
        does not change self.V
        """
        Sun, Tun, Vun = Jolanta_GTO_H(self.alphas, self.Ns, params)
        if self.nc == 0:
            return Vun
        else:
            return np.linalg.multi_dot([self.C.T, Vun, self.C])

    
    def H_theta(self, theta, alpha):
        """ 
        theta: scaling angle for the radial coordinate r: exp(i*theta) 
        returns: the complex scaled Hamiltonian H(r*exp(i*theta))
        """
        z = alpha*np.exp(1j*theta)
        f = z**(-2)
        Vun_rot = Jolanta_GTO_VJrot(self.alphas, self.Ns, self.param, z)
        Hun_rot = f*self.Tun + Vun_rot
        if self.nc == 0:
            return Hun_rot
        else:
            return np.linalg.multi_dot([self.C.T, Hun_rot, self.C])
    
    def Wcap(self, rc):
        """ real matrix W for the CAP, where W(r<rc) = 0 """
        Wun = Jolanta_GTO_W(Jolanta_3D_Wcap, self.alphas, self.Ns, rc)
        if self.nc == 0:
            return Wun
        else:
            return np.linalg.multi_dot([self.C.T, Wun, self.C])

    def V_Coulomb(self):
        Vun = Jolanta_GTO_W(Jolanta_3D_Coulomb, self.alphas, self.Ns, 1.0)
        if self.nc == 0:
            return Vun
        else:
            return np.linalg.multi_dot([self.C.T, Vun, self.C])

    def V_softbox(self, rc):
        Vun =  Jolanta_GTO_W(Jolanta_3D_softbox, self.alphas, self.Ns, rc)
        if self.nc == 0:
            return Vun
        else:
            return np.linalg.multi_dot([self.C.T, Vun, self.C])
        
    def eval_vector(self, cs, rs, u=True):
        """ 
        plotting a wavefunction psi(r) = Sum c_i f_i
        where f_i can be primitive GTO or a contracted function
        parameters:
            cs : coefficient vector
            rs : positions at which to evalutes psi(r)
            u  : return u(r) or R(r), where R(r) = u(r)/r
        returns:
            ys : psi(r)
        """
        if self.nc > 0:
            c_un = np.matmul(self.C, cs)
            return Eval_GTO_wf_3D(self.alphas, self.Ns, c_un, rs, u)
        else:
            return Eval_GTO_wf_3D(self.alphas, self.Ns, cs, rs, u)



def Jolanta_3D_PNorm(a):
    """
    see Analytic integrals notebook in Stab directory for formulas
    integrals of two GTOs: r*exp(-a_j*r**2)  dV = r**2 dr
    return the normalization 1/sqrt(S_jj)
    R is a p-fn, u is a D-fn: 
    4 * 2**(3/4) * sqrt(3) * a1**(5/4) / (3*pi**(1/4))
    """
    return 4 * 2**(3/4) * np.sqrt(3) * a**(5/4) / (3*np.pi**(1/4))


def Jolanta_3D_GTO(a1, a2, param):
    """
    see Analytic integrals notebook in GTO directory for formulas
    integrals of two GTOs: x*exp(-a_j*x**2)
    computes overlap, kinetic energy, and potential
    R1 and R2 are p-fns, u1 and u2 are D-fns:
    the parameter l is ignored (so that 1D and 3D may call the same fn)
    """
    a, b, c = param
    sqrt_pi = np.sqrt(np.pi)
    S = 3 * sqrt_pi / (8*(a1 + a2)**2.5)
    T = sqrt_pi * (1.875*a1*a2 - 0.25*(a1 + a2)**2)/(a1 + a2)**3.5    
    VJ = 3 * sqrt_pi * (5*a - 2*a1*b - 2*a2*b - 2*b*c)/(16*(a1 + a2 + c)**3.5)
    VL = sqrt_pi / (4*(a1 + a2)**1.5)
    return S, T, VJ+VL


def Jolanta_GTO_H(alphas, Ns, param):
    """
    Hamiltonian matrix in the uncontracted GTO basis set
    
    Parameters
    ----------
    alphas : np.array of GTO exponents
    Ns : np.array of normalization constants
    param : (a, b, c): parameters of the Jolanta potential

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
        S[i,i], T[i,i], V[i,i] = Jolanta_3D_GTO(ai, ai, param)
        S[i,i] *= Ni*Ni
        T[i,i] *= Ni*Ni
        V[i,i] *= Ni*Ni
        for j in range(i):
            aj, Nj = alphas[j], Ns[j]
            Sij, Tij, Vij = Jolanta_3D_GTO(ai, aj, param)
            S[i,j] = S[j,i] = Ni*Nj * Sij
            T[i,j] = T[j,i] = Ni*Nj * Tij
            V[i,j] = V[j,i] = Ni*Nj * Vij
    return S, T, V




def Jolanta_3D_CS(a12, param, z):
    """
    computes int dr  r**4 * exp(-ag*r**2) * (VJ + Vl)
    VJ = (a*r**2 - b)*exp(-c*r**2) = Va - Vb 
    Vl = 1/r**2
    for r -> r*exp(i*theta)

    this is for a radial p-GTO: u(r) = R(r)*r
    u1*u2 = r**4 * exp(-(a1+a2)*r**2)

    z = alpha*exp(I*theta)
    both Va and Vb are valid only for 2*theta <= pi/2
    no problem as the max rotation angle is pi/4


    VJ(z*r) = (3*sqrt(pi)*(5*a*z**2 - 2*b*(a12 + c*z**2))
               /(16*(a12 + c*z**2)**(7/2))                                   
    """
    a, b, c = param
    sp = np.sqrt(np.pi)
    f = z**2
    #Va = 15*sp*a*f / (16*(a12 + c*f)**(7/2))
    #Vb = 3*sp*b / (8*(a12 + c*f)**(5/2))    
    VJ = ( 3*sp * (5*a*f - 2*b*(a12 + c*f))
          / (16*(a12 + c*f)**3.5) )
    VL = sp / (4*(a12)**1.5) / f
    return VJ + VL


def Jolanta_3D_Wcap(a, rc):
    """
    computes int_rc^oo dr  r**4 * exp(-a*r**2) * w(r)
    w(r) = (r-rc)**2 for x > rc; else 0

    this is for CAP radial p-GTO: u(r) = R(r)*r
    u1*u2 = r**4 * exp(-(a1+a2)*r**2)

    - rc*exp(-a*rc**2)/(8*a**3) 
    - 3*sqrt(pi)*rc**2*erf(sqrt(a)*rc)/(8*a**(5/2)) 
    + 3*sqrt(pi)*rc**2/(8*a**(5/2)) 
    - 15*sqrt(pi)*erf(sqrt(a)*rc)/(16*a**(7/2)) 
    + 15*sqrt(pi)/(16*a**(7/2))

    W = (- rc*exa/(8*a**3)
         - 3*sp*rc**2 * erf / (8*a**(5/2)) 
         + 3*sp*rc**2       / (8*a**(5/2)) 
         - 15*sp * erf / (16*a**(7/2)) 
         + 15*sp       / (16*a**(7/2))
         )

    W = (- rc*exa / (8*a**3)
         + 3*sp*rc**2 / (8*a**(5/2)) * (1 - erf) 
         + 15*sp / (16*a**(7/2)) * (1 - erf)
         )

    """
    sp = np.sqrt(np.pi)
    exa = np.exp(-a*rc**2)
    erf = scipy.special.erf(rc*np.sqrt(a))

    W = (- rc*exa / (8*a**3)
         + 3*sp*rc**2 / (8*a**(5/2)) * (1 - erf) 
         + 15*sp / (16*a**(7/2)) * (1 - erf)
         )

    return W


def Jolanta_3D_Coulomb(a, rc):
    """
    computes int_rc^oo dr  r**4 * exp(-a*r**2) * (-1/r)

    this is for RAC radial p-GTO: u(r) = R(r)*r
    u1*u2 = r**4 * exp(-(a1+a2)*r**2)
    
    rc is ignored (needed for function uniformity)
    
    returns -1/(2*a**2)
    """
    return -1/(2*a**2)


def Jolanta_3D_softbox(a, rc):
    """
    computes int_rc^oo dr  r**4 * exp(-a*r**2) * w(r)
    w(r) = exp(-4*rc**2/x**2) - 1

    this is for RAC radial p-GTO: u(r) = R(r)*r
    u1*u2 = r**4 * exp(-(a1+a2)*r**2)

    + 3*sqrt(pi)*rc*cosh(4*sqrt(a)*rc)/(2*a**2) 
    - 3*sqrt(pi)*rc*sinh(4*sqrt(a)*rc)/(2*a**2) 
    + 2*sqrt(pi)*rc**2*cosh(4*sqrt(a)*rc)/a**(3/2) 
    - 2*sqrt(pi)*rc**2*sinh(4*sqrt(a)*rc)/a**(3/2) 
    + 3*sqrt(pi)*cosh(4*sqrt(a)*rc)/(8*a**(5/2)) 
    - 3*sqrt(pi)*sinh(4*sqrt(a)*rc)/(8*a**(5/2)) 
    - 3*sqrt(pi)/(8*a**(5/2))

    observe: cosh(a) - sinh(a) = exp(-a)

    W = sp * (  3*rc*cosh/(2*a**2) 
              - 3*rc*sinh/(2*a**2) 
              + 2*rc**2*cosh/a**(3/2) 
              - 2*rc**2*sinh/a**(3/2) 
              + 3*cosh/(8*a**(5/2)) 
              - 3*sinh/(8*a**(5/2)) 
              - 3/(8*a**(5/2))
              )


    """
    sp = np.sqrt(np.pi)
    sqarc = np.sqrt(a)*rc
    #sinh = np.sinh(4*sqarc)
    #cosh = np.cosh(4*sqarc)
    exa = np.exp(-4*sqarc)
    
    W = sp * (  3*rc*exa/(2*a**2) 
              + 2*rc**2*exa/a**(3/2) 
              + 3*(exa-1)/(8*a**(5/2)) 
              ) 
    
    return W




def Jolanta_GTO_W(GTO_fn, alphas, Ns, rc):
    """
    potential w(r) matrix representation in a GTO basis set
    GTO_fn can be:
        Jolanta_3D_Wcap     for the quadratic soft-box for CAP
        Jolanta_3D_Coulomb  for a Coulomb potential for RAC
        Jolanta_3D_softbox  for a inverse GTO soft-box for RAC    
    Parameters
    ----------
    alphas : np.array of GTO exponents
    Ns : np.array of normalization constants
    rc : cutoff of w(r) 

    Returns 
    -------
    W : matrix represention of w(r)

    """
    nbas = len(alphas)
    W=np.zeros((nbas,nbas))
    for i in range(nbas):
        ai, Ni = alphas[i], Ns[i]
        W[i,i] = Ni * Ni * GTO_fn(ai+ai, rc)
        for j in range(i):
            aj, Nj = alphas[j], Ns[j]
            W[i,j] = W[j,i] = Ni * Nj * GTO_fn(ai+aj, rc)
    return W


def Jolanta_GTO_VJrot(alphas, Ns, param, z):
    """
    rotated Jolanta potential V_J(r*exp(I*theta)) in a GTO basis set
    ----------
    Parameters
    alphas : np.array of GTO exponents
    Ns : np.array of normalization constants
    param = (a,b,c) parameters of V_J = (a*r**2 - b)*exp(-c*r**2)
    z = alpha*exp(i*theta), where arg(z) < pi/4;   
    -------
    Returns 
    Vrot : matrix represention of V_J(r*exp(I*theta))
    """
    nbas = len(alphas)
    W=np.zeros((nbas,nbas), complex)
    for i in range(nbas):
        ai, Ni = alphas[i], Ns[i]
        W[i,i] = Ni * Ni * Jolanta_3D_CS(ai+ai, param, z)
        for j in range(i):
            aj, Nj = alphas[j], Ns[j]
            W[i,j] = W[j,i] = Ni * Nj * Jolanta_3D_CS(ai+aj, param, z)
    return W


       
    
def Eval_GTO_wf_3D(alphas, Ns, cs, xs, u=True):
    """
    This is the 3D function of l=1
    u(r) = r**2 * exp(-a*r**2)
    R(r) = r    * exp(-a*r**2)
    input:
        alphas, norms = a basis set 
        cs = GTO coefficient vector
        alphas, Ns, and cs define a wavefunction
        xs = positions at which the wf is to be evaluated
        u=True  evaluate the radial function u(r) = r*R(r)
        u=False evaluate the radial function R(r) = u(r)/r
    """
    if u:
        l=2
    else:
        l=1
        
    nx = len(xs)
    nbas = len(cs)
    ys = np.zeros(nx)
    for i in range(nx):
        y=0
        xsq = xs[i]**2
        for k in range(nbas):
            y += cs[k] * Ns[k] * np.exp(-alphas[k]*xsq)
        ys[i] = y*xs[i]**l
    return ys        
    

