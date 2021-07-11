import numpy as np
import scipy.special

"""
Collects functions defining and evaluating the Jolanta model potential
"""


"""----------------------------------------------------------

Functions for explicit evaluation used in with DVRs


"""


def Jolanta_1D(x, a=0.2, b=0.0, c=0.14):
    """
    default 1D potential:
    bound state:  -12.26336 eV
    resonance: (3.279526396 - 0.2079713j)  eV
    """
    return (a*x*x-b)*np.exp(-c*x*x)

def Jolanta_1Db(x, param):
    """
    c.f. Jolanta_1D
    """
    a, b, c = param
    return (a*x*x-b)*np.exp(-c*x*x)

def Jolanta_3D(r, param, l=1, mu=1):
    """
    standard 1D Jolanta potential in radial form
    plus angular momentum potential
    param=(0.028, 1.0, 0.028), l=1, and mu=1 gives
    Ebound in eV = -7.17051, and 
    Eres in eV = (3.1729420714-0.160845j)
    """
    a, b, c = param
    return (a*r**2-b)*np.exp(-c*r**2) + 0.5*l*(l+1)/r**2/mu

def Jolanta_3D_old(r, a=0.1, b=1.2, c=0.1, l=1, as_parts=False):
    """
    default 3D potential; has a resonance at 1.75 eV - 0.2i eV
    use for DVRs
    """
    if as_parts:
        Va = a*r**2*np.exp(-c*r**2)
        Vb = b*np.exp(-c*r**2)
        Vc = 0.5*l*(l+1)/r**2
        return (Va, Vb, Vc)
    else:
        return (a*r**2-b)*np.exp(-c*r**2) + 0.5*l*(l+1)/r**2


"""----------------------------------------------------------------

Representations in a Gaussian basis set

1D = one-dimensional = straightforward

3D = three-dimensional
      - add an l=1 angular momentum term
      - solve the radial Schroedinger equation
      - careful with the integrals for u = R*r 
      - R are p-GTOs, so u are d-GTOs
"""


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

def Jolanta_3D_PNorm(a):
    """
    see Analytic integrals notebook in Stab directory for formulas
    integrals of two GTOs: r*exp(-a_j*r**2)  dV = r**2 dr
    return the normalization 1/sqrt(S_jj)
    R is a p-fn, u is a D-fn: 
    4 * 2**(3/4) * sqrt(3) * a1**(5/4) / (3*pi**(1/4))
    """
    return 4 * 2**(3/4) * np.sqrt(3) * a**(5/4) / (3*np.pi**(1/4))



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


def Jolanta_3D_GTO(a1, a2, param, l=0):
    """
    see Analytic integrals notebook in Stab directory for formulas
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


def Jolanta_3D_W_old(a1, a2, rc):
    """
    CAP potential = (r-rc)**2 for x > rc; else 0
    a1, a2 exponents of GTO
    u(r) = r*R(r) 
    int(GTO_1 * w(r) * GTO_2) from rc to oo
    -(6*sqrt(pi)*a1*rc**2*sqrt(a1 + a2)*exp(rc**2*(a1 + a2))*erf(rc*sqrt(a1 + a2)) 
    - 6*sqrt(pi)*a1*rc**2*sqrt(a1 + a2)*exp(rc**2*(a1 + a2)) 
    + 2*a1*rc + 6*sqrt(pi)*a2*rc**2*sqrt(a1 + a2)*exp(rc**2*(a1 + a2))*erf(rc*sqrt(a1 + a2)) 
    - 6*sqrt(pi)*a2*rc**2*sqrt(a1 + a2)*exp(rc**2*(a1 + a2)) + 2*a2*rc 
    + 15*sqrt(pi)*sqrt(a1 + a2)*exp(rc**2*(a1 + a2))*erf(rc*sqrt(a1 + a2)) - 15*sqrt(pi)*sqrt(a1 + a2)*exp(rc**2*(a1 + a2)))*exp(-rc**2*(a1 + a2))/(16*(a1**4 + 4*a1**3*a2 + 6*a1**2*a2**2 + 4*a1*a2**3 + a2**4))
    """
    sp = np.sqrt(np.pi)
    s12 = np.sqrt(a1 + a2)
    ex12 = np.exp(rc**2*(a1 + a2))
    ex12m = np.exp(-rc**2*(a1 + a2))
    erf = scipy.special.erf(rc*s12)
    
    W = -(  6*sp*a1*rc**2*s12*ex12*erf 
          - 6*sp*a1*rc**2*s12*ex12 
          + 2*a1*rc 
          + 2*a2*rc 
          + 6*sp*a2*rc**2*s12*ex12*erf  
          - 6*sp*a2*rc**2*s12*ex12 
          + 15*sp*s12*ex12*erf 
          - 15*sp*s12*ex12
          ) * ex12m / (16*(a1+a2)**4)
    return W



def Jolanta_3D_W(a, rc):
    """
    computes int_rc^oo dr  r**4 * exp(-a*r**2) * w(r)
    w(r) = (r-rc)**2 for x > rc; else 0

    this is for radial p-GTO: u(r) = R(r)*r
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
    


def Jolanta_GTO_H(GTO_fn, alphas, Ns, param, l=1):
    """
    Hamiltonian matrix in a GTO basis set
    
    Parameters
    ----------
    GTO_fn : either Jolanta_1D_GTO() or Jolanta_3D_GTO()
    alphas : np.array of GTO exponents
    Ns : np.array of normalization constants
    param : (a, b, c): parameters of the Jolanta potential
    l : = 0 (even) or 1 (odd) in the 1D case; for 3D ignored

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


def Jolanta_GTO_W(alphas, Ns, rc):
    """
    CAP potential w(r) matrix representation in a GTO basis set
    
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
        W[i,i] = Ni * Ni * Jolanta_3D_W(ai+ai, rc)
        for j in range(i):
            aj, Nj = alphas[j], Ns[j]
            W[i,j] = W[j,i] = Ni * Nj * Jolanta_3D_W(ai+aj, rc)
    return W




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
    

