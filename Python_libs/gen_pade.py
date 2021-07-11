import numpy as np
#from scipy import sqrt

"""
Helper functions for using generalized or super Pade approximants:
    
    E^2*P + E*Q + R = 0
    
    where P, Q, and R are polynomials 
    
    normally all three have the same order, so Pade-[n,n,n]


"""



def genpade2_via_lstsq(nQ, nP, nS, xs, ys, rcond=1e-14, return_lists=False):
    """ 
    generalized Pade nq/np/ns approximant 
    y**2 * Q  + y * P  +  S  = 0  
    with Q, P, S being polynomials of x.   
    The coefficients of Q, P, and S are determined by LS fitting 
    numpy.linalg.lstsq is used, so any len(xs)=len(ys) will work. 
    One coefficient is redundant, and to account for this the linear
    coefficient of Q is arbitrarily set to 1.
    If this leads to huge coefficients, scale your x input.
    
    input:
        nQ, nP, nS: order of Q, P, and S
        xs, ys array with data to fit
        recond: handed to lstsq(); determines smoothness vs accuracy
        return_lists: return coefficients as lists or poly1d polynomials
    returned:
        Pade coefficients for Q, P, and S in highest-power first order
    """
    N_coef = nS + nP + nQ + 2
    M_data = len(xs)
    Q0=1
    Q0 = max(xs)-min(xs)
    if M_data < N_coef:
        print('Warning: Underdetermined system in genpade2')
    A=np.zeros((M_data,N_coef))
    b=np.zeros(M_data)
    for k_data in range(M_data):
        i_coef = 0
        E=ys[k_data]
        E2=E*E
        #
        for iq in range(nQ,1,-1): # counts from np down to 2
            A[k_data,i_coef] = E2*xs[k_data]**iq
            i_coef += 1
        if nQ > 0:
            A[k_data,i_coef] = E2*xs[k_data]
            i_coef += 1
        #
        for ip in range(nP,1,-1): # counts from np down to 2
            A[k_data,i_coef] = E*xs[k_data]**ip
            i_coef += 1
        if nP > 0:
            A[k_data,i_coef] = E*xs[k_data]
            i_coef += 1
        A[k_data,i_coef] = E
        i_coef += 1
        #
        for ks in range(nS,1,-1): # counts from np down to 2
            A[k_data,i_coef] = xs[k_data]**ks
            i_coef += 1
        if nS > 0:
            A[k_data,i_coef] = xs[k_data]
            i_coef += 1
        A[k_data,i_coef] = 1.0
        #
        b[k_data] = -E2*Q0 
    coefs, residual, rank, s = np.linalg.lstsq(A,b,rcond=rcond)
    Qs = np.array(list(coefs[:nQ]) + [Q0])
    Ps = coefs[nQ:nQ+nP+1]
    Ss = coefs[-nS-1:]
    if return_lists:
        return Qs, Ps, Ss
    else:
        return np.poly1d(Qs), np.poly1d(Ps), np.poly1d(Ss)


    
def E_lower(x, A, B, C):
    """ lower branch of a generalized Pade approximant """
    if A(x) < 0:
        return -0.5 * (B(x) - np.sqrt(B(x)*B(x)-4*A(x)*C(x)) ) / A(x)
    else:
        return -0.5 * (B(x) + np.sqrt(B(x)*B(x)-4*A(x)*C(x)) ) / A(x) 
    
def E_upper(x, A, B, C):
    """ upper branch of a generalized Pade approximant """
    if A(x) >= 0:
        return -0.5 * (B(x) - np.sqrt(B(x)*B(x)-4*A(x)*C(x)) ) / A(x)
    else:
        return -0.5 * (B(x) + np.sqrt(B(x)*B(x)-4*A(x)*C(x)) ) / A(x)


def dEdL(E, L, P, Q, R):
    """ 
    we know: E^2*P + E*Q + P = 0 
    therefore:
    dEdL = E' = -(E^2*P' + E*Q' + R')/(2*E*P + Q)
    input
    P, Q, R: three polynomials that depend on L
    E: the energy
    L: the independent (scaling) variable
    """
    Pp = P.deriv(1)(L)
    Qp = Q.deriv(1)(L)
    Rp = R.deriv(1)(L)
    return -(E**2*Pp + E*Qp + Rp) / (2*E*P(L) + Q(L))

def E_from_L(L, A, B, C):
    """ 
    given L, solve E^2*A + E*B + C = 0
    return roots
    """
    P = np.poly1d([A(L), B(L), C(L)])
    return P.roots

def E_and_Ep(L, A, B, C):
    """ 
    given L, solve E^2*A + E*B + C = 0
    for every root, compute dEdL
    return energies and abs(derivatives)
    """
    P = np.poly1d([A(L), B(L), C(L)])
    roots = P.roots
    ders = []
    for E in roots:
        ders.append(abs(dEdL(E, L, A, B, C)))
    return roots, ders

#
#  for Newton we solve dEdL = 0 or E' = 0
#
#  so we iterate L[i+1] = L[i] - E'/E'' 
#
#  the fraction E'/E'' can be worked out analytically: 
#  
# (E^2*P' + E*Q' + R')  /
# (2*P*E'^2 + 4*E*E'*P' + E^2*P'' + 2*E'*Q' + E*Q'' + R'')
#
def EpoEpp(E, L, P, Q, R):
    """ E'/E'' needed for Newton's method """
    Pp = P.deriv(1)(L)
    Qp = Q.deriv(1)(L)
    Rp = R.deriv(1)(L)
    Ep = -(E**2*Pp + E*Qp + Rp) / (2*E*P(L) + Q(L))
    Ppp = P.deriv(2)(L)
    Qpp = Q.deriv(2)(L)
    Rpp = R.deriv(2)(L)
    num = E**2*Pp + E*Qp + Rp
    den = 2*P(L)*Ep**2 + 4*E*Ep*Pp + E**2*Ppp + 2*Ep*Qp + E*Qpp + Rpp
    return num/den


def GPA_NewtonRaphson(z_guess, polys, max_step=20, ztol=1e-8, Etol=1e-8, verbose=True):
    """ 
    Newton-Raphson for generalized Pade in the complex plane
    E^2 * A + E * B + C = 0 where A, B, C are polynomials in z
    z_guess : start point
    poly : list of the polynomials defining the GPA
    max_step : maximal number of Newton steps
    ztol, Etol : tolerances in z and E(z) 
    """
    z_curr = z_guess
    A, B, C = polys
    roots, ders = E_and_Ep(z_curr, A, B, C)
    i_root = np.argmin(ders)
    Ecurr=roots[i_root]
    converged = False
    if verbose:
        print('Newton Raphson steps:')
        print(' step    z_curr                   E_curr')
        print('-----------------------------------------------------')
        #print('  0   (0.9250560, 0.6512459)   (3.1595040, 0.1494027)')
    for i in range(max_step):
        delta_z = EpoEpp(Ecurr, z_curr, A, B, C)
        if i == 0:
            delta_0 = abs(delta_z) * 10
        z_curr = z_curr - delta_z
        # compute new Ecurr (two roots, pick closer one to Ecurr)
        Es = E_from_L(z_curr, A, B, C)
        delta_E = min(abs(Es-Ecurr))
        Ecurr = Es[np.argmin(abs(Es-Ecurr))]
        if verbose:
            # print table with L E
            print("%3d   (%.7f, %.7f)   (%.7f, %.7f)" % 
                  (i, z_curr.real, z_curr.imag, Ecurr.real, Ecurr.imag))
        # check convergence
        if abs(delta_z) > delta_0:
            break
        if abs(delta_z) < ztol and delta_E < Etol:
            converged = True
            break

    Es, ders = E_and_Ep(z_curr, A, B, C)
    iroot = np.argmin(ders)
    return converged, z_curr, Es[iroot], ders[iroot]
