import numpy
from scipy import sqrt


""" helper functions using standard Pade approximants E = P/Q """


    
def pade_via_lstsq(np, nq, xs, ys, rcond=1e-14, return_lists=False, ycplx=False):
    """ 
    compute the coeffients of a Pade np/np approximant using numpy.linalg.lstsq
    the lstsq function will take any number of equations, so this should
    work for any len(xs)=len(ys) 
    returned are the Pade coefficients ps and qs in highest-power first order
    """
    N_coef = np + nq + 1
    M_data = len(xs)
    y_type = float
    if ycplx:
        y_type = complex
    A = numpy.zeros((M_data,N_coef), y_type)
    b = numpy.zeros(M_data, y_type) 
    for k_data in range(M_data):
        i_coef = 0
        for ip in range(np,1,-1): # counts from np down to 2
            A[k_data,i_coef] = xs[k_data]**ip
            i_coef += 1
        A[k_data,i_coef] = xs[k_data]
        i_coef += 1
        A[k_data,i_coef] = 1.0
        i_coef += 1
        for iq in range(nq,1,-1): # counts from np down to 2
            A[k_data,i_coef] = -ys[k_data] * xs[k_data]**iq
            i_coef += 1
        A[k_data,i_coef] = -ys[k_data] * xs[k_data]
        b[k_data] = ys[k_data]
    coefs, residual, rank, s = numpy.linalg.lstsq(A,b,rcond=rcond)
    ps = coefs[:np+1]
    qs = numpy.array(list(coefs[np+1:np+nq+1]) + [1])
    if return_lists:
        return ps, qs
    else:
        return numpy.poly1d(ps), numpy.poly1d(qs)
    


def eval_pade_lists(x, ps, qs):
    """ evaluate a standard Pade approximant """
    return numpy.polyval(ps,x)/numpy.polyval(qs,x)


def dEds(s, P, Q):
    """ 
    we have:   E = P/Q 
    therefore: dEds = E' = (P'Q - Q'P) / Q^2
    input
    P, Q: two polynomials that depend on L
    s: the independent (scaling) variable
    """
    Pp = P.deriv(1)(s)
    Qp = Q.deriv(1)(s)
    return (Pp*Q(s) - Qp*P(s)) / Q(s)**2

def EpoEpp(s, P, Q):
    """ 
    we need E'/E'' need for Newton's method
    this expression can be evaluated analytically:
        E'  = (P'Q - Q'P)/Q**2 = N/Q**2
        E'' = (N'Q**2 - 2QQ'N) / Q**4
        
        E'/ E'' = NQ / (N'Q - 2Q'N)
        
        with N' = P''Q - Q''P
    """
    QQ = Q(s)
    PP = P(s)
    Pp = P.deriv(1)(s)
    Qp = Q.deriv(1)(s)
    Ppp = P.deriv(2)(s)
    Qpp = Q.deriv(2)(s)
    N  = Pp*QQ - Qp*PP
    Np = Ppp*QQ - Qpp*PP
    return (N*QQ) / (Np*QQ - 2*Qp*N)
