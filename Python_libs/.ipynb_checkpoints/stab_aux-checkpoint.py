""" 

functions needed for the GPA analysis of stabilization graphs 

convention: 
 
  the scaling variable is called L
  L could be a box-size or a scaling factor of exponents
  however, I have the suspicion is works only, 
  if s = 1/L^2 as in L = boxlength is used


"""



import numpy as np

def dEdL(E, L, P, Q, R):
    """
    we know: E^2*P + E*Q + P = 0
    therefore:
    dEdL = E' = -(E^2*P' + E*Q' + R')/(2*E*P + Q)
    input:
      P, Q, R: three polynomials that depend on L
      E: the energy
      L: the independent (scaling) variable
    output:
      dE/dL derivative of E
    """
    Pp = P.deriv(1)(L)
    Qp = Q.deriv(1)(L)
    Rp = R.deriv(1)(L)
    return -(E**2*Pp + E*Qp + Rp) / (2*E*P(L) + Q(L))

def E_from_L(L, A, B, C):
    """
    given L, solve E^2*A + E*B + C = 0
    return both roots
    """
    P = np.poly1d([A(L), B(L), C(L)])
    return P.roots

def E_and_Ep(L, A, B, C):
    """
    combination of the two functions above
    given L, first solve E^2*A + E*B + C = 0
    for every root found, compute the 1st derivative |dEdL|
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


