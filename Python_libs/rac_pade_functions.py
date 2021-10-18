"""

- evalute the Pade approximants     pade_xy, xy = 21, 31, ... 

"""

import numpy as np
#from scipy.optimize import minimize
#from scipy.optimize import basinhopping
#import matplotlib.pyplot as plt
#from pandas import DataFrame



def res_ene(alpha, beta):
    """ resonance energy from alpha and beta """
    Er = beta**2 - alpha**4
    G = 4*alpha**2 * abs(beta)
    return Er, G

def guess(Er, G):
    """ guess for alpha and beta from Eres """
    ag=0.5*np.sqrt(2.0)*(-2*Er + np.sqrt(4*Er**2 + G**2))**0.25
    bg=0.5*G/np.sqrt(-2*Er + np.sqrt(4*Er**2 + G**2))
    return [ag, bg]



def pade_21(k, params):
    """ Pade [2.1] eq.(2) """
    l = params[0]
    a = params[1]
    b = params[2]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    return l*(k*k + aak2 + a4b2) / (a4b2 + aak2)
    
def pade_31(k, params):
    """ 
    Pade [3,1] eq.(9) corrected by missing factor 2 in a^2k terms 
    """
    l = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2
    return l * num / den 

def pade_32(k, params):
    """ 
    Pade [3,2] as given in 2016 correction paper
    """
    l = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    e = params[4]
    a4b2=a*a*a*a + b*b
    aak2=a*a*k*2
    ddk=d*d*k
    num = (k*k + aak2 + a4b2) * (1 + ddk)
    den = a4b2 + aak2 + ddk*a4b2 + e*e*k*k
    return l * num / den 

def pade_42(k, params):
    """ 
    Pade [4,2] in eqs.(10) and (11) has several typos 
    this is now the correct formula from the 2016 paper
    old arguments: pade_42(k, l, a, b, g, d, w) 
    """
    l = params[0]
    a = params[1]
    b = params[2]
    g = params[3]
    d = params[4]
    w = params[5]
    a4b2=a*a*a*a + b*b
    g4d2=g*g*g*g + d*d
    ta2=2*a*a
    tg2=2*g*g
    k2=k*k
    mu2 = ta2*g4d2  + tg2*a4b2
    num = (k2 + ta2*k + a4b2) * (k2 + tg2*k + g4d2)
    den = a4b2 * g4d2 + mu2*k + w*w*k2
    return l * num / den 


