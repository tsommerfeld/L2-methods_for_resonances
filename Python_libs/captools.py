import numpy as np
import matplotlib.pyplot as plt


"""
aux functions for trajectory analysis:

"""

def naive_derivative(xs, ys):
    """ naive forward or backward derivative """
    return (ys[1]-ys[0])/(xs[1]-xs[0])

def central_derivative(xs, ys):
    """ central derivative at x[1] """
    return (ys[2]-ys[0])/(xs[2]-xs[0])

def five_point_derivative(xs, ys):
    """ 
    five-point derivative at x[2] 
    (-ys[4] + 8*ys[3] - 8*ys[1] + ys[0])/(12*h)  
    """
    return (-ys[4] + 8*ys[3] - 8*ys[1] + ys[0])/(xs[4]-xs[0])/3

def simple_traj_der(xs, ys):
    """
    y(x) is a real or complex trajectory of the real variable x
    compute numerical derivatives of its speed dy/dx
    
    this is appropriate for complex scaling E(theta)
    don't use for CAPs as it assumes equal step length
    """
    n_eta = len(xs)
    ders = np.zeros(n_eta, complex)
    ders[0] = naive_derivative(xs[0:2], ys[0:2])
    ders[1] = central_derivative(xs[0:3], ys[0:3])
    for k in range(2,n_eta-2):
        ders[k] = five_point_derivative(xs[k-2:k+3], ys[k-2:k+3])
    ders[-2] = naive_derivative(xs[-3:], ys[-3:])
    ders[-1] = naive_derivative(xs[-2:], ys[-2:])
    return ders  

def trajectory_derivatives(etas, es):
    """
    given an eta-trajectory es(etas)
    compute the first derivative and its absolute value
    as well as the second derivatives
    these are logarithmic derivatives: dE/dln(eta)
    returns:
        corrs: corrected trajectory, corr = E - dE/dln(eta)
        absd1, absd2: absolute values for 1st and 2nd derivative

    careful, etas are expected to be exactly on a log-scale
    """
    n_eta = len(etas)
    lnetas = np.log(etas)
    ders=np.zeros(n_eta, complex)
    corrs=np.zeros(n_eta, complex)
    absd2=np.zeros(n_eta)
    
  
    ders[0] = naive_derivative(lnetas[0:2], es[0:2])
    ders[1] = central_derivative(lnetas[0:3], es[0:3])
    for k in range(2,n_eta-2):
        ders[k] = five_point_derivative(lnetas[k-2:k+3], es[k-2:k+3])
    ders[-2] = central_derivative(lnetas[-3:], es[-3:])
    ders[-1] = naive_derivative(lnetas[-2:], es[-2:])

    corrs = es - ders
    
    
    for i in range(2,n_eta-2):
        absd2[i] = np.abs( etas[i]*(corrs[i+1]-corrs[i-1])/(etas[i+1]-etas[i-1]) )
        #print i, etas[i], es[i], ders[i]
    
    #corrs[0] = es[0] - ders[0]
    #corrs[-1] = es[-1] - ders[-1]
    absd2[0]=absd2[1]=absd2[2]
    absd2[-1]=absd2[-2]=absd2[-3]
    absd1=np.abs(ders)
    return corrs, absd1, absd2


def trajectory_derivatives_old(etas, es):
    """
    given an eta-trajectory es(etas)
    compute the first derivative and its absolute value
    as well as the second derivatives
    these are logarithmic derivatives: dE/dln(eta)
    returns:
        corrs: corrected trajectory, corr = E - dE/dln(eta)
        absd1, absd2: absolute values for 1st and 2nd derivative
    """
    n_eta = len(etas)
    ders=np.zeros(n_eta, complex)
    corrs=np.zeros(n_eta, complex)
    absd2=np.zeros(n_eta)
    
    for i in range(1,n_eta-1):
        ders[i] = etas[i]*(es[i+1]-es[i-1])/(etas[i+1]-etas[i-1])
        corrs[i] = es[i] - ders[i]
        #print i, etas[i], es[i], ders[i]
        
    for i in range(2,n_eta-2):
        absd2[i] = np.abs( etas[i]*(corrs[i+1]-corrs[i-1])/(etas[i+1]-etas[i-1]) )
        #print i, etas[i], es[i], ders[i]
    
    ders[0]=ders[1]
    ders[-1]=ders[-2]
    corrs[0] = es[0] - ders[0]
    corrs[-1] = es[-1] - ders[-1]
    absd2[0]=absd2[1]=absd2[2]
    absd2[-1]=absd2[-2]=absd2[-3]
    absd1=np.abs(ders)
    return corrs, absd1, absd2



def JZBEK_plot(etas, es):
    """
    Jagau, Zuev, Bravaya, Epifanovsky, Krylov
    JCPL 5, 310, 2014
    plot Re(Es) vs eta for Er
    plot Im(Es) vs eta for Ei
    Input:
        etas:   at which the complex trajectory has been computed
        E_traj: complex trajectory from following the resonance
        
        Needs additions:
            4 plots showing Re(E), dRe(E)/dln(eta), Im(E), dIm(E)/dln(eta) 
            print one Er and one Ei list instead of combined lists
            instead of min/max of Re(E)/Im(E) search derivatives
    """
    fig, ((ax1, ax2)) = plt.subplots(2, 1)
    ax1.semilogx(etas, es.real, marker='.', markersize=8, color='blue', linestyle='-')
    ax1.set(ylabel='$Re(E)$')
    ax2.semilogx(etas, es.imag, marker='.', markersize=8, color='blue', linestyle='-')
    ax2.set(ylabel='$Im(E)$')
    fig.tight_layout()
    plt.show()
    return
    
def find_local_minima(etas, es):
    """
    input:  a real array es  energies(eta)
    output: all local minima
    """
    n = len(es)
    out = []
    for i in range(1,n-1):
        if es[i] < es[i-1] and es[i] < es[i+1]:
            out.append((etas[i],es[i]))
    return out

def find_local_maxima(etas, es):
    """
    input:  a real array es  energies(eta)
    output: all local maxima
    """
    n = len(es)
    out = []
    for i in range(1,n-1):
        if es[i] > es[i-1] and es[i] > es[i+1]:
            out.append((etas[i],es[i]))
    return out
