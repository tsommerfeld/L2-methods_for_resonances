"""

We minimize chi2 = 1/M sum_i (rac(k_i) - lambda_i)**2
for this purpose least_suares is superior to all local minimize() variations 


three different ways to fit:
    - simple least squares
    - least squares with bounds 
    - basin-hopping with least squares as local minimizer


least-squares-with-bounds applies limits on the parameters
- all parameters are > 1e-7
- alpha, beta, lambda0 do not change more than 10% from their 21 values




basin hopping related functions of the rac code are in bh_with_lsq

basin_hopping needs 
(a) a function returning chi2 that will be called directily: bh_chi2()
(b) a local minimizer, which is supposed to minimize chi2, but does not
    get bh_chi2() as a parameter

This behavior is designed around scipy.optimize.minimize, yet, to use
scipy.optimize.least_squares(), which computes chi2 internally, a slight
modification is needed: 
(a) basin_hopping gets its own chi2 function to compute chi2.
(b) The local minimizer (least_squares()) gets as an argument the rac-function 
and rac-jacobian it neededs. These functions never evaluate chi2 explicitly, 
but only the terms in the sum. 

"""

import numpy as np
import rac_aux as racx
#import sys


from scipy.optimize import least_squares
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from pandas import DataFrame

def simple_lsq(fun, jac, p0s, rac_args, nm, verbose=1):
    """ straightforward call to least_squares plus convergence check """
    res = least_squares(fun, p0s, method='trf', jac=jac, args=rac_args, 
                        max_nfev=100*len(p0s)**2)
    if not res.success:
        print('  **** Pade-%s failed to converged ****' % (nm))
        verbose = 4
    if not res.success or verbose > 0:
        print("  x:", abs(res.x))
        if not res.success or verbose > 2:
            print("  message:",res.message)
            print("  success:",res.success)
            print("  njev:",res.njev)
            print("  cost:",res.cost)
            print("  grad:",res.grad)
            print("  Er=%f,  Gamma=%f" % racx.res_ene(res.x[1], res.x[2]))
    return res



def lsq_with_bounds(fun, jac, p0s, rac_args, bnds_on_lab=False, 
                    gt=1e-7, verbose=1):
    """ 
    least_squares with bounds:
        fun, jac: the function to be minimized and its Jabobian
        p0s : start parameters: lambda0, alpha, beta, ...
        rac_args: additional arguments for fun (ls, ks, k2s, ...)
        bnds_on_lab: bounds on the 1st three parameters, lambda0, alpha, beta
        lambda0 plus/minus 5%
        alpha and beta *1.5 and /1.5 
        all other parameters > gt
        only used if len(p0s) > 3
    calls scipy.least_squares()
    returns the optimization result res
    """
    n_para = len(p0s)
    lower = np.full(n_para, 1e-7)
    upper = np.full(n_para, np.inf)
    bnds=(lower, upper)
    if len(p0s) > 3 and bnds_on_lab:
        lower[0], upper[0] = p0s[0]*0.95, p0s[0]*1.05 
        lower[1], upper[1] = p0s[1]/1.5,  p0s[1]*1.5
        lower[2], upper[2] = p0s[2]/1.5,  p0s[2]*1.5
    
    res = least_squares(fun, p0s, method='trf', jac=jac, args=rac_args, 
                        bounds=bnds, max_nfev=300*len(p0s)**2)
    if not res.success or verbose > 2:
        print("  message:",res.message)
        print("  success:",res.success)
        print("  njev:",res.njev)
        print("  cost:",res.cost)
        print("  grad:",res.grad)
    if not res.success or verbose > 0:
        print("  Er=%f,  Gamma=%f" % racx.res_ene(res.x[1], res.x[2]))
        print("  x:", abs(res.x))

    return res




def bh_with_lsq(p0s, n_bh, args, sane_bnds, T=1e-4, verbose=1):
    """
    p0s: start parameters
    n_bh: number of basin hopping steps
    args: (ks, k2s, ls, function_lsq, jacobian_lsq)
    sane_bnds = (Emin, Emax, Gmin, Gmax) 
      sensibility filter for identified solutions, say, 
      RAC-21 plus-minus 50%
    T: temperature for Monte Carlo-like hopping decision

    This is quite slow. Do callback w/o lists?

    """
    jbh = 0
    chi2s = np.zeros(n_bh)
    alphas = np.zeros(n_bh)
    betas = np.zeros(n_bh)
    ps = []
    #ps = np.zeros((n_bh, len(p0s)))
    if verbose > 0:
        print('  Doing %d basin hops.' % n_bh)

    def bh_chi2(params, args=()):
        """
        we need two chi2-returning functions
        one called by basin_hopping() to evaluate a point=params
        one called by the local minimizer
        this is the former
        at the moment   'args':(ks, k2s, ls, f_lsq, j_lsq)
        """
        (ks, k2s, ls, sigmas, f_lsq, j_lsq) = args
        diffs = f_lsq(params, ks, k2s, ls, sigmas)
        return np.sum(np.square(diffs))

    def lsq_wrapper(fun, x0, args=(), method=None, jac=None, hess=None,
                hessp=None, bounds=None, constraints=(), tol=None,
                callback=None, options=None):
        """
        function called by basin_hopping() for local minimization
        returns a Results object
        """
        (ks, k2s, ls, sigmas, f_lsq, j_lsq) = args
        res = least_squares(f_lsq, x0, method='trf', jac=j_lsq, 
                            args=(ks, k2s, ls, sigmas))
        res.fun = res.cost*2
        #if not res.success:
        #    print('wrapper:', res.success)
        #    print(res.message)
        #    print(res.fun, res.x)
        #delattr(res, 'njev')
        return res

    def bh_call_back(x, f, accepted):
        """
        called after every local minimization
        create lists with partial results 
        """
        nonlocal jbh, chi2s, alphas, betas
        chi2s[jbh] = f
        alphas[jbh], betas[jbh] = x[1], x[2]
        ps.append(x)
        #ps[jbh,:] = x
        jbh += 1

    """ Call the minimizer """
    min_kwargs = {'method':lsq_wrapper, 'args':args, 'jac':True} 

    res = basinhopping(bh_chi2, p0s, minimizer_kwargs=min_kwargs, niter=n_bh, 
                   T=T, seed=1, callback=bh_call_back)

    if 'successfully' not in res.message[0] or verbose > 0:
        print("  minimization failures:",res.minimization_failures)
        if 'successfully' not in res.message[0] or verbose > 1:
            print("  message:",res.message)
            print("  x:", abs(res.x))
            print("  nfev:",res.nfev)                
            print("  njev:",res.njev)
            print("  nit:",res.nit)
            print("  Er=%f,  Gamma=%f" % racx.res_ene(res.x[1], res.x[2]))


    """ Process the results stored by bh_call_back """
    Er, G = racx.res_ene(res.x[1], res.x[2])
    chi2 = res.lowest_optimization_result.cost*2
    best = (Er, G, chi2)
    sane, j_sane, df = process_minima(chi2s, alphas, betas, sane_bnds, verbose)
    
    if verbose > 0:
        print('  Best:  %f  %f  %.4e' % best)
        print('  Sane:  %f  %f  %.4e' % sane)

    if verbose > 2:
        plot_chis(df)
    if verbose > 4:
        plot_map(df)
    
    # this parameter set may be rubbish if no sane minimum has been found
    # since ps[j_sane] will be recycled as start parameters
    # this will lead to trouble down the line
    # so we need to test and a default p0s independent of previous nms
    return best, sane, ps[j_sane], df 
    #return best, sane, ps[j_sane,:], df

def process_minima(chis, alphas, betas, bounds, digits=5, verbose=1):
    """
    process the minima visited by basin_hopping()
    - compute Er and Gamma and put into a DataFrame
    - sane DataFrame by filtering the minima by 
      bounds = (Emin, Emax, Gmin, Gmax)
    - find the lowest chi2 and its original index (j_sane)
    - find unique values by: 
        rounding to digits 
        combining Er and G to a string
        using .unique
    
    returns 
    - the best sane energy (Er, G, chi2)
    - the index of this minimum, j_sane
    - the DataFrame with unique sane minima

    """
    Emin, Emax, Gmin, Gmax = bounds
    Ers, Gms = racx.res_ene(alphas, betas)
    df = DataFrame({'chis':chis, 'Er':Ers, 'G':Gms})
    fltr = ((df.Er>Emin) & (df.Er<Emax) & (df.G>Gmin) & (df.G<Gmax)) 
    df_sane = df[fltr].copy()
    
    n = df_sane.shape[0]
    if verbose > 0 or n < 1:
        print('  %d sane minima found' % (n))
        if n < 1:
            return (0, 0, 0), 0, df 
    df_sane.sort_values('chis', inplace=True)
    j_sane = df_sane.chis.idxmin()
    sane = (df['Er'][j_sane], df['G'][j_sane], df['chis'][j_sane])
    df_sane['logs'] = np.log10(df_sane['chis']) 
    #df=sane['Unique'] = str(np.round["Er"]) + "," + str(np.round["G"])
    return sane, j_sane, df_sane


def plot_chis(df, nbins=200):
    """ 
    show a histogram of the log10(chi2s) 
    """
    r, c = df.shape
    n = min(r//4, nbins)
    plt.cla()
    pop, edges, patches = plt.hist(df.logs, bins=n)
    plt.xlabel('$\log \chi^2$', fontsize=10)
    plt.ylabel('number of minima', fontsize=10)    
    plt.show()
    print(pop)
    
def plot_map(df, ymax=1.5):
    """ plot complex energy plane and color code with chi2 """
    plt.cla()
    plt.scatter(df.Er.values, df.G.values, c=df.logs, s=20, cmap='viridis')
    plt.ylim(0, ymax)
    #cb = plt.colorbar()
    plt.tick_params(labelsize=12)
    plt.xlabel('$E_r$ [eV]', fontsize=10)
    plt.ylabel('$\Gamma$ [eV]', fontsize=10)    
    plt.show(block=True)

