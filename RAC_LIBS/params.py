"""
Class for RAC parameters: lambda0, alpha, beta, ...

- provide start parameters
- store optimized sets

- start parameters can be from a previous run lsq_with_bounds(), say,
  or from a mix, lower Pade + ad hoc, or entirely ad hoc

Created on Fri Jun 26 10:08:08 2020
@author: thomas
"""

import sys
from random import uniform
import rac_aux as racx
from numpy import sqrt

class Params:

    """ available Pade models as nm = Pade-[n,m] """
    nms = ['21', '31', '32', '41', '42', '43', '52', '53']

    """ ad hoc guess related """
    delta0 = 0.8   # ad hoc guess for (1-delta**2*kappa) terms
    q20 = 5        # ad hoc guess for c**2*kappa**2 denominator terms
    q30 = 1        # ad hoc guess for d**2*kappa**3 denominator terms

    def __init__(self, E0, lambda0):
        """
        E0 is the estimated Er used to create a guess for alpha and beta
        0.7 * linear extrapolation is reasonable
        """
        self.E0 = E0
        self.l0 = lambda0
        self.G0 = 0.2*E0
        self.popts = {}   # optimized parameters
        self.pah = {}     # ad hoc parameters
        self.addhoc_ps()

    def addhoc_ps(self):
        """ initilize pah with ad hoc parameters for RAC-nm """
        p21 = [self.l0] + racx.guess(self.E0, self.G0)
        self.pah['21'] = p21
        a, b = racx.guess(5*self.E0,10*self.G0)
        d0 = 1/sqrt(a**4 + b**2)
        g0 = sqrt(2.0)*a*d0
        self.pah['gd'] = [d0, g0]  # ad hoc for (1+g**2*k+d**2*k**2)

    def put(self, nm, plist):
        """ add optimized parameters to self.popts dictionary """
        self.popts[nm] = plist

    def start(self, nm, adhoc=False, noise=0):
        """
        generate a start parameter set for RAC-nm
        this is based as much as possible on existing optimized sets
        fallback are the ad hoc parameters
        adhoc=True does not use any previous Pade optimizations
        noise is a relative level (0.05 = 5%) of random noise
        that can be applied after creating the parameter set
        """
        p21 = self.pah['21']
        gd0 = self.pah['gd']
        d0 = Params.delta0
        q20 = Params.q20
        q30 = Params.q30

        if not adhoc:
            if nm in self.popts:
                return self.popts[nm]

            if '21' in self.popts:
                p21 = self.popts['21']
            if '31' in self.popts:
                d0 = self.popts['31'][3]
            if '42' in self.popts:
                gd0 = self.popts['42'][3:5]
            q2s = []
            if '32' in self.popts:
                q2s.append(self.popts['32'][-1])
            if '42' in self.popts:
                q2s.append(self.popts['42'][-1])
            if len(q2s) > 0:
                q20 = sum(q2s)/len(q2s)

        if nm == '21':
            ps = p21
        elif nm == '31':
            ps = p21 + [d0]
        elif nm == '32':
            ps = p21 + [d0, q20]
        elif nm == '41':
            ps = p21 + gd0
        elif nm == '42':
            ps = p21 + gd0 + [q20]
        elif nm == '43':
            ps = p21 + gd0 + [q20, q30]
        elif nm == '52':
            ps = p21 + gd0 + [d0, q20]
        elif nm == '53':
            ps = p21 + gd0 + [d0, q20, q30]
        else:
           sys.exit('Params.start: nm='+str(nm)+' This should not happen.')

        """ adds random noise to each value in ps """
        if noise > 0:
            for i in range(len(ps)):
                ps[i] *= 1 + uniform(-1,1)*noise

        return ps




class Results:
    """ a simple "struct" for three dictonaries Er, G, Chi2 """

    def __init__(self):
        """ start with empty dictionaries """
        self.Er = {}
        self.G = {}
        self.Chi2 = {}

    def put(self, nm, Er, G, Chi2):
        """ one set of results """
        self.Er[nm] = Er
        self.G[nm] = G
        self.Chi2[nm] = Chi2

    def have(self, nm):
        """ check for availability of results """
        return nm in self.Er

    def get(self, nm):
        if nm in self.Er:
             return self.Er[nm], self.G[nm], self.Chi2[nm]
        else:
             print('Warning: %s not in Results' % nm)
             print('Maybe check with .have before calling get')
             return 0 # may cause an error

    def print_nice_table(self, msg):
        """ 
        print a nicely formatted table that
        lists the available results and a message
        """
        hline = '-------------------------------------------------'
        blancs= '                                                 '     
        print(hline)
        c = (len(blancs) - len(msg))//2
        print(blancs[:c], msg)
        print("R-ACCC       Er       Gamma           chi^2      ")
        print(hline)
        for nm in self.Er.keys():
            print("Pade-{0:s}  {1:8.4f}   {2:8.4f}     {3:11.2e}"\
                  .format(nm, self.Er[nm], self.G[nm], self.Chi2[nm]))
        print(hline)
    
        """
        if verbose > 500:
            This is printing BS; check formulas 
            or better solve g,d eq independently
            
            print('\nSecond resonances')
            for nm in ['41', '42', '43', '52', '53']:
                if nm in popts:
                    ps = popts[nm]
                    g, d = ps[3:5]
                    Er, G = racx.res_ene_gd(ps[3], ps[4])
                    print("Pade-%s  %8.4f  %8.4f" % (nm, Er, G))
        """




