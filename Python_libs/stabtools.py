#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:39:25 2021

@author: thomas

Functions for the selection of a plateau or crossing range

"""

# import sys
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize_scalar, curve_fit, least_squares


def find_zero(xs, j0, up=True):
    """
    Find the next "zero" (sign inversion) along xs
    starting at index j0
    up==True searches upwards
    """
    step = -1
    stop = 1
    if up:
        step = 1
        stop = len(xs) - 1
    for j in range(j0, stop, step):
        x1 = xs[j]
        x2 = xs[j + step]
        if x1 * x2 < 0:
            if np.abs(x1) < np.abs(x2):
                return j
            return j + step
    return stop


def min_der(xs, ys, smooth=0):
    """
    Find the minimum (abs) of the derivative along a curve, ys(xs)
    derivative are computed using a spline interpolation

    xs : pointwise curve, x values
    ys : pointwise curve, y values

    Returns:
        jmin: index with the lowest derivative
        xs[jmin]
        ders[jmin]
    """
    sp = interpolate.splrep(xs, ys, s=smooth)
    ders = interpolate.splev(xs, sp, der=1)
    jmin = np.argmin(abs(ders))
    return jmin, xs[jmin], ders[jmin]


def der_and_curvature(xs, ys, smooth=0):
    """
    Derivative and curvature of a pointwise provided curve y(x)
    obtained by standard spline interpolation
    normalized to range [-1,1]
    Any branch of a stabilization graph goes through
    a crossing-plateau-crossing structure, which are
    defined by curavture extrema and a curvature zero inbetween.

    xs : pointwise curve, x values
    ys : pointwise curve, y values

    Returns: the derivatives: dy/dx and d2y/dx2 at the xs
    """
    sp = interpolate.splrep(xs, ys, s=smooth)
    d1s = interpolate.splev(xs, sp, der=1)
    d2s = interpolate.splev(xs, sp, der=2)
    d1s /= np.max(np.abs(d1s))
    d2s /= np.max(np.abs(d2s))
    return d1s, d2s


def crossing(xs, E1, E2, select=0.5, smooth=0):
    """
    find center of a crossing
    select a range of points determined by drop of the
    curvature
    options:
        use drop-offs to max(curature)*select
        use drop-offs of sqrt(max(curvature))

    Parameters:
        xs : scaling parameter E1(x), E2(x)
        E1, E2 : lower and upper branch of the crossing
        select : selection range cutoff as determined by
                 the d2 reduction (0.0 = all to next plateaux)
                 if select < 0, use sqrt(d2_max)
    Returns:
        success : boolean
        xc : center of the crossing
        selected ranges of xs, E1, and E2
    """
    sp1 = interpolate.splrep(xs, E1, s=smooth)
    sp2 = interpolate.splrep(xs, E2, s=smooth)
    d2s1 = interpolate.splev(xs, sp1, der=2)
    d2s2 = interpolate.splev(xs, sp2, der=2)
    j1_mn = np.argmin(d2s1)
    j2_mx = np.argmax(d2s2)
    jc = j1_mn
    if j1_mn - j2_mx > 1:
        if np.abs(xs[j1_mn] - xs[j2_mx]) > 0.05:
            return (False, -1, (j1_mn, j2_mx), d2s1, d2s2)
        else:
            jc = (j1_mn + j2_mx) // 2
    xc = xs[jc]
    d2s1_mn, d2s2_mx = d2s1[j1_mn], d2s2[j2_mx]
    d2s1_cut, d2s2_cut = select * d2s1_mn, select * d2s2_mx
    if select < 0:
        d2s1_cut = -np.sqrt(-d2s1_mn)
        d2s2_cut = np.sqrt(d2s2_mx)
    j_max = find_zero(d2s1 - d2s1_cut, j1_mn, up=True)
    j_min = find_zero(d2s2 - d2s2_cut, j2_mx, up=False)
    x_sel = xs[j_min:j_max + 1]
    E1sel = E1[j_min:j_max + 1]
    E2sel = E2[j_min:j_max + 1]
    return (True, xc, x_sel, E1sel, E2sel)


def plateau(xs, ys, srch_range=(-1, -1), smooth=0):
    """
    find
    - index of minimum of derivative and exact zero of curvature
    - indices and exact positions of extrema of the curvature

    Parameters:
        xs : pointwise curve, x values
        ys : pointwise curve, y values
        srch_range=(xmin, xmax): smaller search range from problems
    Returns:
        j0, j1, j2: indices of zero and extrema of ys
        x0, x1, x2: precise positions of zero and extrema of d^2y/dx^2
        jx = -1 indicates failure
    """
    def der1(x, a, b, c):
        # spline = (a, b, c)
        return interpolate.splev(x, (a, b, c), der=1)

    def der2(x, a, b, c):
        # spline = (a, b, c)
        return interpolate.splev(x, (a, b, c), der=2)

    def mabsder2(x, a, b, c):
        # spline = (a, b, c)
        return -np.abs(interpolate.splev(x, (a, b, c), der=2))

    failure = ((-1, -1, -1), (-1, -1, -1))
    xmin, xmax = srch_range
    # search range, default is xs[2,-2]
    if xmin < 0:
        jmin = 2
        xmin = xs[jmin]
    else:
        jmin = np.argmin(np.abs(xs - xmin))
    if xmax < 0:
        jmax = len(xs) - 2
        xmax = xs[jmax]
    else:
        jmax = np.argmin(np.abs(xs - xmax))

    sp = interpolate.splrep(xs, ys, s=smooth)
    d1s = interpolate.splev(xs, sp, der=1)
    d2s = interpolate.splev(xs, sp, der=2)

    # Find the center x0 and its index j0
    j0 = np.argmin(np.abs(d1s[jmin:jmax])) + jmin
    if j0 == jmin or j0 == jmax:
        print('Failed to find a minimum of 1st derivative in search range.')
        return failure
    res = minimize_scalar(der1, (xmin, xs[j0], xmax), args=sp)
    if res.success:
        x0 = res.x

    # Find extrema of der2 to identify adjenct crossings
    j1 = jmin + np.argmin(d2s[jmin:j0])
    j2 = j0 + np.argmax(d2s[j0:jmax])
    if d2s[j1] * d2s[j2] > 0:
        print('Trouble finding limiting min(der2) or max(der2)')
        return (j0, j1, j2), (x0, -1, -1)
    x1, x2 = -1, -1
    dl, dc, du = np.abs(d2s[j1 - 1:j1 + 2])
    if dc > dl and dc > du:
        xl, xc, xu = xs[j1 - 1:j1 + 2]
        res = minimize_scalar(mabsder2, (xl, xc, xu), args=sp)
        if res.success:
            x1 = res.x
    dl, dc, du = np.abs(d2s[j2 - 1:j2 + 2])
    if dc > dl and dc > du:
        xl, xc, xu = xs[j2 - 1:j2 + 2]
        res = minimize_scalar(mabsder2, (xl, xc, xu), args=sp)
        if res.success:
            x2 = res.x
    return (j0, j1, j2), (x0, x1, x2)


def min_delta(xs, El, Eu):
    """
    Find the minimum energy difference = the crossing
    between two stabilization roots
    This function looks across the whole range, and
    may be less useful as dedicated search up/down

    xs: scaling parameter
    El, Eu: lower and upper branch

    Returns:
        jmin: min distance
        xs[jmin]
        Eu[jmin] - El[jmin]
    """
    diff = Eu - El
    jmin = np.argmin(diff)
    return jmin, xs[jmin], diff[jmin]


def min_delta_search(xs, xp, Ep, Eo, up=True):
    """
    Find the minimum energy difference = the crossing
    between two stabilization roots
    This function starts at xc and searches down in x

    xs: scaling parameter
    xp: center of the plateau of branch Ep
    Eo: other branch  (upper/lower branch for up=False/True)
    up: search direction

    Returns:
        jmin: min distance
        xs[jmin]
        delta-E[jmin]
    """
    jp = np.argmin(abs(xs - xp))
    diff = abs(Ep - Eo)
    step = -1
    end = 0
    if up:
        step = 1
        end = len(xs)
    last_diff = diff[jp]
    jmin = -1
    for j in range(jp + step, end, step):
        curr_diff = diff[j]
        if curr_diff > last_diff:
            jmin = j - step
            break
        last_diff = curr_diff

    if jmin < 0:
        return -1, -1, -1

    return jmin, xs[jmin], diff[jmin]


def carlson_eq_3(z, a0, a1, A, B, z1):
    """
    2-by-2 model for Hamiltonian for crossing
    returns (E-, E+) value for z
    """
    H11 = a0
    H22 = a0 + a1 * (z - z1)
    H12_sq = A + B * (z - z1)**2
    f1 = (H11 + H22) * 0.5
    f2 = 0.5 * ((H11 - H22)**2 + 4 * H12_sq)**0.5
    return np.append(f1 - f2, f1 + f2)


def carlson_eq_3_lsq(params, xdata, ydata):
    '''Returns residuals for eqn_3'''
    a0, a1, A, B, z1 = params
    return carlson_eq_3(xdata, a0, a1, A, B, z1) - ydata


def tbt_ana_curvefit(zs, E1s, E2s, cross_center):
    """
    fits Carlson's 2-by-2 model Hamiltonian
    to the crossing z1, E1, E2
    cross_center is used as a guess (can be replaced by z of closest distance)
    returns the stationary z and energy, and fit quality:
        zr, zi, Er, Ei, chi2
    """
    Es = np.append(E1s, E2s)
    a0 = E2s[0] - (E2s[0] - E1s[-1]) / 2  # resonance H11 is constant
    a1 = (E1s[0] - E2s[-1]) / (zs[0] - zs[-1])  # DC H22 is linear in z
    A = (min(E2s - E1s) / 2)**2  # min distance squared
    B = -1.0  # Guess arbitrary, negative number
    z1 = cross_center
    guess = [a0, a1, A, B, z1]
    params = curve_fit(carlson_eq_3, zs, Es, p0=guess)
    a0, a1, A, B, z1 = params[0]
    # stationary point
    zreal = z1
    zimag = np.sqrt((a1 * A) / (abs(B) * (a1**2 + 4 * B)))
    # resonance energy
    Er = a0
    Ei = -2 * (np.sqrt(A * abs(B))) / (np.sqrt(4 * B + a1**2))
    # chi^2
    Esfit = carlson_eq_3(zs, a0, a1, A, B, z1)
    chi2 = sum((Esfit - Es)**2)
    return zreal, zimag, Er, Ei, chi2


def tbt_ana_lsq(zs, E1s, E2s, cross_center):
    """
    fits Carlson's 2-by-2 model Hamiltonian
    to the crossing z1, E1s, E2s
    cross_center is used as a guess (can be replaced by z of closest distance)
    returns the stationary z and energy, and fit quality:
        zr, zi, Er, Ei, chi2
    """
    Es = np.append(E1s, E2s)
    a0 = E2s[0] - (E2s[0] - E1s[-1]) / 2  # resonance H11 is constant
    a1 = (E1s[0] - E2s[-1]) / (zs[0] - zs[-1])  # DC H22 is linear in z
    A = (min(E2s - E1s) / 2)**2  # min distance squared
    B = -1.0  # Guess arbitrary, negative number
    z1 = cross_center
    guess = [a0, a1, A, B, z1]
    data = (zs, Es)
    res = least_squares(carlson_eq_3_lsq, x0=guess, args=data)
    if not res.success:
        print('Fit failed:', res.message)
    chi2 = res.cost * 2
    a0, a1, A, B, z1 = res.x
    # stationary point
    zreal = z1
    zimag = np.sqrt((a1 * A) / (abs(B) * (a1**2 + 4 * B)))
    # resonance energy
    Er = a0
    Ei = -2 * (np.sqrt(A * abs(B))) / (np.sqrt(4 * B + a1**2))
    return zreal, zimag, Er, Ei, chi2
