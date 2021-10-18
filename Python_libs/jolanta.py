import numpy as np

"""
Functions defining and evaluating the Jolanta model potential

used for plotting and DVR

"""


def Jolanta_1D(x, a=0.2, b=0.0, c=0.14):
    """
    default 1D potential:
    bound state:  -12.26336 eV
    resonance: (3.279526396 - 0.2079713j)  eV
    """
    return (a * x * x - b) * np.exp(-c * x * x)


def Jolanta_1Db(x, param):
    """
    c.f. Jolanta_1D
    """
    a, b, c = param
    return (a * x * x - b) * np.exp(-c * x * x)


def Jolanta_3D(r, param, l=1, mu=1):
    """
    standard 1D Jolanta potential in radial form
    plus angular momentum potential
    param=(0.028, 1.0, 0.028), l=1, and mu=1 gives
    Ebound in eV = -7.17051, and
    Eres in eV = (3.1729420714-0.160845j)
    """
    a, b, c = param
    return (a * r**2 - b) * np.exp(-c * r**2) + 0.5 * l * (l + 1) / r**2 / mu


def Jolanta_3D_old(r, a=0.1, b=1.2, c=0.1, l=1, as_parts=False):
    """
    default 3D potential; has a resonance at 1.75 eV - 0.2i eV
    use for DVRs
    """
    if as_parts:
        Va = a * r**2 * np.exp(-c * r**2)
        Vb = b * np.exp(-c * r**2)
        Vc = 0.5 * l * (l + 1) / r**2
        return (Va, Vb, Vc)
    else:
        return (a * r**2 - b) * np.exp(-c * r**2) + 0.5 * l * (l + 1) / r**2
