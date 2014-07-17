import pandas as pd
import numpy as np
from scipy.stats import gmean
from pyDOE import lhs
import random as rand
import math

def stochastic_step(m=1, n=1, scale=1, type='cauchy'):
    """Takes a step along a stochastic walk defined by type.

    (Yang-Deb Levy not yet)
    Cauchy
    Gaussian
    Uniform"""
    if type.lower() == 'cauchy':
        return levy_flight(scale=scale, m=m, n=n)
    elif type.lower() == 'gaussian':
        return levy_flight(scale=scale, m=m, n=n, exponent=2)
    elif type.lower() == 'yangdeb':
        return yangdeb_flight(m=m, n=n) * scale
    elif type.lower() == 'uniform':
        return np.random.rand(m, n) * scale
    #elif type.lower() == 'poissondisc':
        # Unimplemented ATM
    else:
        raise ValueError("Invalid walk type.")

def yangdeb_flight(m=1, n=1):
    beta = 3/2
    sigma = ((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / 
        (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** \
        (1 / beta)
    u = np.random.normal(size=[m, n]) * sigma
    v = np.random.normal(size=[m, n])
    return u / abs(v) ** (1 / beta)


def levy_flight(exponent=1, skewness=0, scale=1, location=0, 
    type='isotropic', m=1, n=1):
    """ Generates m-by-n levy flight. Rows are single (not cumulative) steps.

    Note that this can be used for m different levy flights, or m steps on the
    same flight (provided you add the steps together cumulatively).

    exponent -> 2 for gaussian, 1 for cauchian
    sdev is stdev
    beta, delta are symmetry parameter (zero drift = 0)
    type is 'isotropic' or 'axis'
    """

    seed = CMS_rand(exponent, skewness, m=m, n=n)

    if type.lower() == 'isotropic':
        seed = unitize(seed)
    elif type.lower() == 'axis':
        pass
    else:
        raise ValueError("Invalid flight type.")

    return seed * scale + location

def CMS_rand(exponent=1, skewness=0, scale=1, location=0, m=1, n=1):
    """ Generates m x n CMS stable random number as an numpy array.

    Returns
    -------

    If m and n are defined, m x n array
    If m xor n are defined, m or n array
    If neither, float

    Refs from 
    + J.H. McCulloch, "On the parametrization of the afocal stable 
    distributions," Bull. London Math. Soc. 28 (1996): 651-55, 
    + V.M. Zolotarev, "One Dimensional Stable Laws," Amer. Math. Soc., 1986.
    + G. Samorodnitsky and M.S. Taqqu, "Stable Non-Gaussian Random Processes,"
     Chapman & Hill, 1994.
    + A. Janicki and A. Weron, "Simulaton and Chaotic Behavior of Alpha-Stable 
    Stochastic Processes," Dekker, 1994.
    + J.H. McCulloch, "Financial Applications of Stable Distributons," 
    Handbook of Statistics, Vol. 14, 1997.

    Code origins:
    + CMS algorithm (J.M. Chambers, C.L. Mallows and B.W. Stuck)
    + MATLAB by J. Huston McCulloch, Ohio State University Econ. Dept. 
    (mcculloch.2@osu.edu).
    + Python by V. Gauthier, Telecom SudParis, CNRS Lab 
    (vgauthier@luxbulb.org)
    + Python (rewrite) by N Badger, nickbadger.com

    Some kind of copyleft is going to be added here when I think about it next
    """

    # Traps
    if exponent <= 0 or exponent > 2 :
        raise ValueError('Characteristic exponent must be in (.1,2)')

    if np.abs(skewness) > 1 :
        raise ValueError('Skewness must be in (-1,1).')

    if n < 0 or m < 0 :
        raise ValueError('n and m must be positive.')

    # Generate source random numbers in dimensions commiserate to input
    if (m == 1) and (n == 1):
        w = -np.log(np.random.rand())
        phi = (np.random.rand() - 0.5) * np.pi
    else:
        w = -np.log(np.random.rand(m,n)).squeeze()
        phi = ((np.random.rand(m,n) - 0.5) * np.pi).squeeze()


    # This is a huge clusterfuck and should be entirely reworked via 
    # http://www.sjsu.edu/faculty/watkins/stablerand.htm


    # Gaussian case (Box-Muller) short-circuit:
    if exponent == 2:
        x = 2 * np.sqrt(w) * np.sin(phi)
    # else: 

    #     if exponent == 1:
    #         skewprime = skewness
    #     else:
    #         skewprime = -np.tan(.5 * np.pi * (1 - exponent)) * \
    #             np.tan(exponent)

    # Symmetrical cases:
    elif skewness == 0:
        if exponent == 1:   # Cauchy case
            x = np.tan(phi)
        else:
            x = ((np.cos((1-exponent)*phi) / w) ** (1/exponent - 1) * \
                np.sin(exponent * phi) / np.cos(phi) ** (1/exponent))
    # General cases:
    else:
        cosphi = np.cos(phi)
        if np.abs(exponent-1) > 1.e-8:
            zeta = skewness * np.tan(np.pi*exponent/2)
            aphi = exponent * phi
            a1phi = (1 - exponent) * phi
            x = ((np.sin(aphi) + zeta * np.cos(aphi)) / cosphi) * \
                ((np.cos(a1phi) + zeta * np.sin(a1phi)) / (w * cosphi)) ** \
                ((1-exponent)/exponent)
        else:
            bphi = (np.pi/2) + skewness * phi
            x = (2/np.pi) * \
                (bphi * np.tan(phi) - skewness * \
                np.log((np.pi/2) * w * cosphi / bphi))
            if exponent != 1:
                x = x + skewness * np.tan(np.pi * exponent/2)

    # Shift, scale, return
    return location + scale * x

def lhs_scaled(var_min, var_max, n):
    """ Latin Hypercube Sampling of variables defined by vardef
    vardef(2, len(N)) <max value, min value>
    N -> number of samples to generate
    LHC.m
    """

    # Get number of variables and their scaling factor
    k = len(var_min)
    spread = var_max - var_min

    # Get linear hypercube sample and scale it, then return
    return var_min + spread * lhs(k, n, criterion='maximin', iterations=100)