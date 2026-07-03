# Import
import sys
sys.path.insert(0, '../src')

import numpy as np
import sympy as sp

from scipy.special import genlaguerre
from scipy.special import lpmv

from basis import mu_const, spher_const


# Phi(r): the radial part of the test functions
def Phi(l, k, r):
    """
    radial part of the test functions
    """
    result  = 0

    # Parameters for the Laguerre
    x       = r**2
    n       = k
    alpha   = l + 1/2

    result = genlaguerre(n, alpha)(x)*(r**(l))

    return result

# Legendre polynomial
def Leg(m, l, t):
    """
    Computes the associated Legendre polynomial evaluated
    at x = cos(theta)
    """
    result = 0

    x = np.cos(t)
    result = lpmv(np.abs(m), l, x)

    return result 

# The part of the spherical harmonic that depend on phi (the azimuth)
def azimuth(m, p):
    """
    the part of the spherical harmonic that depends on phi.
    It should handle all possible cases wrt m
    """
    if m > 0:
        return np.cos(m*p)
    elif m == 0:
        return 1
    else:
        return np.sin(np.abs(m)*p)

# Test function (without the exponential weight)
def test(k, l, m, r, theta ,phi):
    """
    The test function
    """
    result = azimuth(m, phi)            # azimuth:      phi
    # print('azimuth (phi): ', azimuth(m, phi))

    result = result*Leg(m, l, theta)    # Legendre:     theta
    # print('Legendre (theta): ', Leg(m, l, theta))

    result = result*Phi(l, k, r)        # Phi:          r
    # print('radial: ', Phi(l, k, r))

    # print(' -> without constant: ', result) # before multiplying by constant
    
    result = result*spher_const(l, m)   # Constant      

    '''
    here we may or may not include the following constant,
    so that we can use it for the test functions as well
    '''
    result = result*mu_const(k, l) # the other constant

    return result

# symbolic test function (without the exponential weight)
def sym_test(k, l, m, rad, the, phi):
    result = 0

    # symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')

    # alpha
    a = l + 1/2

    # Spherical harmonic
    sphr = sp.simplify(sp.assoc_legendre(l,abs(m), sp.cos(t)))
    sphr = sp.refine(sphr, sp.Q.positive(sp.sin(t)))

    if m >= 0:
        sphr = sphr*sp.cos(m*p)
    else:
        sphr = sphr*sp.sin(abs(m)*p)

    # Radial part
    radial = 1
    if k > 0:
        radial = sp.assoc_laguerre(k, a, r**2)
    radial = radial*r**l

    # the test function
    f = sphr*radial

    # print the test function
    print("test function: ", f)

    # evaluation at point
    point = {"r":rad,"t":the,"p":phi}
    result = f.subs(point)

    # multiply by constant 
    result = result*spher_const(l, m)
    
    # return
    return result

# The main function
def main():
    # Parameters
    k = 1
    l = 1
    m = 1
    
    # Coefficients
    r = 5
    theta = np.pi/5
    phi = np.pi/3.4

    # printing
    print('test function: ', test(k, l, m, r, theta, phi))
    print()
    print('symbolic test function: ', sym_test(k, l, m, r, theta, phi))

# Main function
if __name__ == "__main__":
    main()
