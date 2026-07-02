# imports
import os
import pickle
import numpy as np

# integration
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

# Quadrature order (single source of truth: used both to build the rule and to
# name the collision-operator files via naming.operator_tag). Choose carefully
# before a full run -- accuracy vs. the 6D cost.
N_LAGUERRE = 9   # radial (generalized-Laguerre) nodes
N_LEBEDEV  = 7   # Lebedev order

'''
Spherical coordinates
'''
# radius    
def radius(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

# theta 
def theta(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    if r == 0:
        return 0
    else:
        return np.arccos(z/r)

# phi    
def phi(x, y):
    r = np.sqrt(x**2 + y**2)

    if r == 0:
        return 0
    elif y == 0:
        return np.arccos(x/r)
    else:
        return np.sign(y)*np.arccos(x/r)

'''
Unpack the quadrature
'''
# Unpacks the quadrature so that integration is easy
def unpack_quad(quad):
    '''
    Given the initial form of how the quadrature is saved,
    this, unpacks and rearranges the quadrature, so that
    it is easy to use.

    Input:
        - quad: 
            the quadrature for p and q, with a specific
            order, and in cartesian coordinates for
            the angular part.

    Output: 
        - the quadrature in the shape:
            [weight, points]
            
            weight is a scalar, points is in spherical 
            coordinates, [r_p, t_p, p_p, r_q, t_q, p_q]
    '''

    # radial quadrature for p
    r_p = quad[0][0]
    w_p = quad[0][1]

    # angular quadrature for p
    ang_p   = quad[1][0]    # point (x, y, z) on the unit sphere
    ang_w_p = quad[1][1]

    # radial quadrature for q
    r_q = quad[2][0]
    w_q = quad[2][1]
    
    # angular quadrature for q
    ang_q   = quad[3][0]    # point (x, y, z) on the unit sphere
    ang_w_q = quad[3][1] 

    # cartesian quadrature point on the sphere
    x_p = ang_p[0]
    y_p = ang_p[1]
    z_p = ang_p[2]

    x_q = ang_q[0]
    y_q = ang_q[1]
    z_q = ang_q[2]

    # recover angular variables
    t_p = theta(x_p, y_p, z_p)
    p_p = phi(x_p, y_p)

    t_q = theta(x_q, y_q, z_q)
    p_q = phi(x_q, y_q)

    # full weight 
    weight = w_p*ang_w_p*w_q*ang_w_q 
    
    '''
    Now we need to deal how to return this 
    '''
    pts = [r_p, t_p, p_p, r_q, t_q, p_q]
    return weight, pts
    # return [weight, r_p, t_p, p_p, r_q, t_q, p_q]

# Unpacks the quadrature so that integration is easy
def unpack_mass_quad(quad):
    '''
    Given the initial form of how the quadrature is saved,
    this, unpacks and rearranges the quadrature, so that
    it is easy to use.

    Input:
        - quad: 
            the quadrature for p and q, with a specific
            order, and in cartesian coordinates for
            the angular part.

    Output: 
        - the quadrature in the shape:
            [weight, points]
            
            weight is a scalar, points is in spherical 
            coordinates, [r, t, p]
    '''

    # radial quadrature for p
    r = quad[0][0]
    w = quad[0][1]

    # angular quadrature for p
    ang     = quad[1][0]    # point (x, y, z) on the unit sphere
    ang_w   = quad[1][1]

    # cartesian quadrature point on the sphere
    x = ang[0]
    y = ang[1]
    z = ang[2]

    # recover angular variables
    t = theta(x, y, z)
    p = phi(x, y)

    # full weight 
    weight = w*ang_w
    
    '''
    Now we need to deal how to return this 
    '''
    pts = [r, t, p]
    return weight, pts
 

'''
On this file we try to create a list that
contains all the points and weights to perform
the 6 dimensional integration for the Landau 
collision operator
'''
def quadrature():
    '''
    choose the integration order here
    '''
    n_laguerre  = N_LAGUERRE
    n_lebedev   = N_LEBEDEV

    # extract the coefficients
    alpha = 1/2
    x, w_r = roots_genlaguerre(n_laguerre, alpha, False)
    lag = []
    for point, weight in zip(x, w_r):
        '''
        we change variables. This is 
        needed because we are integrating 
        with the weight e^(-r^2/2), in 
        spherical coordinates.
        '''
        new_point  = np.sqrt(2*point)
        new_weight = weight*np.sqrt(2)

        # append
        lag.append([new_point, new_weight])

    # print("Radial integration: ")
    # print(lag)

    # build library
    leblib = PyLebedev()
    s, w_spher = leblib.get_points_and_weights(n_lebedev)
    leb = []
    for p, w in zip(s,w_spher):
        leb.append([p, 4*np.pi*w])

    # print("Spherical integration:")
    # print(leb)

    '''
    Tensorize these
    '''
    # empty list
    tensorized = []

    # in p
    for radial_p in lag:
        for ang_p in leb:
    
            # in q
            for radial_q in lag:
                for ang_q in leb:

                    tensorized.append([radial_p, ang_p, radial_q, ang_q]) # this is important, need to keep track order for unpacking

    return tensorized

# mass quadrature
def mass_quadrature():
    '''
    choose the integration order here
    '''
    n_laguerre  = N_LAGUERRE
    n_lebedev   = N_LEBEDEV

    # extract the coefficients
    alpha = 1/2
    x, w_r = roots_genlaguerre(n_laguerre, alpha, False)
    lag = []
    for point, weight in zip(x, w_r):
        '''
        we change variables. This is 
        needed because we are integrating 
        with the weight e^(-r^2/2), in 
        spherical coordinates.
        '''
        new_point  = np.sqrt(2*point)
        new_weight = weight*np.sqrt(2)

        # append
        lag.append([new_point, new_weight])

    # print("Radial integration: ")
    # print(lag)

    # build library
    leblib = PyLebedev()
    s, w_spher = leblib.get_points_and_weights(n_lebedev)
    leb = []
    for p, w in zip(s,w_spher):
        leb.append([p, 4*np.pi*w])

    # print("Spherical integration:")
    # print(leb)

    '''
    Tensorize these
    '''
    # empty list
    tensorized = []

    for radial in lag:
        for angle in leb:
                tensorized.append([radial, angle]) # this is important, need to keep track order for unpacking

    return tensorized


# saves the quadrature as pkl file
def save_quadrature():
    # obtain the quadrature rule
    tensorized = quadrature()

    # Save the points AND the order they were built with. Tagging the file lets
    # the operator filename (naming.operator_tag) reflect the ACTUAL order used,
    # even if N_LAGUERRE/N_LEBEDEV are changed later without regenerating this file.
    os.makedirs('./quadrature', exist_ok=True)
    payload = {'points': tensorized, 'n_lag': N_LAGUERRE, 'n_leb': N_LEBEDEV}
    with open('./quadrature/quadrature.pkl', 'wb') as file:
        pickle.dump(payload, file)

    print("operator quadrature has been saved.")
   
# save mass quadrature
def save_mass_quadrature():
    # obtain the mass quadrature
    tensorized = mass_quadrature()

    # tag with the order used (same rationale as save_quadrature) so the mass
    # matrix filename can reflect the actual mass-quadrature order.
    os.makedirs('./quadrature', exist_ok=True)
    payload = {'points': tensorized, 'n_lag': N_LAGUERRE, 'n_leb': N_LEBEDEV}
    with open('./quadrature/mass_quadrature.pkl', 'wb') as file:
        pickle.dump(payload, file)

    print("mass quadrature has been saved.")
 
# loads the quadrature (returns the list of quadrature points)
def load_quad():
    with open('./quadrature/quadrature.pkl', 'rb') as file:
        data = pickle.load(file)
    # new format: dict {'points', 'n_lag', 'n_leb'}; old format: bare points list
    return data['points'] if isinstance(data, dict) else data

# returns the (n_lag, n_leb) order the operator quadrature was actually built with.
# `path` lets callers in other directories (e.g. time_evol/) point at ../src/...
def load_quad_order(path='./quadrature/quadrature.pkl'):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, dict):
        return data['n_lag'], data['n_leb']
    # old bare-list pickle predates order-tagging: fall back to current constants
    return N_LAGUERRE, N_LEBEDEV

def load_mass_quad():
    with open('./quadrature/mass_quadrature.pkl', 'rb') as file:
        data = pickle.load(file)
    # new format: dict {'points', 'n_lag', 'n_leb'}; old format: bare points list
    return data['points'] if isinstance(data, dict) else data

# this tests the full operator quadrature
def test(tensorized):
    '''
    Test the function
    '''
    # test functions
    def f(r, t, p):
        return 1

    def g(r, t, p):
        return 1

    # numerical integration
    partial_sum = 0
    for quad in tensorized:
        # unpack the quadrature 
        weight, points = unpack_quad(quad)
        r_p, t_p, p_p, r_q, t_q, p_q = points # this is the convention
        # perform the partial sum
        f1 = f(r_p, t_p, p_p)
        f2 = g(r_q, t_q, p_q)

        partial_sum = partial_sum + weight*f1*f2

    print("six dimensional test: ", partial_sum)

# test the mass quadrature
def mass_test(quad):
    '''
    Test the function
    '''
    # test functions
    def f(r, t, p):
        # result = (3/2 -r**2)**2
        return 1

    # numerical integration
    partial_sum = 0
    for q in quad:
        # unpack the quadrature
        weight, points = unpack_mass_quad(q)
        r, t, p = points # this is the convention
        # perform the partial sum
        sample = f(r, t, p)

        partial_sum = partial_sum + weight*sample

    print("three dimensional test: ", partial_sum)

# the main function
def main():
    # perfrom the test
    test(quadrature())
    # mass_test(load_mass_quad())
    mass_test(mass_quadrature())
    # save the quadrature
    save_quadrature()
    save_mass_quadrature()

if __name__ == "__main__":
    main()
