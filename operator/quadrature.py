# imports
import pickle
import numpy as np

# integration 
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

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
        return np.arccos(x/r) # new discovery, this might be wrong
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
    return [weight, r_p, t_p, p_p, r_q, t_q, p_q]

'''
this is an older version
'''
def unpack_quadrature(quad):
    # radial quadrature
    r_p = quad[0][0]
    w_p = quad[0][1]

    # angular quadrature
    ang_p   =  quad[1][0]
    ang_w_p = quad[1][1]

    # special radial quadrature
    r_q = quad[2][0]
    w_q = quad[2][1]
    
    # angular quadrature for u
    ang_q   = quad[3][0]
    ang_w_q = quad[3][1] 

    # cartesian quadrature point on the sphere
    x_p = ang_p[0]
    y_p = ang_p[1]
    z_p = ang_p[2]

    x_q = ang_q[0]
    y_q = ang_q[1]
    z_q = ang_q[2]

    # exctract angular variables
    t_p = theta(x_p, y_p, z_p)
    p_p = phi(x_p, y_p)

    t_q = theta(x_q, y_q, z_q)
    p_q = phi(x_q, y_q)

    # full weight 
    weight = w_p*ang_w_p*w_q*ang_w_q 
    
    '''
    Now we need to deal how to return this 
    '''
    return [weight, r_p, t_p, p_p, r_q, t_q, p_q]

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
    n_laguerre  = 8
    n_lebedev   = 5

    # extract the coefficients
    alpha = 1/2
    x,w_r = roots_genlaguerre(n_laguerre, alpha, False)
    lag = []
    for point, weight in zip(x, w_r):
        '''
        we change variables 
        '''
        new_point  = np.sqrt(2*point)
        new_weight = weight*np.sqrt(2)

        # append
        lag.append([new_point, new_weight])

    # print("Radial integration: ")
    # print(lag)

    # build library
    leblib = PyLebedev()
    s,w_spher = leblib.get_points_and_weights(n_lebedev)
    leb = []
    for p, w in zip(s,w_spher):
        leb.append([p, np.pi*4*w])

    # print("Spherical integration:")
    # print(leb)

    '''
    Tensorize these
    '''
    # empty list
    tensorized = []

    for radial in lag:
        for ang_p in leb:
            for ang_u in leb:
                for radial_u in lag:
                    tensorized.append([radial, ang_p, radial_u, ang_u]) # this is important, need to keep track order for unpacking

    return tensorized

# saves the quadrature as pkl file
def save_quadrature():
    # obtain the quadrature rule
    tensorized = quadrature()

    # save full quadrature
    with open('./quadrature/quadrature.pkl', 'wb') as file:
        pickle.dump(tensorized, file)

    print("quadrature has been saved")

# this tests the quadrature
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
        weight, r_p, t_p, p_p, r_q, t_q, p_q = unpack_quad(quad)
        # perform the partial sum
        f1 = f(r_p, t_p, p_p)
        f2 = g(r_q, t_q, p_q)

        partial_sum = partial_sum + weight*f1*f2

    print(partial_sum)

# the main function
def main():
    # obtain the quadrature
    quad = quadrature()

    # perfrom the test
    test(quad)

    # save the quadrature
    save_quadrature()

if __name__ == "__main__":
    main()
