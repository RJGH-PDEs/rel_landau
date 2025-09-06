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
def unpack_quadrature(quad):
    # radial quadrature
    r_p = quad[0][0]
    w_p = quad[0][1]

    # angular quadrature
    ang_p   =  quad[1][0]
    ang_w_p = quad[1][1]

    # special radial quadrature
    r_u = quad[2][0]
    w_u = quad[2][1]
    
    # angular quadrature for u
    ang_u   = quad[3][0]
    ang_w_u = quad[3][1] 

    # cartesian quadrature point on the sphere
    x_p = ang_p[0]
    y_p = ang_p[1]
    z_p = ang_p[2]

    x_u = ang_u[0]
    y_u = ang_u[1]
    z_u = ang_u[2]

    # exctract angular variables
    t_p = theta(x_p, y_p, z_p)
    p_p = phi(x_p, y_p)

    t_u = theta(x_u, y_u, z_u)
    p_u = phi(x_u, y_u)

    # full weight 
    weight = w_p*ang_w_p*w_u*ang_w_u 
    
    '''
    Now we need to deal how to return this 
    '''
    return [weight, r_p, t_p, p_p, r_u, t_u, p_u]

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
                    tensorized.append([radial, ang_p, radial_u, ang_u])

    return tensorized

# saves the quadrature as pkl file
def save_quadrature():
    # obtain the quadrature rule
    tensorized = quadrature()

    # save full quadrature
    with open('quadrature.pkl', 'wb') as file:
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
        # radial quadrature
        r_p = quad[0][0]
        w_p = quad[0][1]

        # angular quadrature
        ang_p   =  quad[1][0]
        ang_w_p = quad[1][1]

        # special radial quadrature
        r_u = quad[2][0]
        w_u = quad[2][1]
        print("special quadrature")
        print(r_u)
        print(w_u)
        print()

        # angular quadrature for u
        ang_u   = quad[3][0]
        ang_w_u = quad[3][1] 

        # cartesian quadrature point on the sphere
        x_p = ang_p[0]
        y_p = ang_p[1]
        z_p = ang_p[2]

        x_u = ang_u[0]
        y_u = ang_u[1]
        z_u = ang_u[2]

        # perform the partial sum
        f1 = f(r_p, theta(x_p, y_p, z_p), phi(x_p, y_p))
        f2 = g(r_u, theta(x_u, y_u, z_u), phi(x_u, y_u))

        partial_sum = partial_sum + (w_p*ang_w_p)*(w_u*ang_w_u)*f1*f2

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
