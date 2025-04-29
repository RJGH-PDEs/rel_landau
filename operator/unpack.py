import numpy as np    
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
        return np.arccos(z/ r)
# phi    
def phi(x, y):
    r = np.sqrt(x**2 + y**2)

    if r == 0:
        return 0
    elif y == 0:
        return np.arccos(x/r) # new discovery, this might be wrong
    else:
        return np.sign(y)*np.arccos(x/r)

# Unpacks the quadrature so that integration is easy
def unpack_quadrature(quad):
    # radial quadrature
    r_p = quad[0][0]
    w_p = quad[0][1]

    # angular quadrature
    ang_p   =  quad[1][0]
    ang_w_p = quad[1][1]

    # radial quadrature for q
    r_q = quad[2][0]
    w_q = quad[2][1]
    
    # angular quadrature for q
    ang_q   = quad[3][0]
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
    points = [r_p, t_p, p_p, r_q, t_q, p_q]

    return [weight, points]