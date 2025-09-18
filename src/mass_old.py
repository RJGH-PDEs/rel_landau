import numpy as np
# integration
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev
# test functions
from test_func import test
from test_func import mu_const
from test_func import Phi

# function to be integrated
def f_full(r, t, p, select):
    # basis function
    k_i = select[0]
    l_i = select[1]
    m_i = select[2]
    # test function
    k_j = select[3]
    l_j = select[4]
    m_j = select[5] 
    # return product of test and trial
    return mu_const(k_i, l_i)*test(k_i, l_i, m_i, r, t, p)*test(k_j, l_j, m_j, r, t, p)

# function to be sampled
def g_full(x, t, p, select):
    r = np.sqrt(2*x)
    return f_full(r, t, p, select) * np.sqrt(2)

# computes integral for the mass matrix
def mass_full(k_i, l_i, m_i, k_j, l_j, m_j):
    '''
    here we follow the convention:
    - i = i(k_i, l_i, m_i) is for the basis function
    - j = j(k_j, l_j, m_j) is for the test function
    '''
    # selection of basis and test 
    select = [k_i, l_i, m_i, k_j, l_j, m_j]

    # integration order
    n_laguerre = 5
    n_lebedev  = 9
    
    # extract Laguerre coefficients
    alpha = 1/2
    x, w_r = roots_genlaguerre(n_laguerre, alpha, False)

    # extract Lebedev coefficients
    leblib = PyLebedev()
    s, w_sph = leblib.get_points_and_weights(n_lebedev)

    # partial sum variable
    sum = 0

    # radial counter
    rad_counter = 0

    # tensor iteration
    for rad_weight in w_r:
        # spherical counter
        sph_counter = 0
        for sph_weight in w_sph:
            # points
            x_samp = x[rad_counter] 
            t_samp = theta(s[sph_counter, 0], s[sph_counter, 1], s[sph_counter, 2])
            p_samp = phi(s[sph_counter, 0], s[sph_counter, 1])

            # update the sum
            sum = sum + g_full(x_samp, t_samp, p_samp, select)*rad_weight*sph_weight

            # update spherical counter
            sph_counter = sph_counter + 1

        # update radial counter
        rad_counter = rad_counter + 1
    
    # multiply by 4 pi
    sum = sum*(4*np.pi)

    print('result is', sum)

'''
Using simplifications
'''

# function to be integrated
def f(r, select):
    # basis function
    k_i = select[0]
    l_i = select[1]
    # test function
    k_j = select[3]
    l_j = select[4]
    # return radial parts
    return Phi(l_i, k_i, r)*Phi(l_j, k_j, r)*mu_const(k_i, l_i)

# function to be sampled
def g(x, select):
    r = np.sqrt(2 * x)
    return f(r, select)*np.sqrt(2)

# now using simplifications
def mass(k_i, l_i, m_i, k_j, l_j, m_j):
    '''
    here we follow the convention:
    - i = i(k_i, l_i, m_i) is for the basis function
    - j = j(k_j, l_j, m_j) is for the test function
    '''

    # consider the orthogonality of the spherical harmonics
    if (l_i != l_j) or (m_i != m_j):
        print('simplified is: ', 0)   
        return 0
    
    # selection of basis and test 
    select = [k_i, l_i, m_i, k_j, l_j, m_j]

    # integration order
    n_laguerre = 5

    # extract integration scheme
    alpha = 1/2
    x, w_r = roots_genlaguerre(n_laguerre, alpha, False)

    # iteration variables
    sum = 0
    counter = 0

    # numerical integration
    for radial_weight in w_r:
        # point
        x_sampled = x[counter]
        # update the sum
        sum = sum + g(x_sampled, select)*radial_weight
        # update the counter 
        counter = counter + 1

    # print the result
    print('simplified is: ', sum)
    return sum

# test both
k_i = 0
l_i = 0
m_i = 0

k_j = 0
l_j = 1
m_j = -1

(mass_full(k_i, l_i, m_i, k_j, l_j, m_j))
(mass(k_i, l_i, m_i, k_j, l_j, m_j))


'''


'''
def lebedev_example():
    # number of points
    n_lebedev  = 9
    
    # extract Lebedev coefficients
    leblib = PyLebedev()
    s, w_sph = leblib.get_points_and_weights(n_lebedev)

    # partial sum variable
    sum = 0
    sph_counter = 0
    # try this
    for sph_weight in w_sph:
        # points
        t_samp = theta(s[sph_counter, 0], s[sph_counter, 1], s[sph_counter, 2])
        p_samp = phi(s[sph_counter, 0], s[sph_counter, 1])

         # update the sum
        sum = sum + np.sin(t_samp)*np.sin(p_samp)*sph_weight

        # update spherical counter
        sph_counter = sph_counter + 1
    
    # multiply by 4 pi
    sum = sum*(4*np.pi)

    print('result is', sum)

lebedev_example()
