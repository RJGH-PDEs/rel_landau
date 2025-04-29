import sympy as sp
import numpy as np

# symbolic parts
from kern import kernel
from basis import basis, grad_weighted, gradient

# the integrand
def integrand(k, f, g, gt, points):
    i = 0

    # the evaluation points
    rp = points[0]
    tp = points[1]
    pp = points[2]
    rq = points[3]
    tq = points[4]
    pq = points[5]

    # print
    a = f(rp, tp, pp)*g(rq, tq, pq)
    b = (k(rp, tp, pp, rq, tq, pq)@(gt(rp, tp, pp) - gt(rq, tq, pq)))
    
    # inner product
    i = np.dot(a.flatten(),b.flatten())

    return i

# produces the pieces given a selection of the indices
def pieces(select, k_sym):
    # choose the three functions
    k = select[0][0]
    l = select[0][1]
    m = select[0][2]

    k1 = select[1][0]
    l1 = select[1][1]
    m1 = select[1][2]

    k2 = select[2][0]
    l2 = select[2][1]
    m2 = select[2][2]

    # produce them
    f_sym       = basis(k1, l1, m1)
    g_sym       = grad_weighted(basis(k2, l2, m2))
    test_sym    = gradient(basis(k, l, m))
    
    # lambdafy
    rp, tp, pp, rq, tq, pq = sp.symbols('rp tp pp rq tq pq')
    r, t, p = sp.symbols('r t p')

    k       = sp.lambdify((rp, tp, pp, rq, tq, pq), k_sym, modules='numpy')
    f       = sp.lambdify((r, t, p), f_sym, modules = 'numpy')
    g       = sp.lambdify((r, t, p), g_sym, modules = 'numpy')
    test    = sp.lambdify((r, t, p), test_sym, modules = 'numpy')


    return k, f, g, test
# testing the integrand
def test_integrand():
    # choose the three functions
    k = 1
    l = 0
    m = 0

    k1 = 1
    l1 = 0
    m1 = 0

    k2 = 1
    l2 = 0
    m2 = 0

    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

    # define the points
    rp = 4
    tp = np.pi/6
    pp = np.pi/3

    rq = 1
    tq = np.pi/5
    pq = np.pi/6

    points = [rp, tp, pp, rq, tq, pq] 

    # produce the sympy parts
    # radial symbol
    r = sp.symbols('r')
    energy = sp.sqrt(1+r**2)  # relativistic

    kern = kernel(energy)
    k, f, g, test = pieces(select, kern)

    # call the integrand
    print(integrand(k, f, g, test, points))

    



