import sympy as sp
import numpy as np

# symbolic parts
from kern import kernel
from basis import basis, grad_weighted, gradient

# the integrand
def integrand(k_sym, f_sym, g_sym, test_sym, points):
    i = 0

    # lambdafy
    rp, tp, pp, rq, tq, pq = sp.symbols('rp tp pp rq tq pq')
    r, t, p = sp.symbols('r t p')

    k       = sp.lambdify((rp, tp, pp, rq, tq, pq), k_sym, modules='numpy')
    f       = sp.lambdify((r, t, p), f_sym, modules = 'numpy')
    g       = sp.lambdify((r, t, p), g_sym, modules = 'numpy')
    grad_test    = sp.lambdify((r, t, p), test_sym, modules = 'numpy')

    # the evaluation points
    rp = points[0]
    tp = points[1]
    pp = points[2]
    rq = points[3]
    tq = points[4]
    pq = points[5]

    # print
    a = f(rp, tp, pp)*g(rq, tq, pq)
    b = (k(rp, tp, pp, rq, tq, pq)@(grad_test(rp, tp, pp) - grad_test(rq, tq, pq)))
    
    # inner product
    i = np.dot(a.flatten(),b.flatten())

    return i

# produces the symbolic pieces given a selection of the indices
def sym_pieces(select):
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
    f       = basis(k1, l1, m1)
    g       = grad_weighted(basis(k2, l2, m2))
    test    = gradient(basis(k, l, m))

    return f, g, test
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
    kern = kernel()
    f, g, test = sym_pieces(select)

    # call the integrand
    print(integrand(kern, f, g, test, points))

    



