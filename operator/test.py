from kern import kernel
from basis import basis, gradient, grad_weighted
from integrand import test_integrand

import sympy as sp
import numpy as np


# k_symb = kernel()
# print(k_symb)
# rp, tp, pp, rq, tq, pq = sp.symbols('rp tp pp rq tq pq')
# k = sp.lambdify((rp, tp, pp, rq, tq, pq), k_symb, modules='numpy')

def main():
    k = 1
    l = 0
    m = 0

    rp = 4
    tp = np.pi/6
    pp = np.pi/3

    rq = 1
    tq = np.pi/3
    pq = np.pi/6

    # basis functions
    f = basis(k, l, m)
    print(f)

    # gradient
    grad = gradient(f)
    print(grad)

    # gradient with weight
    g_w = grad_weighted(f)
    print(g_w)

    # test the integrand
    test_integrand()


if __name__ == '__main__':
    main()