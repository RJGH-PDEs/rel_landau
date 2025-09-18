# sympy
import sympy as sp
# basis
from basis import basis, mu_const
# integration
from quadrature import unpack_mass_quad, load_mass_quad

# the integrand for the mass matrix
def integrand(f, phi, point):
    # the evaluation point
    r = point[0]
    t = point[1]
    p = point[2]

    # parts
    trial   = f(r, t, p)
    test    = phi(r, t, p)
    # print((3/2 - r**2), trial, test)
    # return (3/2 - r**2)**2

    return trial*test

# produces the pieces given a selection of the indices
def pieces(select):
    # trial function
    ki = select[0][0]
    li = select[0][1]
    mi = select[0][2]
    # print(ki, li, mi)
    # test function
    kj = select[1][0]
    lj = select[1][1]
    mj = select[1][2]
    # print(kj, lj, mj)
    
    # produce symbolic pieces
    test_sym    = basis(kj, lj, mj)
    # print(test_sym)

    # the two basis functiosn will include the mu constant
    f_sym       = basis(ki, li, mi)*mu_const(ki,li)
    # print(f_sym) 
    # lambdafy
    r, t, p = sp.symbols('r t p')

    # numpy pieces
    f       = sp.lambdify((r, t, p), f_sym, modules = 'numpy')              # trial function
    test    = sp.lambdify((r, t, p), test_sym, modules = 'numpy')           # test function
    
    # return the pieces
    return f, test

# coefficient
def coefficient(select, quad):
    # produce the numpy pieces
    f, test = pieces(select)

    # numerical integration
    partial_sum = 0
    for quad in quad:
        # unpack the quadrature 
        weight, points = unpack_mass_quad(quad)
        # perform the partial sum

        sample = integrand(f, test, points)
        partial_sum = partial_sum + weight*sample

    return partial_sum

# test a coefficient
def coeff_test():
    # select
    select = [[1, 0, 0],[1, 0, 0]]
    
    # obtain the mass quadrature
    quad = load_mass_quad()

    # test
    coeff = coefficient(select, quad)
    print("for: ", select, " the coeff is: ", coeff)

# The main function
def main():
    coeff_test()

if __name__ == "__main__":
    main()
