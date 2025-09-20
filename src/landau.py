import sympy as sp
# import quadrature unpacker
from quadrature import unpack_quad, load_quad
# import integrand
from integrand import pieces, integrand
# symbolic parts
from kern import kernel

# The Landau Operator
def operator(k, f, g, test, quadrature):
    # numerical integration
    integral = 0
    for q in quadrature:
        # unpack quadrature 
        weight, points = unpack_quad(q)
        # sample the function
        sample = integrand(k, f, g, test, points)
        # update sum
        integral = integral + weight*sample
        # print(sum)

    return integral

# test the operator
def operator_parallel(select, shared_data):
    # unpack shared data
    quad        = shared_data[0]
    sym_kern    = shared_data[1]

    # produce the numpy pieces
    k, f, g, test = pieces(select, sym_kern)

    # compute the landau operator
    result = operator(k, f, g, test, quad)

    # print results
    print("select: ", select, "result: ", result)

    return [select, result]

# test the operator
def operator_test(select, energy):
    # load the quadrature
    quad = load_quad()

    # print the size of the quadrature
    print("quadrature length: ", len(quad))
    print()

    # produce the symbolic pieces
    sym_kern        = kernel(energy)
    # print(sym_kern)
    k, f, g, test   = pieces(select, sym_kern)

    # compute the landau operator
    result = operator(k, f, g, test, quad)

    # print results
    print("select: ", select)
    print("result: ", result ) 

# The main function
def test():
    # radial symbol
    r = sp.symbols('r')

    # test function
    k = 1
    l = 1
    m = 1
    
    # f
    k1 = 0
    l1 = 1
    m1 = 1
    
    # g
    k2 = 0
    l2 = 0
    m2 = 0

    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

    '''
    Choose the energy
    '''
    energy = (1/2)*r**2       # non-relativistic
    # energy = sp.sqrt(1+r**2)  # relativistic
    # energy   = r**3             # polynomial

    # test the operator
    operator_test(select, energy)


# main function
def main():
    test()

if __name__ == "__main__":
    main()
