import sympy as sp
import pickle
# import quadrature unpacker
from unpack import unpack_quadrature
# import integrand
from integrand import integrand
# symbolic parts
from kern import kernel
from integrand import pieces

# loads the quadrature
def load_quad():
    with open('quadrature.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# The Landau Operator
def operator(k, f, g, test, quadrature):
    # numerical integration
    sum = 0
    for q in quadrature:
        # unpack quadrature 
        weight, points = unpack_quadrature(q)
        
        # sample the function
        # sample = integrand(k_sym, f_sym, g_sym, test_sym, points)
        sample = integrand(k, f, g, test, points)
        # update sum
        sum = sum + weight*sample
        # print(sum)

    return sum

# test the operator
def operator_parallel(select, shared_data):
    # unpack shared data
    quad = shared_data[0]
    sym_kern = shared_data[1]

    # produce the symbolic pieces
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
    kern = kernel(energy)
    f, g, test = pieces(select)

    # compute the landau operator
    result = operator(kern, f, g, test, quad)

    # print results
    print("select: ", select)
    print("result: ", result ) 

# The main function
def main():
    # radial symbol
    r = sp.symbols('r')

    # choose the three functions
    k = 1
    l = 2
    m = 0

    k1 = 1
    l1 = 1
    m1 = 1

    k2 = 0
    l2 = 1
    m2 = 1

    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

    '''
    Choose the energy
    '''
    # energy = (1/2)*r**2       # non-relativistic
    energy = sp.sqrt(1+r**2)  # relativistic
    # energy   = r**3             # polynomial

    # test the operator
    operator_test(select, energy)

    
   
# Call the main function
if __name__ == "__main__":
    main()