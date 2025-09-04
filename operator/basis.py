import sympy as sp

# computes the basis - without the weight
def basis(k, l, m):
    # symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')


    # Spherical harmonic
    sphr = sp.simplify(sp.assoc_legendre(l,abs(m), sp.cos(t)))
    sphr = sp.refine(sphr, sp.Q.positive(sp.sin(t)))

    if m >= 0:
        sphr = sphr*sp.cos(m*p)
    else:
        sphr = sphr*sp.sin(abs(m)*p)

    # Radial part
    radial = 1
    
    if k > 0:
        # alpha
        a = l + 1/2
 
        radial = sp.assoc_laguerre(k, a, r**2)

    radial = radial*r**l

    # the test function
    f = sphr*radial

    return f

# Computes the gradient
def gradient(f):
    # symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')

    # take derivatives
    # compute first partials
    fr = sp.simplify(sp.diff(f, r))
    # print("fr: ", fr)
    ft = sp.simplify(sp.diff(f,t)/r)
    # print("ft: ", ft)
    fp = sp.simplify(sp.diff(f,p)/(r*sp.sin(t)))
    # print("fp: ", fp)

    # basis vectors
    v1 = sp.Matrix([sp.sin(t)*sp.cos(p), sp.sin(t)*sp.sin(p), sp.cos(t)])
    v2 = sp.Matrix([sp.cos(t)*sp.cos(p), sp.cos(t)*sp.sin(p), -sp.sin(t)])
    v3 = sp.Matrix([-sp.sin(p), sp.cos(p), 0 ]) 

    # print("v1: ", v1)
    # print("v2: ", v2)
    # print("v3: ", v3)

    # compute the gradient
    gradient = sp.simplify(fr*v1 + ft*v2 + fp*v3)
    
    return gradient

# Computes the gradient, with the Gaussian weight
def grad_weighted(f):
    # symbols
    r = sp.symbols('r')

    # add the exponential
    # g = sp.exp(-r**2)*f
    g = sp.exp((-r**2)/2)*f

    # take the gradient
    grad = gradient(g)

    # take away the Gaussian weight
    # grad = sp.simplify(grad/sp.exp((-r**2)) # should not have any gradient
    grad = sp.simplify(grad/sp.exp((-r**2)/2)) 
    
    return grad

# A test
def test():
    # parameters
    k = 1
    l = 0
    m = 0

    # basis function
    f = basis(k, l, m)

    # print the basis
    print("basis: ", f)

    # print the gradient
    print("gradient: ", gradient(f))

    # print weighted gradient
    print("weighet gradient: ", grad_weighted(f))

# The main function
def main():
    test()

if __name__ == "__main__":
    main()
