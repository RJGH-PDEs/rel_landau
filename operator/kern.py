import sympy as sp
import numpy as np

# energy gradient
def energy_grad(energy):
    # Define symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')
    r_hat = sp.Matrix([sp.sin(t)*sp.cos(p), sp.sin(t)*sp.sin(p), sp.cos(t)])

    # take partial
    partial = sp.diff(energy, r)
    eg = partial*r_hat

    # return 
    return eg

# kernel (projector)
def kernel(energy):
    # symbols
    r, t, p     = sp.symbols('r t p')
    rp, tp, pp  = sp.symbols('rp tp pp')
    rq, tq, pq  = sp.symbols('rq tq pq')

    # obtain the energy
    e = energy_grad(energy)

    # substitutions
    p_sub = {r: rp, t: tp, p: pp}
    q_sub = {r: rq, t: tq, p: pq} 

    # energies
    ep = e.subs(p_sub)
    eq = e.subs(q_sub)

    # Alternative way
    u = ep - eq
    altern =  u.dot(u)*sp.eye(3) - u*u.T
    # altern = sp.simplify(altern)

    # Relativistic part
    z = (ep+eq)/2
    cross = z.cross(u)
    altern = altern - cross*cross.T
    # altern = sp.simplify(altern)
    
    # check conservation
    print("energy: ")
    print(energy)
    print()
    print("kernel: ")
    print(altern)
    print()
    print("u - relative velocity: ")
    print(u)
    print()
    # print("check conservation (S*u) should be zero 0")
    # print()
    # print(sp.simplify(altern * u))
    # print()

    return altern
    ''' 
    # vx, vy, vz, wx, wy, wz = sp.symbols('vx vy vz wx wy wz')
    print()
    print(ep)
    print(eq)

    # vectors
    v = sp.Matrix([vx, vy, vz])
    w = sp.Matrix([wx, wy, wz])

    # relative and mean velocity
    u = v - w

    # the tensor field
    S = u.dot(u)*sp.eye(3) - u*u.T
    S = sp.simplify(S)
     # relativistic part
    z = (v + w)/2
    cross = z.cross(u)
    S = S - cross*cross.T
    S = sp.simplify(S)
    print(S)
    
    # obtain the energy
    e = energy_grad()

    # substitutions
    p_sub = {r: rp, t: tp, p: pp}
    q_sub = {r: rq, t: tq, p: pq} 

    # energies
    ep = e.subs(p_sub)
    eq = e.subs(q_sub)

    
    print()
    print(ep)
    print(eq)
    

    # substitute
    sub = {v[0]: ep[0], v[1]: ep[1], v[2]: ep[2], w[0]: eq[0], w[1]: eq[1], w[2]: eq[2]}
    ker = sp.simplify(S.subs(sub))
    
    print("Kernel:")
    print(ker)
    '''
# cartesian energy grad
def energy_grad_cart():
    # Symbols
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')

    # norm an unit vector
    pos = sp.Matrix([x, y, z])
    r   = sp.sqrt(pos.dot(pos))
    print(r)
    
    # energy 
    e = sp.sqrt(1+r**2)

    # energy gradient
    eg = sp.Matrix([sp.diff(e, x), sp.diff(e, y), sp.diff(e,z)])
    eg = sp.simplify(eg)

    return eg