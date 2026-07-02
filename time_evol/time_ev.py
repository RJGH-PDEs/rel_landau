import os
import sys
import numpy as np
import pickle
from bilinear import landau
from bilinear import update
# reach the shared naming helper + quadrature-order constants in ../src
sys.path.insert(0, '../src')
from naming import operator_tag
from quadrature import load_quad_order

# save flag
save = True

# save function
def save_coeff(i, coeff):
    # location for saving coefficients
    coeff_location = "../plot/coeff/"
    os.makedirs(coeff_location, exist_ok=True)

    # name
    name = coeff_location + str(i) + ".pkl"

    # save it for plotting
    with open(name, 'wb') as file:
        pickle.dump(coeff, file)

# tau
tau = 0.0001
# number of iterations
NUM_ITERATIONS = 10000

# run config -- MUST match the run that produced the operator (selects which
# sparse-operator file to load, via operator_tag).
rel    = True
cons   = False
sparse = True
n      = 3
# read the actual quadrature order from the operator quadrature file in ../src
n_lag, n_leb = load_quad_order('../src/quadrature/quadrature.pkl')
tag    = operator_tag(rel, cons, sparse, n, n_lag, n_leb)

# open mass matrix and operator tensor
with open('../src/mass/mass_inv.pkl', 'rb') as file:
    # mass inverse
    mi = pickle.load(file)
with open(f'../src/sparse_operators/{tag}.pkl', 'rb') as file:
    # sparse operator
    so = pickle.load(file)

# initial condition
f = np.zeros(27)
f[0] = 1
f[1] = 0.1
f[9] = -0.6

# save initial condition
save_coeff(0, f)

# temporary variable
result = np.zeros(27)

# time evolution
for i in range(1, NUM_ITERATIONS):
    # see the evolution
    # print(f)
    # print()

    # apply the landau operator
    landau(so, f, result)
    # apply the inverse of the matrix
    f_next = f + tau*(mi@result)

    # update
    update(f, f_next)

    # save it every few steps
    if i%100 == 0 and save:
        save_coeff(i, f)

'''
check status of final 
state.
'''
print("f: ")
print(f)

landau(so, f, result)
print("Operator on f: ")
print(result)
