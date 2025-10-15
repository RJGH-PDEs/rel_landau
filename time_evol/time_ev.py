import numpy as np
import pickle
from bilinear import landau
from bilinear import update

# save flag
save = True

# save function
def save_coeff(i, coeff):
    # location for saving coefficients
    coeff_location = "../plot/coeff/"
    
    # name 
    name = coeff_location + str(i) + ".pkl"

    # save it for plotting
    with open(name, 'wb') as file:
        pickle.dump(f, file)

# tau
tau = 0.00001
# number of iterations
NUM_ITERATIONS = 10

# initial condition
f = np.zeros(27)
f[0] = 43.6
f[9] = -44.35
f[18] = 14.7

# save initial condition
save_coeff(0, f)

# open mass matrix and operator tensor
with open('../src/mass/mass_inv.pkl', 'rb') as file:
    # mass inverse
    mi = pickle.load(file)
with open('../src/sparse_operators/sparse_new_basis.pkl', 'rb') as file:
    # sparse operator
    so = pickle.load(file)

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
    next = f + tau*(mi@result)
    
    # update
    update(f, next)

    # save it every few steps
    if i%1 == 0 and save:
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
