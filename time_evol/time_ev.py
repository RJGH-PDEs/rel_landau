import numpy as np
import pickle
from bilinear import landau
from bilinear import update

# tau
tau = 0.0001
# number of iterations
NUM_ITERATIONS = 100000
# location for saving coefficients
coeff_location = "../plot/coeff/"

# open mass matrix and operator tensor
with open('../src/mass/mass.pkl', 'rb') as file:
    # mass inverse
    mi = pickle.load(file)
with open('../src/sparse_operators/test.pkl', 'rb') as file:
    # sparse operator
    so = pickle.load(file)

# initial condition
f = np.zeros(27)
f[0] = 1
f[1] = 0.1
f[9] = -0.6

# save it for plotting
name = coeff_location + "0.pkl"
with open(name, 'wb') as file:
    pickle.dump(f, file)

'''
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

print("f: ")
print(f)

# pickle it
name = str(NUM_ITERATIONS) + ".pkl"

with open(name, 'wb') as file:
    pickle.dump(f, file)

landau(so, f, result)
print("Operator on f: ")
print(result)
'''
