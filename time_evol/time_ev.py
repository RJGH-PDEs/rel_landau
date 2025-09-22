import numpy as np
import pickle
from bilinear_operator import landau
from bilinear_operator import update

# tau
tau = 0.00001
# number of iterations
NUM_ITERATIONS = 100000

# open mass matrix and operator tensor
with open('mass_inverse.pkl', 'rb') as file:
    # mass inverse
    mi = pickle.load(file)
with open('sparse_operator.pkl', 'rb') as file:
    # sparse operator
    so = pickle.load(file)

# initial condition
f = np.zeros(27)
f[0] = 1
f[1] = 0.1
f[9] = -0.6


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
