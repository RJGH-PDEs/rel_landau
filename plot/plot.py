import matplotlib.pyplot as plt
import numpy as np
from lc import linear_comb
import pickle

# flag to save the figure
save = True

# generate cartesian points
n = 100
x = np.linspace(-5, 5, n)
y = np.zeros(n)
z = np.zeros(n)

# generate spherical cordinates
def theta(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    if r == 0:
        return 0
    else:
        return np.arccos(z/ r)
    
def phi(x, y):
    r = np.sqrt(x**2 + y**2)

    if r == 0:
        return 0
    elif y == 0:
        return np.arccos(x/r) # new discovery, this might be wrong
    else:
        return np.sign(y)*np.arccos(x/r)
    
def radius(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

# spherical points
r = np.zeros(n)
t = np.zeros(n)
p = np.zeros(n)

# counter
i = 0
for point in x:
    # compute the points
    r[i] = radius(x[i], y[i], z[i])
    t[i] = theta(x[i], y[i], z[i])
    p[i] = phi(x[i], y[i])
    # advance the counter
    i = i + 1

# print the points
'''
print(r)
print(t)
print(p)
'''

# will store the function
f = np.zeros(n)

# coefficients
# coeff = np.zeros(27)
# coeff[0] = 1
# coeff[1] = 0.1
# coeff[9] = -0.6

# open the result
time = 9000
file_name = "coeff/" + str(time) + ".pkl"
# file_name = "1.pkl"

with open(file_name, 'rb') as file:
    data = pickle.load(file)

# print(data)
# counter
i = 0
for point in r:
    func_val = linear_comb(data, r[i], t[i], p[i])
    f[i] = np.exp((-r[i]**2)/2)*func_val
    i = i + 1

# Create the plot
plt.plot(x, f, marker='o')  # marker='o' will put points at each (x, y)
plt.title('Solution at after ' + str(time) + ' iterations')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

if save:
    # Save the figure
    figure_name = "./figures/" + str(time) + ".png"
    plt.savefig(figure_name)
    # plt.show()
