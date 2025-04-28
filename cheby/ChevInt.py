import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

# Define the function to interpolate
def func(x):
    return np.sqrt(1+x*x)

# Gaussian weight
def gaussWeight(x):
	return np.exp(-x*x)

# Interpolate using Chebyshev polynomials
L = 2
deg = 4
cheb_poly = Chebyshev.interpolate(func, deg, [-L,L])
cheb_poly_standard = cheb_poly.convert(kind=np.polynomial.Polynomial)
print(cheb_poly_standard)

# Evaluate the polynomial at a new set of x values
L_g = 5
x_new = np.linspace(-L_g, L_g, 500)  # New points to evaluate the interpolation
y_new = cheb_poly(x_new)

# Evaluate the original function at the new points for comparison
y_exact = func(x_new)
y_weight = gaussWeight(x_new)

# Plot the results
plt.plot(x_new, y_new, label='E_h*exp(-x^2)')
plt.plot(x_new, y_exact, '--', label='E*exp(-x^2) ', color='green')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle('Chebyshev Interpolation with polynomial of degree '+str(deg))
plt.title('Both weighted by the Gaussian')
plt.show()
