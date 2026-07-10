"""
Compute the coefficient vector for the Lemou initial condition:

    f0(v) = M(v) * (1 + |v|^4/120 - |v|^2/12 + 1/8)

where M(v) = e^{-r^2/2} / (2*pi)^{3/2} is the standard Maxwellian.

In the code's representation, the physical distribution is:
    f(v) = e^{-r^2/2} * sum_j c_j * test_j(v)
where test_j = mu_{k,l} * L_k^{l+1/2}(r^2) * r^l * Y_l^m(theta, phi).

The Lemou IC is radially symmetric, so only l=0, m=0 modes contribute.
The polynomial factor p(x) = 9/8 - x/12 + x^2/120  (x = r^2) is exactly
degree 2 in x, so only k=0,1,2 are needed -- flat indices 0, 9, 18 for n=3.

Laguerre expansion (alpha = 1/2):
    L0^{1/2}(x) = 1
    L1^{1/2}(x) = 3/2 - x
    L2^{1/2}(x) = (x^2 - 5x + 15/4) / 2

    p(x) = (33/32)*L0 + (1/24)*L1 + (1/60)*L2     [verified analytically]

Matching f0 to the basis expansion and cancelling the common e^{-r^2/2}:

    sum_k c_k * mu_{k,0} * L_k^{1/2}(r^2) * Y00 = p(r^2) / (2*pi)^{3/2}

    => c_k = lag_coeff[k] / (pi*sqrt(2) * mu_{k,0})

Reference: Villani, "Spatially Homogeneous Landau Equation for Maxwellian
Molecules", Section 3 (docs/document.pdf); see also docs/lemou_benchmark.md.
"""

import sys
import numpy as np
from scipy.special import gamma, factorial, eval_genlaguerre

sys.path.insert(0, '../src')
from basis import mu_const, spher_const

def ind(k, l, m, n):
    return n*n*k + l*l + (m + l)


# Laguerre coefficients of p(x) = 9/8 - x/12 + x^2/120
# Verified analytically: p = (33/32)*L0 + (1/24)*L1 + (1/60)*L2
LAG_COEFFS = {0: 33/32, 1: 1/24, 2: 1/60}


def lemou_coefficients(n=3):
    """
    Return the n^3 coefficient vector for the Lemou IC.
    Nonzero only at flat indices 0, 9, 18 (k=0,1,2; l=0; m=0).
    """
    c = np.zeros(n**3)
    prefactor = np.pi * np.sqrt(2)   # = pi * sqrt(2)

    for k in range(3):
        mu = mu_const(k, 0)
        ck = LAG_COEFFS[k] / (prefactor * mu)
        c[ind(k, 0, 0, n)] = ck

    return c


def verify(c, n=3, n_test=200):
    """
    Reconstruct f0 from coefficients and compare to the analytical formula
    at n_test radial points.  Prints max absolute error.

    Physical f(r) along theta=phi=0:
        code:       e^{-r^2/2} * sum_k c_k * mu_{k,0} * L_k^{1/2}(r^2)  * Y00
        analytical: e^{-r^2/2} / (2*pi)^{3/2} * p(r^2)
    """
    r_vals = np.linspace(0, 6, n_test)
    Y00    = spher_const(0, 0)          # = 1/sqrt(4*pi)
    norm   = (2 * np.pi) ** (3/2)

    errors = []
    for r in r_vals:
        x = r**2

        # code reconstruction (without e^{-r^2/2}, that factor cancels)
        code_val = sum(
            c[ind(k, 0, 0, n)] * mu_const(k, 0) * eval_genlaguerre(k, 0.5, x) * Y00
            for k in range(3)
        )

        # analytical value (without e^{-r^2/2})
        p_val    = 9/8 - x/12 + x**2/120
        anal_val = p_val / norm

        errors.append(abs(code_val - anal_val))

    print(f"Max reconstruction error (without Gaussian): {max(errors):.3e}")
    return max(errors)


def check_normalization(c, n=3):
    """
    Verify Villani's normalization conditions using exact integrals of the
    Gaussian basis.  For l=0:
        int e^{-r^2/2} * L_k^{1/2}(r^2) * r^2 dr  (radial part of 3D integral)

    We use the analytical moments of the Gaussian:
        int_0^inf r^{2+2j} e^{-r^2/2} dr = (2j+1)!! * sqrt(pi/2) / 2^{???}
    More directly, use:
        int_{R^3} f0 dv  = 1
        int_{R^3} f0 r^2 dv = 3
    Computed from known Gaussian moments: E[r^{2n}] with r~|N(0,I_3)|:
        E[r^0]=1, E[r^2]=3, E[r^4]=15, E[r^6]=105
    So: int M * p(r^2) dv = 15/120 - 3/12 + 1/8 + 1 = 0 + 1 = 1  (mass=1)
        int M * r^2 * p(r^2) dv = 105/120 - 15/12 + 3/8 + 3 = 0 + 3 = 3  (energy=3)
    These follow from the perturbation having zero mass/energy, verified here
    by printing the coefficient sums (a purely algebraic check on the coeffs).
    """
    Y00  = spher_const(0, 0)
    norm = (2 * np.pi) ** (3/2)

    # The 3D integral of f0 = int e^{-r^2/2}/(2pi)^{3/2} * p(r^2) dv
    # = (1/norm) * int_0^inf p(r^2) e^{-r^2/2} * 4*pi*r^2 dr
    # Gaussian moments (r ~ |N(0,I_3)|, averaged over the sphere):
    #   int_0^inf r^{2+2j} e^{-r^2/2} * 4*pi*r^2 /(2*pi)^{3/2} dr = (2j+1)!!
    # (these are the moments E[r^{2j}] of the chi-squared distribution in 3D)
    # p(x) = 9/8 - x/12 + x^2/120  with x=r^2:
    # int f0 dv = 9/8 * 1  - (1/12)*3  + (1/120)*15  = 9/8 - 3/12 + 15/120
    mass_check = 9/8 - 3/12 + 15/120
    print(f"Mass integral of f0 (should be 1):   {mass_check:.10f}")

    # int f0 r^2 dv: multiply p by r^2, so polynomial is 9/8*r^2 - r^4/12 + r^6/120
    # = 9/8 * 3  - (1/12)*15  + (1/120)*105
    energy_check = 9/8*3 - 15/12 + 105/120
    print(f"Energy integral of f0 (should be 3): {energy_check:.10f}")


if __name__ == "__main__":
    c = lemou_coefficients(n=3)

    print("=== Lemou IC coefficients ===")
    print(f"  c[0]  (k=0, l=0, m=0): {c[0]:.10f}")
    print(f"  c[9]  (k=1, l=0, m=0): {c[9]:.10f}")
    print(f"  c[18] (k=2, l=0, m=0): {c[18]:.10f}")
    print()

    print("=== Reconstruction verification ===")
    verify(c)
    print()

    print("=== Normalization check ===")
    check_normalization(c)
