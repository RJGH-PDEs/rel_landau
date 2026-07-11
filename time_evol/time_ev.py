import os
import sys
import numpy as np
import pickle
from bilinear import landau
from bilinear import update
# reach the shared naming helper + quadrature-order constants in ../src
sys.path.insert(0, '../src')
from naming import operator_tag, mass_tag, load_with_meta
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

# run config -- used to LOCATE the sparse-operator file (via operator_tag). The
# authoritative n / config come from the loaded artifact's metadata, not here.
rel    = False
cons   = False
sparse = False
n      = 3
# read the actual quadrature order from the operator quadrature file in ../src
n_lag, n_leb = load_quad_order('../src/quadrature/quadrature.pkl')
tag    = operator_tag(rel, cons, sparse, n, n_lag, n_leb)

# open the sparse operator tensor + its metadata
so, meta = load_with_meta(f'../src/sparse_operators/{tag}.pkl')
n = meta['n']                       # authoritative dof from the artifact

# open the mass matrix that MATCHES this run's mass quadrature and n
m_lag, m_leb = load_quad_order('../src/quadrature/mass_quadrature.pkl')
mi, mass_meta = load_with_meta(f'../src/mass/{mass_tag(n, m_lag, m_leb)}.pkl')
assert mass_meta['n'] == n, f"mass matrix n={mass_meta['n']} != operator n={n}"

# initial condition: 'symmetric' | 'asymmetric' | 'zero_momentum' | 'lemou'
#   symmetric     -- double hump, radially symmetric (no l>0 modes)
#   asymmetric    -- same double hump + cos(theta) perturbation (non-zero net momentum)
#   zero_momentum -- asymmetric but net momentum = 0 in all directions;
#                    coeff[11] chosen so C0*coeff[2] + C1*coeff[11] = 0
#                    where C_k = mu(k,1) * integral L_k^{3/2}(r^2) r^4 e^{-r^2/2} dr
#                    ratio -C0/C1 = 0.632456  (basis-dependent, verified numerically)
#   lemou         -- Villani/Lemou exact benchmark (Maxwellian molecules, isotropic);
#                    f0 = M(v)*(1 + r^4/120 - r^2/12 + 1/8), exact analytical solution
#                    h(t,v) = M(v)*(1 + e^{-8t}*(r^4/120 - r^2/12 + 1/8))
ic_mode = 'zero_momentum'

# ratio -C0/C1 that zeroes the discrete z-momentum
_ZM_RATIO = 0.632456

if ic_mode == 'lemou':
    sys.path.insert(0, '../experiments/lemou_benchmark')
    from lemou_ic import lemou_coefficients
    f = lemou_coefficients(n)
else:
    f = np.zeros(n**3)
    f[0] =  1.0    # (k=0, l=0, m=0)  Gaussian
    f[9] = -0.5    # (k=1, l=0, m=0)  radial correction → double hump
    if ic_mode == 'asymmetric':
        f[2]  =  0.1                  # (k=0, l=1, m=0)  cos(theta) asymmetry
    if ic_mode == 'zero_momentum':
        f[2]  =  0.1                  # (k=0, l=1, m=0)  cos(theta) asymmetry
        f[11] =  _ZM_RATIO * f[2]    # (k=1, l=1, m=0)  cancels net z-momentum

# save initial condition
save_coeff(0, f)

# temporary variable
result = np.zeros(n**3)

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

    # print norm and save every few steps
    if i%100 == 0:
        print(f"step {i:5d} / {NUM_ITERATIONS}  ||f|| = {np.linalg.norm(f):.6f}")
        if save:
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
