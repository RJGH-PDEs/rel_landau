"""
End-to-end sanity check for the (non-relativistic) collision-tensor pipeline.

Purpose: confirm the pipeline runs and that the operator conserves what it should.
This is NOT a full-accuracy run -- it uses a REDUCED-ORDER quadrature so it finishes
quickly. The conservation identities checked here are pointwise-exact in the integrand
(they hold at any quadrature order), so a low-order run is a valid correctness check;
full-order accuracy of the nonzero entries is validated separately (see TODO Track A,
Parts 2 & 4). The full-order tensor is expensive and is meant to be run on a cluster.

Prereqs: run `python mass_matrix.py` in src/ first (needs src/mass/mass_inv.pkl).

Run:  cd src && python ../pipeline_check.py        (or adjust N_LAG/N_LEB below)

NOTE (macOS): must stay under `if __name__ == '__main__':` -- parallel() uses a
multiprocessing Pool with the 'spawn' start method, which re-imports this module in
each worker; unguarded top-level work would spawn runaway pools.
"""
import os
import pickle
import numpy as np
import sympy as sp
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

from kern import kernel
from parallel import parallel
from sparse import non_zeros, simple_index, dense_op, sparse_op, ind

N_LAG = 5   # radial (generalized-Laguerre) nodes  -- production uses 9
N_LEB = 3   # Lebedev order                        -- production uses 7


def low_quad(n_lag, n_leb):
    """Reduced-order 6D quadrature, same construction as quadrature.py:quadrature()."""
    x, w_r = roots_genlaguerre(n_lag, 0.5, False)
    lag = [[np.sqrt(2 * pt), w * np.sqrt(2)] for pt, w in zip(x, w_r)]
    s, w_s = PyLebedev().get_points_and_weights(n_leb)
    leb = [[p, 4 * np.pi * w] for p, w in zip(s, w_s)]
    return [[rp, ap, rq, aq] for rp in lag for ap in leb for rq in lag for aq in leb]


def main():
    n = 3
    r = sp.symbols('r')
    quad = low_quad(N_LAG, N_LEB)
    print("reduced-order 6D quadrature points:", len(quad))

    kern = kernel((sp.Rational(1, 2)) * r**2, verbose=False, rel=False)   # non-relativistic
    res = parallel([quad, kern], n, rel=False)

    so = sparse_op(dense_op(simple_index(non_zeros(res, 1e-6), n), n))     # list of n^3 CSR matrices
    with open('mass/mass_inv.pkl', 'rb') as f:
        mi = pickle.load(f)
    M = np.linalg.inv(mi)

    def Q(f):
        return np.array([f @ (so[i] @ f) for i in range(n**3)])

    # 1. equilibrium (Maxwellian = e_0) is stationary
    e0 = np.zeros(n**3); e0[0] = 1.0
    print("\n1. ||Q(equilibrium e_0)||_inf =", np.max(np.abs(Q(e0))), "(want ~0)")

    # 2/3. evolution + conservation of the collision invariants
    cons = {'mass': ind(0, 0, 0, n), 'px': ind(0, 1, -1, n), 'py': ind(0, 1, 0, n),
            'pz': ind(0, 1, 1, n), 'energy': ind(1, 0, 0, n)}
    f = np.zeros(n**3); f[0] = 1.0; f[3] = 0.15; f[ind(0, 1, 0, n)] = -0.1
    tau, NT = 1e-3, 4000
    m0 = {k: (M @ f)[i] for k, i in cons.items()}
    q0 = np.max(np.abs(Q(f)))
    for _ in range(NT):
        f = f + tau * (mi @ Q(f))
    qT = np.max(np.abs(Q(f)))
    mT = {k: (M @ f)[i] for k, i in cons.items()}

    print("2. evolution ||Q(f)||_inf: start=%.3e end=%.3e  finite=%s  max|f|=%.3f"
          % (q0, qT, np.all(np.isfinite(f)), np.max(np.abs(f))))
    print("3. conserved-moment drift over %d steps:" % NT)
    for k in cons:
        print("   %-7s drift=%.2e" % (k, abs(mT[k] - m0[k])))


if __name__ == '__main__':
    main()
