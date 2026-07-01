"""
Centralized naming for collision-operator files.

Single source of truth for the filename tag so the producer (parallel.py) and the
consumers (sparse.py, time_evol/time_ev.py) cannot drift out of sync. Each caller
prepends its own directory prefix (results/, sparse_operators/, ../src/...).

The tag encodes the full run configuration:
  model   : 'rel' | 'nonrel'          (relativistic energy or not)
  energy  : 'cons' | 'noncons'        (discrete/"conservative" energy or not)
  struct  : 'sparse' | 'dense'        (cai/andrea pruning applied or not)
  n        : degrees of freedom (flat dim = n**3)
  q<lag>x<leb> : quadrature order (Gauss-Laguerre x Lebedev)

Example: operator_tag(True, False, True, 3, 9, 7) -> 'rel_noncons_sparse_n3_q9x7'
"""


def operator_tag(rel, cons, sparse, n, n_lag, n_leb):
    model  = 'rel' if rel else 'nonrel'
    energy = 'cons' if cons else 'noncons'
    struct = 'sparse' if sparse else 'dense'
    return f'{model}_{energy}_{struct}_n{n}_q{n_lag}x{n_leb}'
