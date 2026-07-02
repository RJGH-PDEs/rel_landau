"""
Centralized naming + metadata I/O for pipeline artifacts.

Single source of truth for the filename tags so the producer (parallel.py) and the
consumers (sparse.py, time_evol/time_ev.py) cannot drift out of sync. Each caller
prepends its own directory prefix (results/, sparse_operators/, ../src/...).

Every pipeline artifact is stored self-describing as {'meta': {...}, 'data': <payload>}
via save_with_meta / load_with_meta, so downstream stages read n and the run config
from the file they load (used for (k,l,m) indexing and state-vector size) instead of
re-typing them.

The operator tag encodes the full run configuration:
  model   : 'rel' | 'nonrel'          (relativistic energy or not)
  energy  : 'cons' | 'noncons'        (discrete/"conservative" energy or not)
  struct  : 'sparse' | 'dense'        (cai/andrea pruning applied or not)
  n        : degrees of freedom (flat dim = n**3)
  q<lag>x<leb> : quadrature order (Gauss-Laguerre x Lebedev)

Example: operator_tag(True, False, True, 3, 9, 7) -> 'rel_noncons_sparse_n3_q9x7'
The mass matrix is physics-independent, so its tag encodes only n + quadrature order:
  mass_tag(3, 9, 7) -> 'mass_inv_n3_q9x7'
"""

import os
import pickle


def operator_tag(rel, cons, sparse, n, n_lag, n_leb):
    model  = 'rel' if rel else 'nonrel'
    energy = 'cons' if cons else 'noncons'
    struct = 'sparse' if sparse else 'dense'
    return f'{model}_{energy}_{struct}_n{n}_q{n_lag}x{n_leb}'


def mass_tag(n, n_lag, n_leb):
    return f'mass_inv_n{n}_q{n_lag}x{n_leb}'


def save_with_meta(path, data, meta):
    """Store an artifact as {'meta': meta, 'data': data}, creating parent dirs."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump({'meta': meta, 'data': data}, file)


def load_with_meta(path):
    """Load an artifact saved by save_with_meta; returns (data, meta).

    Falls back to (obj, None) for legacy bare pickles that predate metadata.
    """
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    if isinstance(obj, dict) and 'meta' in obj and 'data' in obj:
        return obj['data'], obj['meta']
    return obj, None
