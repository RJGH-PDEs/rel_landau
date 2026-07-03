# Petrov-Galerkin Approach for the Relativistic Landau Equation

A spectral discretization of the relativistic (and non-relativistic) Landau collision
operator, using a Petrov-Galerkin method with spherically-defined basis functions
(a generalized-Laguerre radial part times real spherical harmonics).

## Pipeline

The computation is a sequence of stages, each passing data to the next via pickle files:

1. `src/quadrature.py` — build the Gauss-Laguerre × Lebedev quadrature rules.
2. `src/mass_matrix.py` — assemble and invert the Galerkin mass matrix.
3. `src/parallel.py` — compute the collision tensor (parallelized over coefficients).
4. `src/sparse.py` — threshold and sparsify into a list of CSR matrices.
5. `time_evol/time_ev.py` — time-integrate the bilinear form `Q(f, f)`.
6. `plot/plot.py` — reconstruct and plot the solution.

See `CLAUDE.md` for architecture, index conventions, and how to run each stage. The
mathematical derivation is maintained in the companion write-up repository
(`relativistic-landau-paper`).

## Plotting (`plot/plot.py`)

Set the `mode` flag to one of:
- `'single'` — one snapshot or hard-coded coefficient vector
- `'grid'`   — multi-panel grid, one panel per iteration index in `steps`
- `'overlay'`— all snapshots overlaid on one plot (plasma colormap, early→late)

Set `steps` (list of iteration indices or `'equil'`) and `tau` (time step) for the
multi-snapshot modes. Physical time labels are computed as `step × tau`.
Run from `plot/`: `cd plot && python plot.py`.

## Experiments

### Non-relativistic relaxation (2026-07-02)

**Operator:** `nonrel_noncons_dense_n3_q9x7`
(non-relativistic, non-conservative, dense tensor; n=3, 9×7 Gauss-Laguerre×Lebedev quadrature)

**Initial condition:** double-hump profile
- `coeff[0] = 1`   — (k=0, l=0, m=0) Gaussian mode
- `coeff[9] = -0.5` — (k=1, l=0, m=0) first radial correction

**Time integration:** forward Euler, Δt = 0.0001, 10 000 iterations (t_final = 1.0)

**Result:** converged to a smooth isotropic Gaussian by t ≈ 0.15 (step ≈ 1500).
Final state has only three non-negligible coefficients (indices 0, 9, 18 — all l=0),
and `‖Q(f,f)‖ ≤ 10⁻¹²` at the end. Coefficient snapshots saved every 100 steps
in `plot/coeff/`; figures in `plot/figures/`.
