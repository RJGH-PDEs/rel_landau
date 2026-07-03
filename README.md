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

## Plotting

Two plotting scripts, both run from `plot/` and share the same `experiment` flag
(`'symmetric'` | `'asymmetric'` | `'zero_momentum'`) that sets the IC label and output
filename automatically. Set `experiment` to match the `ic_mode` used in `time_evol/time_ev.py`.

**`plot/plot.py`** — 1D cut along the polar axis:
- `mode = 'single'` — one snapshot or hard-coded coefficient vector
- `mode = 'grid'`   — multi-panel grid, one panel per step in `steps`
- `mode = 'overlay'`— all snapshots overlaid on one axes (plasma colormap, early→late)

**`plot/plot2d.py`** — 2D heatmap on the x-z plane:
- Evaluates f (with Gaussian weight) on an 80×80 grid; one panel per step in `steps`
- The x-z plane captures the l=1, m=0 (cos θ) asymmetry as a top-bottom difference in z
- Colormap is chosen automatically: `plasma` when f ≥ 0, `RdBu_r` when f goes negative

Run from `plot/`: `cd plot && python plot.py` or `cd plot && python plot2d.py`.

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

### Non-relativistic relaxation — asymmetric IC (2026-07-02)

Same operator and time step as above, with a cos(θ) perturbation that creates a non-zero
net z-momentum:
- `coeff[0] = 1`, `coeff[9] = -0.5` (double-hump base)
- `coeff[2] = 0.1` — (k=0, l=1, m=0) cos(θ) mode, net momentum ≠ 0

**Result:** the l=0 modes relax as before, but the l=1 mode stabilises at a non-zero value
because the scheme conserves the net z-momentum exactly. The final state is an asymmetric
Gaussian shifted along the x-axis. Figure: `relaxation_asymmetric_overlay.png`.

### Non-relativistic relaxation — zero-momentum asymmetric IC (2026-07-02)

Same operator and time step; a cos(θ) asymmetry is introduced but net momentum is set to
zero by adding a second l=1 mode in the cancelling ratio −C₀/C₁ = 0.632456:
- `coeff[0] = 1`, `coeff[9] = -0.5`
- `coeff[2] = 0.1` — (k=0, l=1, m=0)
- `coeff[11] = 0.0632` — (k=1, l=1, m=0), chosen so that ∑ C_k coeff[ind(k,1,0)] = 0

Verified: `(M·f)` at the momentum indices (1,2,3) is machine-zero at t=0.

**Result:** with no conserved momentum to maintain the asymmetry, the l=1 modes decay to
machine-zero and the system relaxes to the same isotropic Gaussian equilibrium as the
symmetric case (same final ‖f‖ ≈ 1.2161). Figure: `relaxation_zero_momentum_overlay.png`.
