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
