# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a **Petrov-Galerkin spectral method** that computes the **relativistic Landau collision operator**. The discretization uses spherically-defined basis functions: a generalized-Laguerre radial part times a real spherical harmonic, with a Gaussian weight. The authoritative derivation lives in the read-only write-up repo (see below).

**Current branch: `relativistic`** — all work here uses the relativistic energy $E(r) = \sqrt{1+r^2}$. The classical (non-relativistic) experiments live on the `classical` branch.

## Mathematical reference (READ-ONLY)

The authoritative, up-to-date math lives in a **separate git repository** — treat it as read-only, cite it, never modify it:

- Path: `/Users/rjgh/Documents/Research/Latex/Relativistic Landau/` (core file `main.tex`)
- Remote: `https://github.com/RJGH-PDEs/relativistic-landau-paper.git`

Key sections:
- **Weak / Petrov-Galerkin form** — Sec. 3 (Eq. 11), mass matrix Eq. 14/46, collision matrices Eq. 15.
- **Basis & test functions** — Gaussian Eqs. 42–46 (μ_kl in Eq. 43).
- **Collision kernel** — Sec. 2: scalar field Λ (Eq. 9), tensor field 𝕊 (Eq. 10), relativistic energy and gradient.
- **Sparsity / conservation** — Sec. 7: directional rule (`cai`), anisotropic rule (`andrea`); mass/momentum conservation Prop. 1 (Sec. 5.1).

## Key difference from classical branch

In the relativistic case, the energy is $E(p) = \sqrt{1+|p|^2}$ and its gradient is $\nabla E = p/E$. This means:

- **The (1,0,0) test function is NOT skipped** — energy is not conserved by `Phi_simple` relativistically (unlike the classical case where $u = \nabla E_p - \nabla E_q \propto p-q$ is killed by $S$). The `rel=True` flag in `parallel.py` controls this.
- **Conservative energy** uses a Chebyshev polynomial approximation to $\sqrt{1+r^2}$ from `cheby/eh.pkl`. Set `cons=True` in `parallel.py` to use it.

## Running things

Scripts use bare module imports and hard-coded relative paths — run each from its own directory:

```bash
cd src       && python quadrature.py      # build quadrature.pkl and mass_quadrature.pkl
cd src       && python mass_matrix.py     # build mass inverse
cd src       && python parallel.py        # compute collision tensor (expensive; run on TACC)
cd src       && python sparse.py          # threshold + build CSR matrices
cd time_evol && python time_ev.py         # forward-Euler time integration
cd plot      && python plot.py            # 1D z-axis slice
cd plot      && python plot2d.py          # 2D heatmap
```

Output directories (`src/quadrature/`, `src/results/`, `src/mass/`, `src/sparse_operators/`, `plot/coeff/`) are gitignored — create them before running.

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`, `pylebedev`.

## Pipeline (run in this order)

1. **`src/quadrature.py`** — builds GL+Lebedev quadrature rules. Saves `quadrature.pkl` (6D) and `mass_quadrature.pkl` (3D).
2. **`src/mass_matrix.py`** — assembles and inverts the mass matrix. Saves to `src/mass/`.
3. **`src/parallel.py`** — main driver. Computes collision tensor in parallel. Energy set by `rel`/`cons` flags (currently `rel=True`, `cons=False` → exact relativistic energy). Saves to `src/results/`.
4. **`src/sparse.py`** — thresholds, remaps indices, builds list of CSR matrices. Saves to `src/sparse_operators/`.
5. **`time_evol/time_ev.py`** — forward-Euler time integration.
6. **`plot/plot.py`** — 1D z-axis cut; **`plot/plot2d.py`** — 2D heatmap.

## Choosing the physics (set in `src/parallel.py` → `compute_col_tensor`)

- `rel = True, cons = False` — exact relativistic energy $\sqrt{1+r^2}$ **(current setting)**
- `rel = True, cons = True` — Chebyshev polynomial energy from `cheby/eh.pkl`
- `rel = False` — classical energy $\frac{1}{2}r^2$ (use `classical` branch instead)

## Index and data conventions

- **Basis function** identified by `(k, l, m)`: `k` = radial index, `(l,m)` = spherical harmonic. Default `n=3` → 27 basis functions.
- **Flat index:** `ind(k,l,m,n) = n²k + l² + (m+l)`.
- **`select`**: `[[k,l,m], [k1,l1,m1], [k2,l2,m2]]` = `[test fn, f(p), ∇g(q)]`.
- **Quadrature point order:** `[r_p, t_p, p_p, r_q, t_q, p_q]` (set by `unpack_quad`).
- **Sparsity rules** in `src/sparse_rules.py`: `andrea` (anisotropic) and `cai` (directional).

## Experiments

Experiment-specific code lives under `experiments/`, one subdirectory per experiment:

```
experiments/
└── sparsity/               — collision tensor sparsity visualization
    ├── plot_sparsity.py
    └── count_sparsity.py
```

## Chebyshev energy (`cheby/`)

`cheby/ChevInt.py` Chebyshev-interpolates $\sqrt{1+x^2}$ into a plain polynomial, saved as `cheby/eh.pkl`. Used as the "conservative" relativistic energy when `cons=True` in `parallel.py`.

## Gotchas

- Run scripts **from their own directory** — bare imports and relative pickle paths will fail otherwise.
- The `(1,0,0)` test function is **not skipped** in the relativistic case (`rel=True`).
- `sparse.py` reads a specific named file from `results/` — verify the filename matches what `parallel.py` wrote.
- Several scripts have large commented-out test/scratch blocks; the active entry point is `main()`.
