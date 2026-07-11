# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a **Petrov-Galerkin spectral method** that computes the (relativistic and non-relativistic) **Landau collision operator**. The discretization uses spherically-defined basis functions: a generalized-Laguerre radial part times a real spherical harmonic, with a Gaussian weight. The authoritative derivation lives in the read-only write-up repo (see below). This is exploratory research code, not a packaged library — expect WIP scripts, commented-out blocks, and hard-coded parameters/paths.

The write-up defines **two basis families**: a Gaussian-weight basis (`e^{-r²/2}`, write-up Sec. 5) and an exponential-weight basis (`e^{-r/2}`, Sec. 6). **The current code implements the Gaussian basis.** The exponential basis is the intended next direction (cf. the last commit, "will move on to different basis").

## Mathematical reference (READ-ONLY)

The authoritative, up-to-date math lives in a **separate git repository** — treat it as read-only, cite it, never modify it:

- Path: `/Users/rjgh/Documents/Research/Latex/Relativistic Landau/` (core file `main.tex`)
- Remote: `https://github.com/RJGH-PDEs/relativistic-landau-paper.git`

Key sections to cite when verifying code against the math:
- **Weak / Petrov-Galerkin form** — Sec. 3 (Eq. 11), mass matrix Eq. 14/46, collision matrices Eq. 15.
- **Basis & test functions** — Gaussian Eqs. 42–46 (μ_kl in Eq. 43), exponential Eqs. 52–57; spherical harmonics Eqs. 28–31.
- **Collision kernel** — Sec. 2: scalar field Λ (Eq. 9), tensor field 𝕊 (Eq. 10), relativistic energy and gradient; discrete-energy "conservative" kernel Sec. 5.2 (Eqs. 35–37).
- **Sparsity / conservation** — Sec. 7: directional rule (`cai`, Sec. 7.2), anisotropic rule (`andrea`, Sec. 7.3); mass/momentum conservation Prop. 1 (Sec. 5.1).

## Running things

There is no build system, test runner, or dependency manifest. Each script is run directly with Python and most do their real work in a `main()` / `if __name__ == "__main__"` block.

**Critical: scripts use bare module imports** (e.g. `from kern import kernel`, `from bilinear import landau`) and **hard-coded relative `.pkl` paths** (e.g. `./quadrature/...`, `../src/mass/...`). You must run each script *with the working directory set to the script's own folder*, or imports and file I/O will fail:

```bash
cd src       && python parallel.py     # NOT python src/parallel.py
cd time_evol && python time_ev.py
cd plot      && python plot.py
```

The output directories consumed/produced by the pipeline (`src/quadrature/`, `src/results/`, `src/mass/`, `src/sparse_operators/`, `plot/coeff/`, `plot/figures/`) are **gitignored** and may not exist on a fresh checkout — create them before running, since `pickle.dump` will not create parent dirs.

Dependencies (install manually): `numpy`, `scipy`, `sympy`, `matplotlib`, `pylebedev`.

## Pipeline (run in this order)

The whole system is a sequence of scripts that pass data via pickle files:

1. **`src/quadrature.py`** — builds the numerical integration rules: generalized-Laguerre nodes (radial, `roots_genlaguerre`) tensored with Lebedev nodes (angular, `PyLebedev`). Saves `quadrature.pkl` (6D, for the collision operator over points p and q) and `mass_quadrature.pkl` (3D, for the mass matrix) into `src/quadrature/`.
2. **`src/mass_matrix.py`** — assembles the Galerkin mass matrix `⟨test_i, basis_j⟩`, inverts it, and saves the inverse to `src/mass/`. Run once before time evolution.
3. **`src/parallel.py`** — the main driver. Computes every collision-tensor coefficient `Q[test][f][g]` in parallel via `multiprocessing.Pool`, applying sparsity rules to skip provably-zero entries and skipping conservation-law test functions. Saves a flat list of `[select, value]` to `src/results/`. Calls into `landau.py` → `integrand.py` → `kern.py` + `basis.py`.
4. **`src/sparse.py`** — post-processes that flat list: thresholds non-zeros by tolerance, remaps `[k,l,m]` triples to flat indices, and builds the collision tensor as a **list of `scipy` CSR matrices** (one matrix per test function). Saves to `src/sparse_operators/`.
5. **`time_evol/time_ev.py`** — forward-Euler time integration of the bilinear form `Q(f,f)` (see `bilinear.py`: `result[i] = f @ (tensor[i] @ f)`), multiplied by the mass inverse. Dumps the coefficient vector at each step to `plot/coeff/`.
6. **`plot/plot.py`** (+ `plot/lc.py`, `plot/test_func.py`) — reconstructs `f(r,θ,φ)` from saved coefficient vectors as a linear combination of basis functions and plots it.

Supporting / analysis:
- **`cheby/ChevInt.py`** — Chebyshev-interpolates the relativistic energy `√(1+x²)` into a plain polynomial, saved as `cheby/eh.pkl`. Used as the "conservative" relativistic energy in `parallel.py`.

## Index and data conventions (must stay consistent across files)

These conventions are duplicated by hand in several files (`basis.py`, `sparse.py`, `mass_matrix.py`, `parallel.py`, `plot/lc.py`) — changing one means changing all.

- **Basis function** is identified by `(k, l, m)`: `k` = radial (Laguerre) index, `(l, m)` = spherical harmonic. Default `n = 3` → flat dimension `n³ = 27`.
- **Flat index:** `ind(k, l, m, n) = n*n*k + l*l + (m + l)`.
- **`select`** is the coefficient address `[[k,l,m], [k1,l1,m1], [k2,l2,m2]]` = `[test function, f(p), ∇g(q)]`. This ordering is assumed everywhere (kernel, integrand, sparsity rules).
- **Quadrature point order:** `points = [r_p, t_p, p_p, r_q, t_q, p_q]` (set by `unpack_quad`); the integrand depends on this exact ordering.
- **Sparsity rules** in `src/sparse_rules.py`: `andrea` (anisotropic, an `l`-selection rule) and `cai` (directional, an `m`-selection rule). `parallel.py` uses them to prune the coefficient list before computing; `sparse.py`'s `analyse()` re-checks how often they're violated in the computed result.

## Choosing the physics (set in `src/parallel.py` → `compute_col_tensor`)

The energy function defines the model and is selected by the `rel` / `cons` flags:
- non-relativistic: `energy = (1/2) * r**2`
- relativistic: `energy = sqrt(1 + r**2)`
- "conservative" relativistic: polynomial energy loaded from `cheby/eh.pkl`

`n` (degrees of freedom) and the output `file_name` are also hard-coded here. The kernel (`kern.py`) takes the energy gradient and builds the Landau projection operator from it.

## Experiments

Experiment-specific code, figures, and documentation live under `experiments/`, one subdirectory per experiment. General pipeline code (`src/`, `time_evol/`, `plot/`) stays in place.

```
experiments/
├── lemou_benchmark/        — Maxwellian-molecules exact analytical validation (Villani/Lemou)
│   ├── lemou_benchmark.md  — full documentation of the benchmark (moved from docs/)
│   ├── lemou_ic.py         — IC coefficient computation (moved from time_evol/)
│   ├── plot_lemou.py       — log-scale decay + radial profiles
│   ├── plot_lemou_coeffs.py — coefficient evolution α_{k,0,0}(t)
│   ├── plot_lemou_compare.py — analytical vs numerical side-by-side
│   ├── plot_lemou_villani_time.py — same data in Villani's rescaled time t̃=2t
│   └── figures/            — lemou_benchmark.png, lemou_coeffs.png, lemou_compare.png, lemou_villani_time.png
├── relaxation_symmetric/   — double-hump IC (coeff[0]=1, coeff[9]=-0.5), isotropic relaxation
│   └── figures/            — relaxation_overlay.png, relaxation_2d_xz*.png, conservation_symmetric.png, ic_candidates.png
├── relaxation_asymmetric/  — cos(θ) perturbation (coeff[2]=0.1), nonzero net z-momentum conserved
│   └── figures/            — relaxation_2d_xz_asymmetric.png, conservation_asymmetric.png, ic_asymmetric.png
├── relaxation_zero_momentum/ — cos(θ) perturbation, zero net momentum (cancelling l=1 modes)
│   └── figures/            — relaxation_zero_momentum_overlay.png, conservation_zero_momentum.png, ic_zero_momentum.png
└── sparsity/               — collision tensor sparsity visualization
    ├── plot_sparsity.py    — (moved from plot/)
    └── figures/            — sparsity_nonrel_noncons_dense_n3_q9x7.png
```

**⚠️ Stale relative paths in moved scripts:** All scripts under `experiments/` were moved from `plot/` or `time_evol/` and still carry their original relative paths (e.g. `sys.path.insert(0, '../src')`, `COEFF_DIR = './coeff'`). These will need updating when the scripts are next run — fix paths to point to the repo root, e.g. `../../src` and `../../plot/coeff` respectively.

## Gotchas

- **Path mismatches exist between stages.** e.g. `mass_matrix.py` saves to `./mass/mass_inv.pkl`, while `time_ev.py` loads `../src/mass/mass.pkl`; `parallel.py` writes to `results/` while `sparse.py` reads a specific named file. Verify the actual filenames when wiring stages together — do not assume they already line up.
- Several scripts have large commented-out test/scratch blocks at the bottom (`bilinear.py`, `parallel.py`); the active entry point is the uncommented `main()`.
