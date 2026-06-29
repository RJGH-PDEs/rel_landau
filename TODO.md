# TODO — rel_landau (working branch `dev`)

`main` is the frozen legacy backup; all work happens on `dev`.
Math reference (read-only): `/Users/rjgh/Documents/Research/Latex/Relativistic Landau/main.tex`
(repo `github.com/RJGH-PDEs/relativistic-landau-paper`). Cite equations, never modify it.

Guidance: do **Track A (verification)** before/alongside **Track B (cleanup)** for any given piece,
so we don't refactor code whose correctness we haven't yet pinned to the math.

---

## Track A — Math ↔ code verification (against `main.tex`)

Already confirmed during planning:
- [x] `cai` directional sparsity (`src/sparse_rules.py:40-45`) matches Sec. 7.2: `|m_i| = |m_s ± m_t|`.
- [x] `andrea` anisotropic sparsity (`src/sparse_rules.py:21`) matches Sec. 7.3:
      `l_s+l_t-l_i` even and `0 ≤ l_s+l_t-l_i ≤ 2·min(l_s,l_t)`.

To verify:
- [ ] **⚠️ `mu_const` factor-of-2** (`src/basis.py:28-30`): code = `√(2·k!/Γ(k+l+3/2))`,
      Eq. (43) = `√(k!/(2·Γ(k+l+3/2)))`. Decide whether a weight convention
      (`e^{-r²/2}` vs `e^{-r²}`) reconciles it, or the code is wrong.
- [ ] `spher_const` (`src/basis.py:6-19`) vs spherical-harmonic constants Eqs. (28)–(30).
- [ ] Radial trial basis `e^{-r²/2} L_k^{l+1/2}(r²) r^l` vs Eq. (42); test functions unweighted
      (`L_k^{l+1/2}(r²) r^l Y_lm`) vs Eq. (44) — confirm the Petrov-Galerkin weighting split.
- [ ] Kernel (`src/kern.py`): scalar field Λ = `(E_p E_q)(ρ+1)²(ρτ)^{-3/2}` vs Eq. (9);
      tensor field `|u|²Id − u⊗u − (z×u)⊗(z×u)` vs Eq. (10); energy gradient `p/√(1+p²)` vs Sec. 2;
      presence of the relativistic `(z×u)⊗(z×u)` term.
- [ ] Conservation laws: skipped basis functions `(0,0,0)`, `(0,1,·)`, `(1,0,0)`
      (`src/parallel.py:34-40`) vs Prop. 1 (mass/momentum) and Sec. 5.2 (discrete energy).
- [ ] Mass matrix block structure (nonzero only if `l_i=l_j`, `m_i=m_j`) vs Eq. (46).
- [ ] "Conservative energy" Chebyshev path (`cheby/ChevInt.py` + `src/parallel.py`) vs the
      discrete-energy projection Sec. 5.2, Eqs. (35)–(37).

---

## Track B — Code cleanup

### Critical (correctness bugs)
- [ ] Path mismatch: `time_evol/time_ev.py:27` reads `../src/mass/mass.pkl`, but
      `src/mass_matrix.py:136` writes `mass_inv.pkl` → FileNotFoundError.
- [ ] `save_coeff()` (`time_evol/time_ev.py:10-19`) dumps the global `f` instead of its `coeff` arg.

### High
- [ ] Shadowed builtins: `sum`/`min` (`src/sparse.py:82,95`, `src/sparse_rules.py:42`);
      `for quad in quad:` (`src/quadrature.py:376`, `src/mass_matrix.py:64`).
- [ ] Fragile bare-import + run-from-own-directory pattern across `src/`, `time_evol/`, `plot/`.
      Consider a real package (`__init__.py` / `pyproject.toml`) + a central paths/config module.
- [ ] No tests, no `requirements.txt`, no package structure.

### Medium
- [ ] Consolidate duplicated functions: `mu_const`/`spher_const` (`src/basis.py` ↔
      `plot/test_func.py`), `ind`/`lm_index` (`src/sparse.py` ↔ `plot/lc.py`),
      `radius`/`theta`/`phi` (`src/quadrature.py` ↔ `plot/plot.py`).
- [ ] Centralize hard-coded params: `n=3`, magic `27`, `tau`, `NUM_ITERATIONS`, `tol`,
      quadrature orders (`n_laguerre=9`, `n_lebedev=7`), energy choice.
- [ ] Centralize the inconsistent hard-coded relative output paths across stages.
- [ ] Remove dead/commented scratch blocks (`time_evol/bilinear.py:81-139`, `plot/lc.py:32-47`,
      `src/quadrature.py:149-191`).
- [ ] Resolve the "this might be wrong" coordinate comments (`src/quadrature.py:32`,
      `plot/plot.py:42`) — likely a `phi` quadrant/sign question; verify against the coordinate
      convention in the write-up (Sec. 3.3).
