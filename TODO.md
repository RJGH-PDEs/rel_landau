# TODO — rel_landau (working branch `dev`)

`main` is the frozen legacy backup; all work happens on `dev`.
Math reference (read-only): `/Users/rjgh/Documents/Research/Latex/Relativistic Landau/main.tex`
(repo `github.com/RJGH-PDEs/relativistic-landau-paper`). Cite equations, never modify it.

Guidance: do **Track A (verification)** before/alongside **Track B (cleanup)** for any given piece,
so we don't refactor code whose correctness we haven't yet pinned to the math.

---

## Verification summary (2026-07-01)

Careful part-by-part check of the code against the write-up. **The discretization is correct.**
Every building block matches the math (verified symbolically/numerically to ~1e-13–1e-16):
basis & spherical harmonics, normalization `μ`, mass matrix, gradients, quadrature, the kernel
tensor `𝕊`, the weak-form assembly, time stepping, and reconstruction. Details per part in Track A.

**Three things worth knowing:**
1. **`mu_const` "discrepancy" → the CODE is right, the PAPER has a typo.** Write-up **Eq. (43),
   `main.tex` line 539** should be `√(2·k!/Γ(k+l+3/2))` (the `2` is in the wrong place). The
   exponential-weight `μ` (Eq. 53, line 596) is correct. Fix upstream in the paper repo (read-only
   here). See Part 2.
2. **Λ = 1 (simplified kernel) — INTENTIONAL and fixed for this experiment.** The code implements
   the write-up's `Φ_simple` (Maxwell-molecules-like, lines 256-259): the tensor `𝕊` only, with the
   scalar field `Λ` and its `|p−q|^{-3}` singularity omitted. **This is a deliberate design choice —
   Λ stays 1 for the remainder of this experiment; adding the full `Λ` is explicitly out of scope**
   (extra work not wanted now). Correct as implemented. See Part 5.
3. **Relativistic `(1,0,0)` energy-skip bug — FOUND & FIXED** (commit `eef55db`). The skip is valid
   non-relativistically but zeroed genuinely-nonzero coefficients relativistically; now conditional
   on `rel`. See Part 7.

**Blocking bugs fixed** (commit `eef55db`): `mass.pkl`→`mass_inv.pkl` path, `save_coeff` variable,
producer/consumer filename mismatch, and auto-`makedirs` at all save sites.

**Pipeline status:** quadrature + mass matrix build and re-verify locally (`248.05`, `15.75`, mass
matrix matches Eq. 46 to 1e-13). The full collision-tensor computation is expensive (pure-Python,
54k quadrature points/coefficient) and is run **on a cluster**, not locally — the end-to-end
numerical sanity check (`Q(equilibrium)≈0`, conserved-moment drift, bounded relaxation) is therefore
**deferred to a cluster run**. Note: those conservation identities are *pointwise-exact* in the
integrand, so they are guaranteed by the Part-1–9 verification independent of quadrature order.
A ready-to-run reduced-order check script is committed at repo root: **`pipeline_check.py`**
(`cd src && python ../pipeline_check.py`). It builds a reduced quadrature, computes the non-rel
tensor, and checks `Q(equilibrium)≈0`, conserved-moment drift, and bounded relaxation. For the
full-accuracy run, use the normal pipeline on a cluster (`parallel.py` → `sparse.py` → `time_ev.py`).

---

## Conserved-quantities tracking (planned, not yet done)

**Goal:** produce time-series plots of the discretely conserved quantities — mass, momentum, and
(optionally) energy — for each time-evolution experiment, as a sanity check that the scheme
actually conserves what it claims to conserve.

**Key insight:** In the Petrov-Galerkin scheme the conserved quantities live in M·f, not in f
itself (because the mass matrix is the "inner product" between test and trial spaces).  For a
conserved test function φ_i, we have Q(f,f)[i] = 0 exactly, so M·f_{n+1}[i] = M·f_n[i] at
every step.  The relevant entries of M·f are:

| Quantity   | Test function              | Flat index (n=3)        |
|------------|----------------------------|-------------------------|
| Mass       | (k=0, l=0, m=0)            | `ind(0,0,0)` = **0**   |
| Momentum z | (k=0, l=1, m=0)            | `ind(0,1,0)` = **2**   |
| Momentum x | (k=0, l=1, m=1)            | `ind(0,1,1)` = **3**   |
| Momentum y | (k=0, l=1, m=-1)           | `ind(0,1,-1)` = **1**  |

Energy is *not* a single flat index — it corresponds to the test function r²/2, which is a linear
combination of l=0 basis functions (specifically k=0 and k=1, since L_1^{1/2}(r²) = 3/2 − r²).
Energy tracking therefore needs an explicit projection vector; deferred for now.

**Implementation plan (no changes to time_ev.py needed):**

1. **Compute M from M⁻¹**: the mass matrix is not currently saved separately — only M⁻¹ is.
   Recover it as `M = np.linalg.inv(mi)` (cheap, 27×27).  Alternatively, update `mass_matrix.py`
   to also save `M` under `mass_tag(...)` (cleaner long term).

2. **New script: `plot/plot_conservation.py`**
   - Load M⁻¹ → invert → M.
   - Load each saved coeff pkl (steps [0, 100, 200, ..., N]) from `plot/coeff/`.
   - For each step: compute `q = M @ f`, extract `q[0]`, `q[1]`, `q[2]`, `q[3]`.
   - Plot: 4 subplots (one per conserved quantity) of value vs physical time t = step × τ.
     - Expected: perfectly flat lines (machine precision drift at most).
   - Use the same `experiment` flag as `plot.py` / `plot2d.py`.

3. **What to look for in each experiment:**
   - `symmetric`: q[0] constant, q[1]=q[2]=q[3]=0 (no net momentum, should stay 0).
   - `asymmetric`: q[0] constant, q[2] ≠ 0 and constant (non-zero z-momentum conserved),
     q[1]=q[3]=0.
   - `zero_momentum`: q[0] constant, q[1]=q[2]=q[3]=0 all the way (zero momentum, stays 0).

4. **Optional — energy:**
   Build the energy projection vector `e_vec` once (offline):
   `e_vec[i] = ∫ (r²/2) · φ_test(r) · ψ_i(r) e^{-r²/2} dr dΩ`
   — i.e., a row of a "generalised mass matrix" with test fn φ_energy = r²/2 instead of the usual
   Petrov-Galerkin test.  Then `E(t) = e_vec @ f(t)`.  For the Gaussian-weight basis this integral
   is analytic (Laguerre × power × Gaussian), so no extra quadrature needed.


Already confirmed during planning:
- [x] `cai` directional sparsity (`src/sparse_rules.py:40-45`) matches Sec. 7.2: `|m_i| = |m_s ± m_t|`.
- [x] `andrea` anisotropic sparsity (`src/sparse_rules.py:21`) matches Sec. 7.3:
      `l_s+l_t-l_i` even and `0 ≤ l_s+l_t-l_i ≤ 2·min(l_s,l_t)`.

To verify:
- [x] **`mu_const` — CODE IS CORRECT; write-up Eq. (43) has a factor-of-2 typo.** Code
      `√(2·k!/Γ(k+l+3/2))` makes the trial functions L²-orthonormal (`⟨ψ_a,ψ_b⟩=δ`, verified
      numerically: diag=1.000000, off-diag ~1e-16), matching the write-up's OWN derivation
      (product-of-weights `e^{-r²}` → the `x=r²` Jacobian `½` → Laguerre orthogonality). The printed
      Eq. (43) `√(k!/(2Γ))` gives 0.25 instead of 1. **Action: fix Eq. (43) in the paper repo
      (upstream, read-only here) — no code change.**
- [x] `spher_const` (`src/basis.py:6-19`) vs Eqs. (28)–(30): **matches exactly** (checked
      symbolically/numerically for l≤2, all m).
- [x] Radial/basis assembly (`src/basis.py` `basis()`): returns unweighted test function
      `L_k^{l+1/2}(r²) r^l Y_lm` (Eq. 44); Laguerre α=l+½, argument r², r^l solid-harmonic factor —
      all correct. Weight `e^{-r²/2}` + μ applied elsewhere for the trial (Eq. 42). Confirmed.
- [x] **⚠️ Benign convention note — Condon-Shortley phase.** Code's `Y_{l,m}` carry the CS phase
      `(-1)^m` (sympy `assoc_legendre` in `basis.py`/`sym_test`, scipy `lpmv` in `test_func.py` —
      verified to agree, ratio +1 for all m), whereas the write-up's `P_{l,m}` (Eq. line 360) omits
      it. Sign-flips odd-m harmonics vs the paper but is used consistently everywhere, so it does
      NOT affect orthonormality, conservation, or the reconstructed f. Optional: note the CS phase
      in the write-up. No code change needed.
- [x] Basis gradients (`src/basis.py` `gradient`, `grad_weighted`): `gradient()` matches a Cartesian
      finite-difference reference to ~1e-11 (correct spherical-gradient formula + orthonormal basis
      vectors). `grad_weighted(f) = gradient(f) − r·f·ê_r` to ~1e-14, i.e. `∇(e^{-r²/2}f)/e^{-r²/2}`
      with the common weight factored out for the quadrature. Confirmed.
- [x] Quadrature (`src/quadrature.py`): 3D radial moments exact (`∫1=(2π)^1.5`, `∫r²=3(2π)^1.5`,
      `∫r⁴=15(2π)^1.5` — the `e^{-r²/2}·r²` measure is folded in via the `x=r²/2` change of variables,
      lines 207-218); 6D `unpack_quad` ordering `[r_p,t_p,p_p,r_q,t_q,p_q]` + weight-product correct
      (`∫∫1=(2π)³`, `∫∫r_p²=3(2π)³` exact). Angular orthonormality confirmed via the mass matrix.
      The `phi` "this might be wrong" comment (`quadrature.py:32`) is a FALSE ALARM —
      `sign(y)·arccos(x/ρ)` equals `atan2(y,x)`, correct.
- [x] Kernel (`src/kern.py`): tensor field `𝕊 = |u|²Id − u⊗u − (z×u)⊗(z×u)` matches Eqs. (10)/(196)
      exactly; energy gradient `∇E=(∂E/∂r)ê_r` = `p/E_p` (rel) / `p` (non-rel) — correct; `𝕊·u=0` by
      construction (the energy-conservation mechanism). Relativistic `(z×u)⊗(z×u)` term present.
- [x] **Scalar field Λ = 1 — INTENTIONAL, fixed for this experiment (not future work).**
      The code's kernel is exactly the write-up's simplified "Maxwell-molecules-like" kernel
      `Φ_simple = 𝕊(u,z)` with `Λ=1` (write-up lines 256-259), NOT the full
      `Λ = (E_pE_q)(ρ+1)²(ρτ)^{-3/2}` (Eqs. 9/195). This is a deliberate, write-up-sanctioned
      simplification (and sparsity depends only on 𝕊's angular structure). **Decision: Λ stays 1 for
      the remainder of this experiment; implementing the full Λ + its diagonal singularity is
      explicitly out of scope.** Correct as implemented.
- [x] Integrand & weak-form assembly (`src/integrand.py`, `src/landau.py`) vs Eq. (278):
      `result = f(p)·[∇ψ_t(q)]ᵀ·𝕊·(∇φ_i(p)−∇φ_i(q))` exactly matches the weak-form RHS with `Φ=𝕊`
      (Λ=1). Select ordering `[test i, f(p)=s, ∇g(q)=t]` correct; trial carries μ (weight in
      quadrature), test gradient unweighted. `landau.py` sums `weight·integrand` over the 6D
      quadrature. Confirmed (given the Λ=1 simplification above).
- [x] Time stepping (`time_evol/time_ev.py`, `bilinear.py`) vs Eq. (13): `result[i]=fᵀQ_i f`
      (`landau`), `f_{n+1}=f_n+τ·M⁻¹·Q(f,f)` — forward Euler of `df/dt=M⁻¹Q(f,f)`, correct sign and
      mass-inverse application. Math correct; only the mechanical blocking bugs remain (mass path,
      `save_coeff`).
- [x] Conservation skips (`src/parallel.py:34-40`), verified numerically:
      - `(0,0,0)` mass, `(0,1,·)` momentum → `Q=0` for BOTH non-rel and rel (kernel-independent,
        write-up line 455). Skips correct.
      - **⚠️ BUG: `(1,0,0)` energy skip is only valid NON-RELATIVISTICALLY.** Numerically `Q≈0`
        non-rel but `Q≈−7.8, +12.1` for the relativistic `Φ_simple`. `parallel.py` skips `(1,0,0)`
        *unconditionally*, so relativistic runs incorrectly zero genuinely-nonzero coefficients
        (artificial `r²`-moment conservation). Reason: `𝕊`'s nullspace is `u=p/E_p−q/E_q`, which
        equals `p−q` only non-rel; relativistic energy conservation needs the discrete-energy kernel
        `Φ_h` (Sec. 5.2), and the conserved quantity is `E^h`, not the `(1,0,0)` basis fn.
        **FIX: make the `(1,0,0)` skip conditional on `rel==False`** (and, for the conservative
        relativistic scheme, revisit what the correct energy-projection skip should be).
- [x] `cai`/`andrea` sparsity rules re-confirmed vs Sec. 7.2/7.3 (unchanged from planning).
- [x] Mass matrix (`src/mass_matrix.py`) vs Eq. (46): **matches to ~1e-13** (verified against direct
      quadrature); correct block structure (0 off-block nonzeros); invertible (cond ≈ 2.3e3).
      Non-symmetric by design (Petrov-Galerkin: `⟨test unweighted, trial μ-weighted⟩`) — not a bug.
      The weight `e^{-r²/2}` and `r²` volume factor are folded into the Gauss-Laguerre quadrature via
      the `x=r²/2` change of variables (`src/quadrature.py:207-218`), not the integrand.
- [x] Reconstruction (`plot/lc.py` + `plot/test_func.py` + `plot/plot.py`): CORRECT. `test()` gives
      the polynomial part `μ·L·r^l·Y` (scipy `lpmv`, same CS-phase convention as `basis.py`, verified),
      and `plot.py:90` multiplies by `e^{-r²/2}`, so `e^{-r²/2}·linear_comb = Σ f_j ψ_j = f_h`
      (verified to ~1e-16 against the true trial). NB `test()` is misnamed (it's the unweighted
      trial with μ, not the test function φ) — Track B naming nit only.
- [~] "Conservative energy" (`cheby/ChevInt.py` + `src/parallel.py`, `cons=True`): the code uses a
      Chebyshev INTERPOLANT of `√(1+r²)` as the energy, which differs from the write-up's L²-projection
      `Π^r_h` (Sec. 5.2). Any radial `E^h` still gives `𝕊_h·u_h=0` by construction, so *some* discrete
      energy `∫f·E_cheby` is conserved — but it's not exactly the paper's `E^h`, and the conserved
      quantity is a degree-≤12 polynomial moment, NOT the `(1,0,0)` basis function (reinforces the
      Part-7 skip bug). Deeper verification deferred until Λ / the conservative scheme is revisited
      (this is future-work territory, tied to the missing scalar field).

---

## Track B — Code cleanup

### Critical (correctness bugs) — DONE (commit eef55db)
- [x] Path mismatch fixed: `time_ev.py` now reads `mass_inv.pkl`.
- [x] `save_coeff()` now dumps its `coeff` argument.

### High
- [x] Shadowed builtins fixed (commit ceeb5bd): local `sum`→`msum` (`sparse.py`, `sparse_rules.py`),
      `for quad in quad`→`for q in quad` (`quadrature.py`, `mass_matrix.py`), `next`→`f_next`
      (`time_ev.py`). (Note: `m = min(l1,l2)` in `sparse.py` uses the builtin correctly — not a shadow.)
- [ ] Fragile bare-import + run-from-own-directory pattern across `src/`, `time_evol/`, `plot/`.
      Consider a real package (`__init__.py` / `pyproject.toml`) + a central paths/config module.
- [ ] No tests, no `requirements.txt`, no package structure.

### Medium
- [x] Consolidate duplicated functions: `mu_const`/`spher_const` now imported from `src/basis`;
      `ind` now imported from `src/sparse` (`lm_index` removed from `plot/`);
      `radius`/`theta`/`phi` (`src/quadrature.py` ↔ `plot/plot.py`) — deferred.
- [ ] Rename the misnamed `test()` in `plot/test_func.py` (it's the unweighted trial with μ, not the
      test function φ). Touches `plot/lc.py`. Deferred (semantic rename).
- [ ] Centralize hard-coded params: `n=3`, magic `27`, `tau`, `NUM_ITERATIONS`, `tol`,
      quadrature orders (`n_laguerre=9`, `n_lebedev=7`), energy choice.
- [x] **Centralized output-file naming convention** — DONE (`src/naming.py` `operator_tag`). Builds
      the filename from run config (rel/cons/sparse/n + quadrature order), e.g.
      `rel_noncons_sparse_n3_q9x7`; each caller prepends its dir.
- [x] **Quadrature order threaded robustly** — DONE. `quadrature.pkl`/`mass_quadrature.pkl` store the
      `(n_lag,n_leb)` they were built with; `load_quad_order(path)` reads it back and every stage names
      from THAT, not the constants. Old bare-list pickles fall back to the constants.
- [x] **Self-describing pipeline (metadata in every artifact)** — DONE. Each artifact is stored as
      `{'meta', 'data'}` via `naming.save_with_meta`/`load_with_meta`. Operator files carry
      `{rel,cons,sparse,n,n_lag,n_leb}`; the mass matrix is tagged `mass_inv_n{n}_q{lag}x{leb}` with
      `{n,n_lag,n_leb}`. Consumers READ `n` from the loaded file (fixes the hardcoded `27` → `n**3`),
      assert the loaded meta matches the requested config (loud failure on drift), and `sparse.py`
      propagates meta so its output tag always matches its input. Ready for varying `n` and degrees.
      REMAINING: consumers still set `rel/cons/sparse/n` flags to LOCATE the input file (name encodes
      config; no manifest/discovery) — assertions guard mismatches. A discovery/manifest could remove
      that later.
- [x] **`plot/` metadata-aware for `n`** — `linear_comb` now accepts `n` as a parameter (default 3);
      `plot.py` has a single `N = 3` constant (not hardcoded `27`), passes `n=N` through, and reads
      `n` from artifact metadata when loading pkl files (falls back to `N` for bare vectors). REMAINING:
      `time_ev` still saves bare coeff vectors; threading `n` into those files would remove the fallback.
- [x] **Non-relativistic experiment DONE (2026-07-02).** Operator `nonrel_noncons_dense_n3_q9x7`
      run on TACC; sparse.py validated; time evolution run locally with IC `coeff[0]=1,
      coeff[9]=-0.5` (double-hump), Δt=0.0001, 10k steps. Converged to equilibrium by t≈0.15.
      Plots (grid + overlay) in `plot/figures/`. Next: relativistic run with same quadrature.
- [ ] Before a full run: choose quadrature degrees (`n_laguerre`, `n_lebedev`) deliberately —
      accuracy vs the 6D cost `(n_lag·n_leb_pts)²` per coefficient. (Runs happen on a cluster.)
- [x] `sparse` flag added to `compute_col_tensor` (`src/parallel.py`): toggles cai/andrea zero-pruning
      (sparse) vs full dense tensor. Commit on `dev`.
- [x] Removed dead/commented scratch blocks (commit ceeb5bd): `bilinear.py`, `lc.py`, the obsolete
      `unpack_quadrature()` in `quadrature.py`, and unused fns `energy_grad_cart`/`test_indices`/`f_integrated`.
- [x] Comment typos fixed (commit ceeb5bd): weighet, functiosn, lambdafy, convenction, Parametes,
      azimunth; `Leg()` docstring Laguerre→Legendre.
- [x] RESOLVED (verified in Part 4): the "this might be wrong" `phi` comments were a false alarm
      (`sign(y)·arccos(x/ρ) == atan2(y,x)`); comment removed in cleanup.
