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
- [x] **⚠️ MAIN MISSING PHYSICS: scalar field Λ is NOT implemented — code computes `Φ_simple` (Λ=1).**
      The code's kernel is exactly the write-up's simplified "Maxwell-molecules-like" kernel
      `Φ_simple = 𝕊(u,z)` with `Λ=1` (write-up lines 256-259), NOT the full
      `Λ = (E_pE_q)(ρ+1)²(ρτ)^{-3/2}` (Eqs. 9/195) with its `|p−q|^{-3}` singularity. This is a
      deliberate, write-up-sanctioned simplification (sparsity depends only on 𝕊's angular structure),
      but means the code does NOT yet compute the physically complete Landau operator. **Adding Λ
      (and handling its diagonal singularity in the 6D quadrature) is the key future-work item.**
- [x] Integrand & weak-form assembly (`src/integrand.py`, `src/landau.py`) vs Eq. (278):
      `result = f(p)·[∇ψ_t(q)]ᵀ·𝕊·(∇φ_i(p)−∇φ_i(q))` exactly matches the weak-form RHS with `Φ=𝕊`
      (Λ=1). Select ordering `[test i, f(p)=s, ∇g(q)=t]` correct; trial carries μ (weight in
      quadrature), test gradient unweighted. `landau.py` sums `weight·integrand` over the 6D
      quadrature. Confirmed (given the Λ=1 simplification above).
- [ ] Conservation laws: skipped basis functions `(0,0,0)`, `(0,1,·)`, `(1,0,0)`
      (`src/parallel.py:34-40`) vs Prop. 1 (mass/momentum) and Sec. 5.2 (discrete energy).
- [x] Mass matrix (`src/mass_matrix.py`) vs Eq. (46): **matches to ~1e-13** (verified against direct
      quadrature); correct block structure (0 off-block nonzeros); invertible (cond ≈ 2.3e3).
      Non-symmetric by design (Petrov-Galerkin: `⟨test unweighted, trial μ-weighted⟩`) — not a bug.
      The weight `e^{-r²/2}` and `r²` volume factor are folded into the Gauss-Laguerre quadrature via
      the `x=r²/2` change of variables (`src/quadrature.py:207-218`), not the integrand.
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
- [x] RESOLVED (verified in Part 4): the "this might be wrong" `phi` comments (`src/quadrature.py:32`,
      `plot/plot.py:42`) are a false alarm — `sign(y)·arccos(x/ρ) == atan2(y,x)`. Safe to just delete
      the misleading comment when cleaning.
