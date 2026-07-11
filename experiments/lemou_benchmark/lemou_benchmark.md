# Lemou Benchmark: Exact Analytical Solution for the Isotropic Landau Equation

**Reference:** C. Villani, "On the Spatially Homogeneous Landau Equation for Maxwellian Molecules"
(`docs/document.pdf`), Section 3.

---

## Key Findings

### 1. Our kernel IS the Maxwellian-molecules kernel

Villani's Landau kernel for Maxwellian molecules (γ = 0, Λ = 1) is:

```
a_ij(z) = |z|² δ_ij − z_i z_j
```

Our code (`src/kern.py`, line 46) computes, for non-relativistic energy E = r²/2:

```python
u = ep - eq          # u = ∇_p E − ∇_q E = p − q
kern = u.dot(u)*eye(3) − u*u.T    # |u|² I − u⊗u
```

With u = p − q this is |p−q|² δ_ij − (p−q)_i (p−q)_j, which matches a_ij(p−q) exactly.
**Conclusion: the non-relativistic code already implements Maxwellian molecules.**

### 2. The Lemou initial condition is exactly representable in our n = 3 basis

For N = 3 and Villani's normalization conditions (∫f = 1, ∫fv = 0, ∫f|v|² = 3),
Lemou's initial datum is (Villani p. 6):

```
f₀(v) = M(v) (1 + |v|⁴/120 − |v|²/12 + 1/8)
```

where M(v) = e^{−|v|²/2} / (2π)^{3/2} is the standard Maxwellian.

This satisfies Villani's normalization conditions:
- ∫f₀ dv = 1  (the perturbation integrates to zero: 15/120 − 3/12 + 1/8 = 0)
- ∫f₀ v dv = 0  (radial symmetry)
- ∫f₀ |v|² dv = 3  (verified: 105/120 − 15/12 + 3/8 = 0 net perturbation to energy)

Our basis functions (for l = 0, m = 0) have the form:

```
ψ_{k,0,0}(v) = μ_{k,0} · L_k^{1/2}(r²) · Y₀⁰ · e^{−r²/2}
```

where r = |v|, μ_{k,0} = √(2 k! / Γ(k + 3/2)), Y₀⁰ = 1/√(4π).

Since f₀ = M(v) × (polynomial of degree 2 in r²), and the Laguerre polynomials
{L₀^{1/2}, L₁^{1/2}, L₂^{1/2}} span all polynomials of degree ≤ 2 in r², f₀ lives
**exactly** in the span of {ψ_{0,0,0}, ψ_{1,0,0}, ψ_{2,0,0}} — flat indices 0, 9, 18 with n = 3.

The Laguerre decomposition of the polynomial factor p(x) = 9/8 − x/12 + x²/120 (x = r²):

```
L₀^{1/2}(x) = 1
L₁^{1/2}(x) = 3/2 − x
L₂^{1/2}(x) = (x² − 5x + 15/4) / 2

p(x) = (33/32) L₀^{1/2} + (1/24) L₁^{1/2} + (1/60) L₂^{1/2}
```

The basis coefficients c_k are then:

```
c_k · μ_{k,0} · Y₀⁰ = [1/(2π)^{3/2}] · (Laguerre coeff of p)
```

These should be computed numerically (see plan below) to avoid rounding errors in the
γ-function expressions for μ_{k,0}.

### 3. The exact analytical solution

Villani's Fokker-Planck (for isotropic f satisfying the normalization conditions) is written
in his rescaled time t̃ = (N−1) t = 2t (N = 3). In that variable, the solution is:

```
h(t̃, v) = M(v) (1 + e^{−4t̃} (|v|⁴/120 − |v|²/12 + 1/8))
```

Our code's time t corresponds to Villani's original time (before the (N−1) rescaling), so
**in our code's time units the predicted decay is:**

```
h(t, v) = M(v) (1 + e^{−8t} (|v|⁴/120 − |v|²/12 + 1/8))
```

The non-equilibrium part decays at rate 8 in our time units. The equilibrium is M(v), i.e.
the coefficient vector converges to the pure k=0, l=0, m=0 component.

---

## Basis and Coefficient Conventions

This section documents the full chain from the physical distribution function f(v) down to
the numbers stored in the coefficient array, so the benchmark can be read independently of
the code.

### Reconstruction formula

The physical distribution function is expanded directly in the trial basis:

```
f(v) = Σ_{k,l,m} α_{k,l,m} · ψ_{k,l,m}(v)
```

where the **trial basis functions** are (consistent with the LaTeX write-up, eq. trial-basis):

```
ψ_{k,l,m}(v) = μ_{k,l} · r^l · e^{−r²/2} · L_k^{l+1/2}(r²) · Y_l^m(θ, φ)
```

with r = |v|, L_k^α the generalized Laguerre polynomial, Y_l^m a real spherical harmonic,
and μ_{k,l} = √(2 k! / Γ(k + l + 3/2)) a normalization constant. The Gaussian weight
e^{−r²/2} is part of the trial basis function.

**Code note:** In the implementation (`plot/lc.py`, `plot/test_func.py`), the Gaussian is
applied separately — `test_func` returns μ_{k,l} · L_k^{l+1/2}(r²) · Y_l^m (no Gaussian),
and `linear_comb` multiplies the whole sum by e^{−r²/2} afterward. This is a code
convenience; the two representations are equivalent.

### Flat index

For an n=3 basis (k = 0,1,2 and l = 0,…,n−1), the flat index of (k, l, m) is:

```
ind(k, l, m) = n² k + l² + (m + l)
```

For the isotropic modes (l = 0, m = 0): ind(0,0,0) = 0, ind(1,0,0) = 9, ind(2,0,0) = 18.

### Isotropic case (l = 0, m = 0)

For radially symmetric f (only l = 0, m = 0 modes active), r^l = 1 and Y_0^0 = 1/√(4π),
so the trial functions reduce to ψ_{k,0,0}(v) = μ_{k,0} · e^{−r²/2} · L_k^{1/2}(r²) · Y₀⁰,
and the reconstruction becomes:

```
f(v) = e^{−r²/2} · Y₀⁰ · Σ_k α_{k,0,0} · μ_{k,0} · L_k^{1/2}(r²)
```

(the Gaussian and Y₀⁰ factor out of the sum since they are independent of k).
Numerical values of μ_{k,0}:

| k | μ_{k,0} = √(2 k! / Γ(k + 3/2)) |
|---|----------------------------------|
| 0 | ≈ 1.5023  (= √(4/√π))           |
| 1 | ≈ 1.2266  (= √(8/(3√π)))         |
| 2 | ≈ 1.0970  (= √(16/(15√π)))       |

### Deriving the Lemou IC coefficients

Matching f₀(v) = M(v)(1 + p(r²)) to the expansion (with M(v) = e^{−r²/2}/(2π)^{3/2}):

```
[1/(2π)^{3/2}] · (1 + p(r²)) = [1/√(4π)] · Σ_k α_{k,0,0} · μ_{k,0} · L_k^{1/2}(r²)
```

Rearranging (the prefactor (2π)^{3/2}/√(4π) = π√2):

```
1 + p(r²) = π√2 · Σ_k α_{k,0,0} · μ_{k,0} · L_k^{1/2}(r²)
```

Using the Laguerre decomposition 1 + p(x) = (33/32)L₀ + (1/24)L₁ + (1/60)L₂:

```
α_{k,0,0} = LAG_k / (π√2 · μ_{k,0})
```

where LAG_0 = 33/32, LAG_1 = 1/24, LAG_2 = 1/60.

### Numerical values

| Flat index | (k,l,m) | LAG_k  | μ_{k,0} | α_{k,0,0}  |
|------------|---------|--------|---------|------------|
| 0          | (0,0,0) | 33/32  | 1.5023  | 0.154506   |
| 9          | (1,0,0) | 1/24   | 1.2266  | 0.007646   |
| 18         | (2,0,0) | 1/60   | 1.0970  | 0.003419   |

All other 24 coefficients (l > 0 or k > 2) are exactly zero by symmetry.
Reconstruction error against the exact f₀ on a dense radial grid: ≤ 10⁻¹⁶ (machine precision).

---

## Implementation Plan

### Step 1 — Compute the Lemou IC coefficient vector

Write a small script (e.g. `time_evol/lemou_ic.py` or a function in `time_ev.py`) that:

1. Evaluates μ_{k,0} for k = 0, 1, 2 and Y₀⁰ from `basis.py`.
2. Solves the linear system (or uses the analytic Laguerre coefficients above) to find
   c[0], c[9], c[18] such that Σ_k c_k ψ_{k,0,0}(v) = f₀(v).
3. **Verifies** the IC by reconstructing f₀ on a grid of (r, θ, φ) sample points and
   comparing to M(v)(1 + r⁴/120 − r²/12 + 1/8) — should agree to machine precision.
4. **Verifies** the normalization conditions (∫f = 1, ∫fv = 0, ∫f|v|² = 3) using the
   mass quadrature rule.

### Step 2 — Add `ic_mode = 'lemou'` to `time_ev.py`

Add a branch in the IC block:

```python
elif ic_mode == 'lemou':
    f[0]  = c0   # k=0, l=0, m=0
    f[9]  = c1   # k=1, l=0, m=0
    f[18] = c2   # k=2, l=0, m=0
```

where c0, c1, c2 are the values computed in Step 1. All other entries remain 0 (the IC is
exactly isotropic and in the span of n=3).

Choose time-step τ and NUM_ITERATIONS so that t_final = τ × N_iter covers several decay
times — since the decay rate is 8, a few units of t (e.g. t_final ~ 1–2) is sufficient.
Save coefficients at every step (or every few steps) for comparison.

### Step 3 — Run the time evolution

Use the existing non-relativistic dense operator
(`src/sparse_operators/nonrel_noncons_dense_n3_q9x7.pkl`, rel=False, cons=False,
sparse=False, n=3). This is the Maxwellian-molecules operator we verified above.

### Step 4 — Compare to the analytical solution

Write `plot/plot_lemou.py`:

1. **Coefficient trajectory:** For each saved time step, extract c[9](t) and c[18](t).
   Plot them on a log scale vs t. Expected: straight line with slope −8.

2. **Analytical overlay:** On the same plot, draw e^{−8t} × (initial amplitude of each
   mode). Should sit on top of the numerical curve.

3. **Radial profile comparison:** At several times t = 0, 0.1, 0.25, 0.5, reconstruct
   f(r) = Σ_k c_k(t) ψ_{k,0,0}(r) and overlay with h(t, r) = M(r)(1 + e^{−8t}(r⁴/120
   − r²/12 + 1/8)). Should agree to within quadrature accuracy.

4. **Conservation check:** Mass, momentum, and energy vs time should be flat to machine
   precision (reuse `time_evol/conservation_check.py`).

### Step 5 — Interpret and document

- If the numerical decay rate matches 8 to good accuracy, this validates the full pipeline
  (quadrature, mass matrix, bilinear form, time stepping) against an exact analytical
  benchmark for the first time.
- Note any discrepancy and attribute it (quadrature degree, time-step error, etc.).
- Record the result in the write-up repo.

---

## Results (session 2026-07-09)

All steps completed successfully. Summary of findings:

### Decay rate

| Mode | c_k(0) | c_k(t=1) | ratio | predicted e^{-8} | measured rate |
|------|--------|----------|-------|-----------------|---------------|
| c[9]  (k=1,l=0,m=0) | 7.6459e-03 | 2.5588e-06 | 3.347e-04 | 3.355e-04 | **8.0024** |
| c[18] (k=2,l=0,m=0) | 3.4193e-03 | 1.1443e-06 | 3.347e-04 | 3.355e-04 | **8.0024** |

- **Predicted decay rate: 8. Measured: 8.0024. Discrepancy: 0.03%** — attributable
  entirely to forward-Euler time-stepping error O(τ) = O(10⁻⁴). Not a physics error.
- Both modes decay at **identical** rates (ratio9/ratio18 = 1.000001), consistent with
  Villani's result that the full non-equilibrium perturbation is a single eigenmode.

### Isotropic confinement

All 24 coefficients outside {c[0], c[9], c[18]} remained at machine precision (≤ 10⁻²⁰)
throughout the entire evolution. The collision operator maps the l=0 subspace to itself
exactly — no spurious coupling to l > 0 modes.

### Radial profiles

The reconstructed f(r, t) and the analytical h(t, r) = M(r)(1 + e^{-8t}(r⁴/120 − r²/12 +
1/8)) are indistinguishable at plotting resolution (dpi=150) for all four snapshots
t = 0, 0.25, 0.50, ~1.0.

### Conclusion

**This is the first exact analytical benchmark of the full pipeline**, validating:
quadrature (9×7), mass matrix, bilinear collision-tensor assembly, time stepping, and
reconstruction — all against a known closed-form solution. The code passes.

Figures saved:
- `plot/figures/lemou_benchmark.png` — two-panel: log-scale decay + radial profiles
- `plot/figures/lemou_compare.png`  — side-by-side analytical vs numerical profiles
- `plot/figures/lemou_coeffs.png`   — coefficient evolution: c[0] flat, c[9]/c[18] → 0

---

## Notes / Caveats

- **Time-step error:** Forward Euler introduces O(τ) error per step. Use a small enough τ
  (e.g. τ = 1e-4 as currently set) to keep the Euler error well below the quadrature error.
- **Only valid for the non-relativistic non-conservative case.** The relativistic kernel
  adds the `−(z × u) ⊗ (z × u)` term (kern.py line 54), breaking the Maxwellian-molecule
  structure. The Villani analytical solution does NOT apply to the relativistic operator.
- **Exact representability:** The Lemou IC has zero projection onto any l > 0 or k > 2
  mode. With n = 3 the basis is complete for this IC. If n were increased, the extra
  coefficients should remain zero throughout the evolution (another implicit check).
