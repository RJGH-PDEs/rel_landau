"""
Side-by-side comparison: analytical vs numerical Lemou solution.

Left  — analytical:  h(t,r) = M(r)*(1 + e^{-8t}*(r^4/120 - r^2/12 + 1/8))
Right — numerical:   f(r,t) reconstructed from saved coefficient snapshots

Both panels use the same axes, colors, and time snapshots, so any discrepancy
between them is immediately visible.
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre

sys.path.insert(0, '../src')
from basis import mu_const, spher_const

# ── palette ────────────────────────────────────────────────────────────────────
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
# sequential blues light → dark for t = 0 → 1
SEQ = ['#86b6ef', '#3987e5', '#256abf', '#104281']

# ── config ─────────────────────────────────────────────────────────────────────
TAU            = 0.0001
COEFF_DIR      = './coeff'
N              = 3
SNAPSHOT_STEPS = [0, 2500, 5000, 9900]
SNAPSHOT_TIMES = [s * TAU for s in SNAPSHOT_STEPS]
SNAPSHOT_LABELS = ['t = 0', 't = 0.25', 't = 0.50', 't ≈ 1.0']
# ──────────────────────────────────────────────────────────────────────────────


def ind(k, l, m, n=3):
    return n*n*k + l*l + (m + l)


def load_coeff(step):
    with open(os.path.join(COEFF_DIR, f'{step}.pkl'), 'rb') as fh:
        return pickle.load(fh)


def reconstruct_radial(coeff, r_vals, n=3):
    """f(r) = e^{-r²/2} * Σ_k c_k * μ_{k,0} * L_k^{1/2}(r²) * Y00"""
    Y00    = spher_const(0, 0)
    result = np.zeros_like(r_vals, dtype=float)
    for k in range(n):
        result += coeff[ind(k, 0, 0, n)] * mu_const(k, 0) * eval_genlaguerre(k, 0.5, r_vals**2) * Y00
    return np.exp(-r_vals**2 / 2) * result


def analytical_profile(t, r_vals):
    """h(t, r) = M(r) * (1 + e^{-8t} * (r^4/120 - r^2/12 + 1/8))"""
    M    = np.exp(-r_vals**2 / 2) / (2 * np.pi) ** 1.5
    pert = r_vals**4 / 120 - r_vals**2 / 12 + 1/8
    return M * (1 + np.exp(-8 * t) * pert)


# ── build profiles ─────────────────────────────────────────────────────────────
r_vals = np.linspace(0.0, 5.5, 400)

analytical = [analytical_profile(t, r_vals) for t in SNAPSHOT_TIMES]
numerical  = [reconstruct_radial(load_coeff(s), r_vals) for s in SNAPSHOT_STEPS]

# shared y limits
y_max = max(f.max() for f in analytical) * 1.06
y_min = -0.002

# ── figure ─────────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(12, 4.8),
    facecolor=SURFACE,
    sharey=True,
    gridspec_kw={'wspace': 0.10}
)

for ax, profiles, title in [
    (ax_l, analytical, 'Analytical'),
    (ax_r, numerical,  'Numerical'),
]:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)

    for f, label, col in zip(profiles, SNAPSHOT_LABELS, SEQ):
        ax.plot(r_vals, f, color=col, lw=2, label=label, zorder=3)

    ax.set_xlim(0, 5.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$r_p = |p|$', color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.6, ls='--', zorder=0)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=12, fontweight='bold', pad=10)

ax_l.set_ylabel(r'$f(r_p)$', color=INK2, fontsize=10)
ax_r.tick_params(labelleft=False)

# single shared legend below both panels
handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=4, fontsize=9,
           framealpha=0.9, edgecolor=AXIS, labelcolor=INK2, facecolor=SURFACE,
           bbox_to_anchor=(0.5, -0.08))

# max pointwise error across all snapshots
max_err = max(
    np.max(np.abs(a - n_)) for a, n_ in zip(analytical, numerical)
)
fig.suptitle(
    'Lemou benchmark — analytical vs numerical  '
    f'(max pointwise error {max_err:.2e})',
    color=INK, fontsize=10, y=1.02
)

os.makedirs('./figures', exist_ok=True)
out = './figures/lemou_compare.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'saved {out}')
plt.show()
