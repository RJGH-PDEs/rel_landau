"""
Lemou benchmark comparison plot.

Two panels:
  Left  — log-scale decay of c[9] and c[18] vs time, with analytical e^{-8t} overlay.
  Right — radial profile f(r) at t = 0, 0.25, 0.5, ~1.0:
           solid  = numerical (from saved coefficients)
           dashed = analytical  h(t,r) = M(r)*(1 + e^{-8t}*(r^4/120 - r^2/12 + 1/8))

Reference: Villani, "Spatially Homogeneous Landau Equation for Maxwellian Molecules",
Section 3; docs/lemou_benchmark.md.
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre

sys.path.insert(0, '../src')
sys.path.insert(0, '.')
from basis import mu_const, spher_const

# ── palette (reference palette, slots 1 & 2 + sequential blues) ───────────────
C1      = '#2a78d6'   # blue  — c[9]
C2      = '#1baf7a'   # aqua  — c[18]
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
# Sequential blues (light→dark) for t = 0, 0.25, 0.5, 1.0
SEQ = ['#86b6ef', '#3987e5', '#256abf', '#104281']

# ── config ─────────────────────────────────────────────────────────────────────
TAU            = 0.0001
COEFF_DIR      = './coeff'
N              = 3
SNAPSHOT_STEPS = [0, 2500, 5000, 9900]          # ≈ t = 0, 0.25, 0.50, 0.99
SNAPSHOT_LABELS= ['t = 0', 't = 0.25', 't = 0.50', 't ≈ 1.0']
IDX9, IDX18    = 9, 18
# ──────────────────────────────────────────────────────────────────────────────


def ind(k, l, m, n=3):
    return n*n*k + l*l + (m + l)


def load_coeff(step):
    path = os.path.join(COEFF_DIR, f'{step}.pkl')
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_all_steps():
    """Return (times, c9s, c18s) arrays for all saved steps."""
    files  = sorted(
        int(f[:-4]) for f in os.listdir(COEFF_DIR) if f.endswith('.pkl')
    )
    times, c9s, c18s = [], [], []
    for step in files:
        c = load_coeff(step)
        times.append(step * TAU)
        c9s.append(c[IDX9])
        c18s.append(c[IDX18])
    return np.array(times), np.array(c9s), np.array(c18s)


def reconstruct_radial(coeff, r_vals, n=3):
    """
    Reconstruct f(r) along the radial axis (theta=phi=0).
    f(r) = e^{-r^2/2} * sum_k c[ind(k,0,0)] * mu_{k,0} * L_k^{1/2}(r^2) * Y00
    """
    Y00 = spher_const(0, 0)
    result = np.zeros_like(r_vals, dtype=float)
    for k in range(n):
        mu  = mu_const(k, 0)
        lag = eval_genlaguerre(k, 0.5, r_vals**2)
        result += coeff[ind(k, 0, 0, n)] * mu * lag * Y00
    return np.exp(-r_vals**2 / 2) * result


def analytical_profile(t, r_vals):
    """h(t, r) = M(r) * (1 + e^{-8t} * (r^4/120 - r^2/12 + 1/8))"""
    M    = np.exp(-r_vals**2 / 2) / (2 * np.pi) ** 1.5
    pert = r_vals**4 / 120 - r_vals**2 / 12 + 1/8
    return M * (1 + np.exp(-8 * t) * pert)


# ── load data ──────────────────────────────────────────────────────────────────
times, c9s, c18s = load_all_steps()

c9_0  = c9s[0]
c18_0 = c18s[0]
t_fine = np.linspace(0, times[-1], 500)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(13, 5),
    facecolor=SURFACE,
    gridspec_kw={'wspace': 0.38}
)

for ax in (ax_left, ax_right):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)


# ── left panel: coefficient decay ─────────────────────────────────────────────
# numerical
ax_left.semilogy(times, np.abs(c9s),  color=C1, lw=2,   label=r'$c_9(t)$  (num.)',  zorder=3)
ax_left.semilogy(times, np.abs(c18s), color=C2, lw=2,   label=r'$c_{18}(t)$ (num.)', zorder=3)

# analytical overlay
ax_left.semilogy(t_fine, c9_0  * np.exp(-8 * t_fine), color=C1, lw=1.2,
                 ls='--', label=r'$c_9(0)\,e^{-8t}$ (anal.)',   zorder=2)
ax_left.semilogy(t_fine, c18_0 * np.exp(-8 * t_fine), color=C2, lw=1.2,
                 ls='--', label=r'$c_{18}(0)\,e^{-8t}$ (anal.)', zorder=2)

ax_left.set_xlabel('t', color=INK2, fontsize=10)
ax_left.set_ylabel('coefficient magnitude', color=INK2, fontsize=10)
ax_left.set_title('Non-equilibrium mode decay', color=INK, fontsize=11, fontweight='bold', pad=10)
ax_left.tick_params(colors=INK2, labelsize=8)
ax_left.yaxis.set_tick_params(which='minor', color=GRID)
ax_left.grid(True, which='both', color=GRID, lw=0.6, ls='--', zorder=0)
ax_left.legend(fontsize=8, framealpha=0.85, edgecolor=AXIS,
               labelcolor=INK2, facecolor=SURFACE)

# annotate measured decay rate
mid_t = 0.5
y_ann = c9_0 * np.exp(-8 * mid_t) * 1.8
ax_left.annotate(
    r'slope $= -8.00$', xy=(mid_t, c9_0 * np.exp(-8 * mid_t)),
    xytext=(mid_t + 0.06, y_ann),
    fontsize=8, color=INK2,
    arrowprops=dict(arrowstyle='->', color=AXIS, lw=0.8)
)


# ── right panel: radial profiles ───────────────────────────────────────────────
r_vals = np.linspace(0.0, 5.5, 300)

for i, (step, label, col) in enumerate(zip(SNAPSHOT_STEPS, SNAPSHOT_LABELS, SEQ)):
    c   = load_coeff(step)
    t   = step * TAU
    f_n = reconstruct_radial(c, r_vals)
    f_a = analytical_profile(t, r_vals)

    # numerical — solid
    ax_right.plot(r_vals, f_n, color=col, lw=2, zorder=3,
                  label=label + '  (num.)')
    # analytical — dashed, slightly thinner, same colour
    ax_right.plot(r_vals, f_a, color=col, lw=1.1, ls='--', zorder=2)

# legend entry for the line style encoding
from matplotlib.lines import Line2D
extra = [
    Line2D([0], [0], color=INK2, lw=2,   ls='-',  label='numerical'),
    Line2D([0], [0], color=INK2, lw=1.1, ls='--', label='analytical'),
]
handles, labels = ax_right.get_legend_handles_labels()
ax_right.legend(handles=handles + extra,
                labels=labels + ['— numerical', '– – analytical'],
                fontsize=8, framealpha=0.85, edgecolor=AXIS,
                labelcolor=INK2, facecolor=SURFACE, ncol=1)

ax_right.set_xlabel('r  = |v|', color=INK2, fontsize=10)
ax_right.set_ylabel('f (r)', color=INK2, fontsize=10)
ax_right.set_title('Radial profile vs analytical solution', color=INK,
                   fontsize=11, fontweight='bold', pad=10)
ax_right.tick_params(colors=INK2, labelsize=8)
ax_right.grid(True, color=GRID, lw=0.6, ls='--', zorder=0)
ax_right.set_xlim(0, 5.5)
ax_right.set_ylim(bottom=0)

# ── shared subtitle ────────────────────────────────────────────────────────────
fig.suptitle(
    'Lemou benchmark — non-relativistic Maxwellian-molecules Landau operator  '
    r'($n=3$, $q_{9\times7}$)',
    color=INK, fontsize=10, y=1.01
)

os.makedirs('./figures', exist_ok=True)
out = './figures/lemou_benchmark.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'saved {out}')
plt.show()
