"""
Coefficient evolution for the Lemou benchmark.
c[0]  (k=0, l=0, m=0) — Maxwellian mode, converges to equilibrium
c[9]  (k=1, l=0, m=0) — non-equilibrium, decays as e^{-8t}
c[18] (k=2, l=0, m=0) — non-equilibrium, decays as e^{-8t}
"""

import os, pickle
import numpy as np
import matplotlib.pyplot as plt

# ── palette ────────────────────────────────────────────────────────────────────
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
C0_COL  = '#2a78d6'   # blue  — c[0]
C9_COL  = '#1baf7a'   # aqua  — c[9]
C18_COL = '#eda100'   # yellow — c[18]

TAU      = 0.0001
COEFF_DIR = './coeff'

# ── load ───────────────────────────────────────────────────────────────────────
steps = sorted(int(f[:-4]) for f in os.listdir(COEFF_DIR) if f.endswith('.pkl'))
times, c0s, c9s, c18s = [], [], [], []
for s in steps:
    with open(f'{COEFF_DIR}/{s}.pkl', 'rb') as fh:
        c = pickle.load(fh)
    times.append(s * TAU)
    c0s.append(c[0])
    c9s.append(c[9])
    c18s.append(c[18])

times = np.array(times)
c0s   = np.array(c0s)
c9s   = np.array(c9s)
c18s  = np.array(c18s)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for spine in ax.spines.values():
    spine.set_color(AXIS)
    spine.set_linewidth(0.8)

ax.semilogy(times, c0s,  color=C0_COL,  lw=2, label=r'$c_0$  (k=0, equilibrium)')
ax.semilogy(times, c9s,  color=C9_COL,  lw=2, label=r'$c_9$  (k=1, decays $e^{-8t}$)')
ax.semilogy(times, c18s, color=C18_COL, lw=2, label=r'$c_{18}$ (k=2, decays $e^{-8t}$)')

ax.set_xlabel('t', color=INK2, fontsize=11)
ax.set_ylabel('coefficient magnitude', color=INK2, fontsize=11)
ax.set_title('Lemou benchmark — coefficient evolution', color=INK,
             fontsize=12, fontweight='bold', pad=10)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(True, which='both', color=GRID, lw=0.6, ls='--', zorder=0)
ax.legend(fontsize=9, framealpha=0.9, edgecolor=AXIS,
          labelcolor=INK2, facecolor=SURFACE)

os.makedirs('./figures', exist_ok=True)
out = './figures/lemou_coeffs.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'saved {out}')
plt.show()
