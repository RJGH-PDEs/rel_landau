"""
Same Lemou coefficient data plotted against Villani's rescaled time t̃ = 2t.

In t̃ = (N-1)t = 2t (N=3), the predicted decay rate is e^{-4t̃} (Villani
Section 3), recovering his factor of 4. Our code works in the original
unrescaled time t, so the rate there is e^{-8t} = e^{-4t̃}. This plot
shows the two are the same result with different clock conventions.
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
C9_COL  = '#1baf7a'   # aqua
C18_COL = '#eda100'   # yellow

TAU       = 0.0001
COEFF_DIR = './coeff'

# ── load ───────────────────────────────────────────────────────────────────────
steps = sorted(int(f[:-4]) for f in os.listdir(COEFF_DIR) if f.endswith('.pkl'))
times, c9s, c18s = [], [], []
for s in steps:
    with open(f'{COEFF_DIR}/{s}.pkl', 'rb') as fh:
        c = pickle.load(fh)
    times.append(s * TAU)
    c9s.append(c[9])
    c18s.append(c[18])

times   = np.array(times)
c9s     = np.array(c9s)
c18s    = np.array(c18s)
t_tilde = 2 * times          # Villani's rescaled time t̃ = (N-1)t = 2t

t_fine  = np.linspace(0, t_tilde[-1], 500)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for sp in ax.spines.values():
    sp.set_color(AXIS)
    sp.set_linewidth(0.8)

# numerical (plotted against t̃)
ax.semilogy(t_tilde, c9s,  color=C9_COL,  lw=2, label=r'$c_9$  (numerical)')
ax.semilogy(t_tilde, c18s, color=C18_COL, lw=2, label=r'$c_{18}$ (numerical)')

# analytical overlay: e^{-4 t̃}  (Villani's formula)
ax.semilogy(t_fine, c9s[0]  * np.exp(-4 * t_fine), color=C9_COL,
            lw=1.2, ls='--', label=r'$c_9(0)\,e^{-4\tilde{t}}$ (Villani)')
ax.semilogy(t_fine, c18s[0] * np.exp(-4 * t_fine), color=C18_COL,
            lw=1.2, ls='--', label=r'$c_{18}(0)\,e^{-4\tilde{t}}$ (Villani)')

# measured rate in t̃
rate = -np.log(c9s[-1] / c9s[0]) / t_tilde[-1]
ax.annotate(f'slope = $-${rate:.2f}', xy=(0.98, 0.55), xycoords='axes fraction',
            ha='right', fontsize=10, color=INK2)

ax.set_xlabel(r'$\tilde{t} = 2t$  (Villani rescaled time)', color=INK2, fontsize=11)
ax.set_ylabel('coefficient magnitude', color=INK2, fontsize=11)
ax.set_title(r'Same data, Villani time $\tilde{t}=2t$ — slope $= -4$',
             color=INK, fontsize=12, fontweight='bold', pad=10)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(True, which='both', color=GRID, lw=0.6, ls='--', zorder=0)
ax.legend(fontsize=9, framealpha=0.9, edgecolor=AXIS,
          labelcolor=INK2, facecolor=SURFACE)

os.makedirs('./figures', exist_ok=True)
out = './figures/lemou_villani_time.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'saved {out}  (measured rate in t̃: {rate:.4f}, Villani predicts 4)')
plt.show()
