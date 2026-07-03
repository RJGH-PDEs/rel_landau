import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
import os

# ── flags ─────────────────────────────────────────────────────────────────────
# experiment: 'symmetric' | 'asymmetric' | 'zero_momentum'
# Must match the CSV produced by time_evol/conservation_check.py.
experiment = 'zero_momentum'

_IC_LABELS = {
    'symmetric':     'coeff[0]=1,  coeff[9]=-0.5',
    'asymmetric':    'coeff[0]=1,  coeff[9]=-0.5,  coeff[2]=0.1',
    'zero_momentum': 'coeff[0]=1,  coeff[9]=-0.5,  coeff[2]=0.1,  coeff[11]=0.0632',
}

save = True
show = False
# ── end flags ─────────────────────────────────────────────────────────────────

csv_path = f'../time_evol/conservation/{experiment}.csv'
data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
# columns: step, t, mass, px, py, pz, energy
t      = data[:, 1]
mass   = data[:, 2]
px     = data[:, 3]
py     = data[:, 4]
pz     = data[:, 5]
energy = data[:, 6]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

# Mass and energy have large absolute values but tiny variation, so plot
# deviation from the initial value to keep the y-axis readable.
# Momentum is already naturally small and is shown in absolute terms.
dmass   = mass   - mass[0]
denergy = energy - energy[0]

# ── mass drift ────────────────────────────────────────────────────────────────
axes[0].plot(t, dmass, color='steelblue', linewidth=1.5)
axes[0].axhline(0, color='gray', linewidth=0.7, linestyle='--')
axes[0].set_ylabel('$\\Delta$ mass  $(M\\mathbf{f})_0 - (M\\mathbf{f})_0^{t=0}$')
axes[0].grid(True, alpha=0.4)
axes[0].text(0.01, 0.97, f'initial value: {mass[0]:.6g}',
             transform=axes[0].transAxes, fontsize=8, va='top', color='gray')

# ── momentum (absolute) ───────────────────────────────────────────────────────
axes[1].plot(t, px, color='tomato',    linewidth=1.5, label='$P_x$')
axes[1].plot(t, py, color='goldenrod', linewidth=1.5, label='$P_y$')
axes[1].plot(t, pz, color='seagreen',  linewidth=1.5, label='$P_z$')
axes[1].axhline(0, color='gray', linewidth=0.7, linestyle='--')
axes[1].set_ylabel('momentum  $(M\\mathbf{f})_{1,2,3}$')
axes[1].legend(fontsize=9, loc='right')
axes[1].grid(True, alpha=0.4)

# ── energy drift ──────────────────────────────────────────────────────────────
axes[2].plot(t, denergy, color='mediumpurple', linewidth=1.5)
axes[2].axhline(0, color='gray', linewidth=0.7, linestyle='--')
axes[2].set_ylabel('$\\Delta$ energy proxy  $(M\\mathbf{f})_9 - (M\\mathbf{f})_9^{t=0}$')
axes[2].set_xlabel('physical time  $t$')
axes[2].grid(True, alpha=0.4)
axes[2].text(0.01, 0.97, f'initial value: {energy[0]:.6g}',
             transform=axes[2].transAxes, fontsize=8, va='top', color='gray')

ic = _IC_LABELS.get(experiment, experiment)
fig.suptitle(
    f'Conserved quantities — {experiment.replace("_", " ")} IC\nIC:  {ic}',
    fontsize=11
)
plt.tight_layout()

os.makedirs('./figures', exist_ok=True)
fig_name = f'./figures/conservation_{experiment}.png'
if show:
    plt.show()
if save:
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f'saved {fig_name}')
