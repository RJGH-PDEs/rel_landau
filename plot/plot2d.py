import sys
sys.path.insert(0, '../src')

import matplotlib.pyplot as plt
import numpy as np
import os
from lc import linear_comb
from naming import load_with_meta

# ── flags ─────────────────────────────────────────────────────────────────────
N          = 3
tau        = 0.0001
steps      = [0, 100, 300, 1000, 'equil']
equil_step = 1500
ic_label   = 'coeff[0]=1,  coeff[9]=-0.5,  coeff[2]=0.1,  coeff[11]=0.0632'
save       = True
show       = False

N_PTS = 80      # grid resolution per axis (80×80 = 6 400 evaluations)
x_max = 4.0     # plot extent in both x and z
# ── end flags ─────────────────────────────────────────────────────────────────

xv = np.linspace(-x_max, x_max, N_PTS)
zv = np.linspace(-x_max, x_max, N_PTS)
X, Z = np.meshgrid(xv, zv)

# Spherical coordinates on the x-z plane (y=0):
#   theta in [0,pi] measured from z-axis  →  arctan2(|x|, z) works for all quadrants
#   phi = 0 (x>=0) or pi (x<0); m=0 modes are phi-independent so the plane is symmetric in x
R     = np.sqrt(X**2 + Z**2)
THETA = np.arctan2(np.abs(X), Z)
PHI   = np.where(X >= 0, 0.0, np.pi)

os.makedirs('./figures', exist_ok=True)


def load_step(step):
    fname = f'coeff/{equil_step}.pkl' if step == 'equil' else f'coeff/{step}.pkl'
    coeff, meta = load_with_meta(fname)
    return coeff, (meta['n'] if meta is not None else N)


def step_label(step):
    if step == 'equil':
        return f'equilibrium  (t={equil_step * tau:.4f})'
    return f't = {step * tau:.4f}'


def eval_f_2d(coeff, n_basis):
    """Evaluate f (including Gaussian weight) on the x-z plane grid."""
    R_flat = R.ravel()
    T_flat = THETA.ravel()
    P_flat = PHI.ravel()
    F_flat = np.array([
        np.exp(-R_flat[i]**2 / 2) * linear_comb(coeff, R_flat[i], T_flat[i], P_flat[i], n=n_basis)
        for i in range(len(R_flat))
    ])
    return F_flat.reshape(R.shape)


# ── evaluate all panels ───────────────────────────────────────────────────────
panels = []
labels = []
for step in steps:
    print(f'evaluating step {step} ...', flush=True)
    coeff, n_basis = load_step(step)
    panels.append(eval_f_2d(coeff, n_basis))
    labels.append(step_label(step))

# choose colormap: diverging if f goes negative, sequential otherwise
vmin_all = min(F.min() for F in panels)
vmax_all = max(F.max() for F in panels)
if vmin_all < -0.01 * vmax_all:
    cmap = 'RdBu_r'
    v = max(abs(vmin_all), vmax_all)
    vmin, vmax = -v, v
else:
    cmap = 'plasma'
    vmin, vmax = 0.0, vmax_all

# ── plot ──────────────────────────────────────────────────────────────────────
ncols = min(3, len(steps))
nrows = (len(steps) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                         sharex=True, sharey=True)
axes_flat = np.array(axes).flatten()

for ax, F, label in zip(axes_flat, panels, labels):
    im = ax.pcolormesh(xv, zv, F, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axvline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)

for ax in axes_flat[len(steps):]:
    ax.set_visible(False)

fig.colorbar(im, ax=axes_flat[len(steps) - 1], shrink=0.8, label='f(x, 0, z)')
title = f'Non-relativistic relaxation — x-z plane  ($\\Delta t = {tau}$)'
if ic_label:
    title += f'\nIC:  {ic_label}'
fig.suptitle(title, fontsize=11)
plt.tight_layout()

fig_name = './figures/relaxation_2d_xz.png'
if show:
    plt.show()
if save:
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f'saved {fig_name}')
