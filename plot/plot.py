import sys
sys.path.insert(0, '../src')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from lc import linear_comb
from naming import load_with_meta

# ── flags ─────────────────────────────────────────────────────────────────────
# experiment: 'symmetric' | 'asymmetric' | 'zero_momentum'
# Set this to match ic_mode in time_evol/time_ev.py before running.
experiment = 'symmetric'

_IC_LABELS = {
    'symmetric':     r'$\alpha_{0,0,0}=1$, $\alpha_{1,0,0}=-1/2$',
    'asymmetric':    r'$\alpha_{0,0,0}=1$, $\alpha_{1,0,0}=-1/2$, $\alpha_{0,1,0}=0.1$',
    'zero_momentum': r'$\alpha_{0,0,0}=1$, $\alpha_{1,0,0}=-1/2$, $\alpha_{0,1,0}=0.1$, $\alpha_{1,1,0}\approx 0.0632$',
}
_FIG_NAMES_OVERLAY = {
    'symmetric':     './figures/relaxation_symmetric_overlay.png',
    'asymmetric':    './figures/relaxation_asymmetric_overlay.png',
    'zero_momentum': './figures/relaxation_zero_momentum_overlay.png',
}

# basis truncation (flat dim = N**3)
N = 3
# time step size (for physical-time labels in grid/overlay modes)
tau = 0.0001
# plotting mode: 'single' | 'grid' | 'overlay'
mode = 'overlay'
save = True
show = False

# single mode: one snapshot or hard-coded coefficients
time             = 1000
hard_coded_coeff = False

# grid / overlay mode: list of iteration indices (int) or 'equil'
steps      = [0, 100, 300, 1000, 'equil']
equil_step = 1500    # which saved pkl is treated as the equilibrium snapshot

# overlay mode: IC annotation and figure name derived from experiment
ic_label   = _IC_LABELS[experiment]
# ── end flags ─────────────────────────────────────────────────────────────────

# x-axis evaluation grid (along the x-axis: y=z=0)
N_PTS = 200
x    = np.linspace(-5, 5, N_PTS)
r    = np.abs(x)
t_sp = np.where(x >= 0, 0.0, np.pi)   # theta: 0 for x>0, pi for x<0
p_sp = np.zeros(N_PTS)                 # phi = 0

os.makedirs('./figures', exist_ok=True)


def eval_f(coeff, n_basis):
    """Reconstruct f(x) = e^{-r^2/2} * sum_j coeff_j psi_j along the x-axis."""
    return np.array([
        np.exp(-r[i]**2 / 2) * linear_comb(coeff, r[i], t_sp[i], p_sp[i], n=n_basis)
        for i in range(N_PTS)
    ])


def load_step(step):
    """Load coefficient vector for a given iteration index or 'equil'."""
    fname = f'coeff/{equil_step}.pkl' if step == 'equil' else f'coeff/{step}.pkl'
    coeff, meta = load_with_meta(fname)
    return coeff, (meta['n'] if meta is not None else N)


def step_label(step):
    """Human-readable label showing physical time."""
    if step == 'equil':
        return f'equilibrium  (t = {equil_step * tau:.4f})'
    return f't = {step * tau:.4f}'


# ── single mode ───────────────────────────────────────────────────────────────
if mode == 'single':
    if hard_coded_coeff:
        coeff    = np.zeros(N**3)
        coeff[0] = 1
        coeff[9] = -0.5
        n_basis  = N
        plt_name = 'hard-coded coefficients'
        fig_name = './figures/hardcoded.png'
    else:
        coeff, n_basis = load_step(time)
        plt_name = step_label(time)
        fig_name = f'./figures/{time}.png'

    f = eval_f(coeff, n_basis)
    plt.figure(figsize=(7, 5))
    plt.plot(x, f, marker='o', markersize=3)
    plt.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    plt.title(plt_name)
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'$f(p)$')
    plt.grid(True, alpha=0.4)
    if show: plt.show()
    if save:
        plt.savefig(fig_name, dpi=150)
        print('saved', fig_name)

# ── grid mode ─────────────────────────────────────────────────────────────────
elif mode == 'grid':
    ncols = 3
    nrows = (len(steps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()

    for ax, step in zip(axes_flat, steps):
        coeff, n_basis = load_step(step)
        f = eval_f(coeff, n_basis)
        ax.plot(x, f)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_title(step_label(step))
        ax.grid(True, alpha=0.4)

    for ax in axes_flat[len(steps):]:
        ax.set_visible(False)

    fig.suptitle(f'Non-relativistic relaxation  ($\\Delta t = {tau}$)', fontsize=13)
    plt.tight_layout()
    fig_name = './figures/relaxation_grid.png'
    if show: plt.show()
    if save:
        plt.savefig(fig_name, dpi=150)
        print('saved', fig_name)

# ── overlay mode ──────────────────────────────────────────────────────────────
elif mode == 'overlay':
    colors = cm.plasma(np.linspace(0, 0.9, len(steps)))
    fig, ax = plt.subplots(figsize=(9, 6))

    for color, step in zip(colors, steps):
        coeff, n_basis = load_step(step)
        f = eval_f(coeff, n_basis)
        ax.plot(x, f, color=color, label=step_label(step))

    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_xlabel(r'$p_x$')
    ax.set_ylabel(r'$f(p)$')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.4)
    if ic_label:
        ax.text(0.02, 0.97, f'IC:  {ic_label}', transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig_name = _FIG_NAMES_OVERLAY[experiment]
    if show: plt.show()
    if save:
        plt.savefig(fig_name, dpi=150)
        print('saved', fig_name)
