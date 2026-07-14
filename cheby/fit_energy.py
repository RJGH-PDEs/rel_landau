"""
Unweighted least-squares fit of sqrt(1+r^2) onto span{1, r^2, r^4}
at the GL quadrature nodes. Spreads the error evenly across all nodes
rather than concentrating it near r=0 (as the Gaussian-weighted L^2 does).

Output: eh.pkl — SymPy expression  a0 + a2*r**2 + a4*r**4
"""

import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os

# ── load GL radial nodes ───────────────────────────────────────────────────
with open('../src/quadrature/quadrature.pkl', 'rb') as f:
    q = pickle.load(f)
r_nodes = np.unique([p[0][0] for p in q['points']])

E_exact = lambda r: np.sqrt(1 + r**2)

# ── GL-weighted least squares: min sum_i w_i*(a0 + a2*r_i^2 + a4*r_i^4 - E(r_i))^2
# extract GL radial weights (one per unique node)
w_nodes = np.array([next(p[0][1] for p in q['points'] if abs(p[0][0] - r) < 1e-10)
                    for r in r_nodes])

# weighted least squares: scale rows by sqrt(w_i)
sw = np.sqrt(w_nodes)
A  = np.column_stack([np.ones_like(r_nodes), r_nodes**2, r_nodes**4])
Aw = A * sw[:, None]
bw = E_exact(r_nodes) * sw
coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
a0, a2, a4 = coeffs

print('GL-weighted least-squares fit coefficients (even powers only):')
print(f'  a0 = {a0:.10f}')
print(f'  a2 = {a2:.10f}')
print(f'  a4 = {a4:.10f}')
print()

E_fit   = a0 + a2*r_nodes**2 + a4*r_nodes**4
rel_err = np.abs(E_fit - E_exact(r_nodes)) / E_exact(r_nodes)

print(f'{"r":>8}  {"exact":>12}  {"fit":>12}  {"rel err":>10}')
for r, ex, fi, re in zip(r_nodes, E_exact(r_nodes), E_fit, rel_err):
    print(f'{r:8.4f}  {ex:12.8f}  {fi:12.8f}  {re:10.2e}')
print()
print(f'Max relative error at GL nodes: {rel_err.max():.2e}')

# ── save SymPy expression ──────────────────────────────────────────────────
r = sp.symbols('r')
expr = a0 + a2*r**2 + a4*r**4

with open('./eh.pkl', 'wb') as f:
    pickle.dump(expr, f)
print('\nsaved eh.pkl')

# ── plot ───────────────────────────────────────────────────────────────────
r_plot = np.linspace(0, r_nodes.max() * 1.05, 400)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(r_plot, E_exact(r_plot), 'k-', lw=2, label=r'$\sqrt{1+r^2}$')
ax.plot(r_plot, a0 + a2*r_plot**2 + a4*r_plot**4, 'b--', lw=2,
        label=r'$a_0 + a_2 r^2 + a_4 r^4$')
ax.scatter(r_nodes, E_exact(r_nodes), color='red', zorder=5, s=40, label='GL nodes')
ax.set_xlabel('r'); ax.set_ylabel('E(r)')
ax.legend(); ax.set_title('Energy')

ax2 = axes[1]
ax2.semilogy(r_nodes, rel_err, 'o-', color='crimson')
ax2.set_xlabel('r'); ax2.set_ylabel('relative error')
ax2.set_title('Relative error at GL nodes')
ax2.grid(True, which='both', ls='--', lw=0.5)

plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/energy_fit.png', dpi=150, bbox_inches='tight')
print('saved ./figures/energy_fit.png')
plt.show()
