import sys
sys.path.insert(0, '../src')

import csv
import glob
import os
import numpy as np
from pathlib import Path
from naming import load_with_meta, mass_tag
from quadrature import load_quad_order

# ── flags ─────────────────────────────────────────────────────────────────────
# experiment: 'symmetric' | 'asymmetric' | 'zero_momentum'
# Must match the ic_mode used when the coeff files in plot/coeff/ were generated.
experiment = 'zero_momentum'

tau   = 0.0001
n     = 3
# run config — used only to locate the mass-matrix file
rel   = False
cons  = False
# ── end flags ─────────────────────────────────────────────────────────────────

# conserved-quantity flat indices (n=3, ind(k,l,m) = 9k + l^2 + (m+l))
#   0 = ind(0,0,0)  — mass
#   1 = ind(0,1,-1) — P_y  (Y_{1,-1} ∝ sinθ sinφ)
#   2 = ind(0,1, 0) — P_z  (Y_{1, 0} ∝ cosθ)
#   3 = ind(0,1, 1) — P_x  (Y_{1, 1} ∝ sinθ cosφ)
#   9 = ind(1,0, 0) — energy proxy  (L_1^{1/2}(r²) = 3/2 − r²)
CONSERVED = {'mass': 0, 'px': 3, 'py': 1, 'pz': 2, 'energy': 9}

# ── load mass matrix and recover M ────────────────────────────────────────────
m_lag, m_leb = load_quad_order('../src/quadrature/mass_quadrature.pkl')
mi, _ = load_with_meta(f'../src/mass/{mass_tag(n, m_lag, m_leb)}.pkl')
M = np.linalg.inv(mi)   # 27×27 — cheap

# ── collect and sort coeff files ──────────────────────────────────────────────
coeff_files = sorted(
    glob.glob('../plot/coeff/*.pkl'),
    key=lambda p: int(Path(p).stem)
)
if not coeff_files:
    raise FileNotFoundError('No coeff pkl files found in ../plot/coeff/')

print(f'Found {len(coeff_files)} coeff files  (steps '
      f'{int(Path(coeff_files[0]).stem)} – {int(Path(coeff_files[-1]).stem)})')

# ── compute conserved quantities at each step ─────────────────────────────────
rows = []
for fpath in coeff_files:
    step = int(Path(fpath).stem)
    f, _ = load_with_meta(fpath)   # bare pickle → meta is None, that is fine
    q = M @ f
    rows.append({
        'step':   step,
        't':      step * tau,
        'mass':   q[CONSERVED['mass']],
        'px':     q[CONSERVED['px']],
        'py':     q[CONSERVED['py']],
        'pz':     q[CONSERVED['pz']],
        'energy': q[CONSERVED['energy']],
    })

# ── save CSV ──────────────────────────────────────────────────────────────────
os.makedirs('./conservation', exist_ok=True)
csv_path = f'./conservation/{experiment}.csv'
fieldnames = ['step', 't', 'mass', 'px', 'py', 'pz', 'energy']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'Saved {csv_path}')
