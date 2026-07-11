import sys
import os

sys.path.insert(0, '../../src')
from sparse_rules import andrea, cai
from naming import operator_tag, load_with_meta

# ── config ────────────────────────────────────────────────────────────────────
N_VALUES   = [2, 3, 4, 5]
# for actual nonzero comparison (only n=3 has been computed locally)
REL, CONS  = False, False
N_LAG, N_LEB = 9, 7
# ─────────────────────────────────────────────────────────────────────────────

def basis_triples(n):
    triples = []
    for k in range(n):
        for l in range(n):
            for m in range(-l, l + 1):
                triples.append((k, l, m))
    return triples

def count_zeros(n):
    basis = basis_triples(n)
    N     = len(basis)   # should equal n^3

    andrea_zeros = 0
    cai_zeros    = 0
    both_zeros   = 0
    rule_zeros   = 0

    for i in basis:
        for s in basis:
            for r in basis:
                select = [list(i), list(s), list(r)]
                a_pass = andrea(select)
                c_pass = cai(select)
                if not a_pass:
                    andrea_zeros += 1
                if not c_pass:
                    cai_zeros += 1
                if (not a_pass) and (not c_pass):
                    both_zeros += 1
                if (not a_pass) or (not c_pass):
                    rule_zeros += 1

    return N, N**3, andrea_zeros, cai_zeros, both_zeros, rule_zeros

def load_actual_nnz(n):
    tag  = operator_tag(REL, CONS, False, n, N_LAG, N_LEB)
    path = f'../../src/sparse_operators/{tag}.pkl'
    if not os.path.exists(path):
        return None
    tensor, _ = load_with_meta(path)
    return sum(mat.nnz for mat in tensor)

# ── header ────────────────────────────────────────────────────────────────────
print(f"{'n':>2}  {'N':>4}  {'N^3':>8}  "
      f"{'andrea_zeros':>14}  {'cai_zeros':>12}  "
      f"{'both_zeros':>12}  {'rule_zeros':>12}  "
      f"{'rule_%':>8}  {'actual_nnz':>12}")
print("-" * 110)

for n in N_VALUES:
    N, N3, az, cz, bz, rz = count_zeros(n)
    actual = load_actual_nnz(n)
    actual_str = str(actual) if actual is not None else "n/a"
    print(f"{n:>2}  {N:>4}  {N3:>8}  "
          f"{az:>8} ({100*az/N3:5.1f}%)  "
          f"{cz:>7} ({100*cz/N3:5.1f}%)  "
          f"{bz:>7} ({100*bz/N3:5.1f}%)  "
          f"{rz:>7} ({100*rz/N3:5.1f}%)  "
          f"{actual_str:>12}")

print()
print("rule_zeros = entries guaranteed zero by the selection rules (andrea OR cai fails)")
print("Inclusion-exclusion check: andrea_zeros + cai_zeros - both_zeros == rule_zeros")
