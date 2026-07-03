import sys
sys.path.insert(0, '../src')

from test_func import test
from sparse import ind

def linear_comb(coefficients, r, t, p, n=3):
    result = 0

    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):
                result = result + coefficients[ind(k, l, m, n)]*test(k, l, m, r, t, p)

    return result

