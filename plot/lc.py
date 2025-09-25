# import numpy as np
from test_func import test
import pickle

def linear_comb(coefficients, r, t, p):
    # max k, l, m
    n = 3

    # partial result
    result = 0

    # counter 
    i = 0

    for k in range(0,n):
        for l in range(0, n):
            for m in range(-l, l+1):
                # print([k, l, m] , i)
                result = result + coefficients[i]*test(k, l, m, r, t, p)
                i = i + 1

    return result

'''
# open the result
with open('result.pkl', 'rb') as file:
    data = pickle.load(file)

i = 0
for s in data:
    print(i, data[i])
    i = i + 1

# test it
coeff = np.zeros(27)
coeff[0] = 1

linear_comb(coeff, 1, 0, 0)
'''