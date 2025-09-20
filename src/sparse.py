import pickle 
import numpy as np
from scipy.sparse import csr_matrix

# l and m map
def lm_index(ll, m): 
    return ll*ll + (m + ll)

# k, l, m map
def ind(k, ll, m, L):
    '''
    here, we use the convenction that 
    l is between 0 and L
    '''
    return (L+1)*(L+1)*k + lm_index(ll, m)

def test_indices():
    k = 0
    l = 0
    m = 0
    L = 2
    print(ind(k, l, m, L))

# loads and returns data
def load_operator(name):
    # load
    with open(name, 'rb') as file:
        data = pickle.load(file)

    # return
    return data

# save operator 
def save_sparse_op(name, operator):
    with open(name, 'wb') as file:
        pickle.dump(operator, file)

# extracts non-zero entries
def non_zeros(operator, tol):
    '''
    At this point, operator is a list (1-dim) containing
    objects of the form [select, value], we extract the 
    non-zeros based on a tolerance.
    '''
    # will store the non-zeros
    nz = []
    
    # extract non zeros
    for data in operator:

        # append the non-zero
        if np.abs(data[1]) > tol:
            nz.append(data)    

    # print the number of non-zeros
    print('Number of non-zeros: ', len(nz))
    return nz

# checks sparsity patterns
def analyse(nz):
    # counts how many times the rule is broken
    counter = 0

    # iterate over all non-zeros
    for e in nz:
        # extract coefficients
        select  = e[0]
        t       = select[0] # test 
        f1      = select[1] # function 1, f(p)
        f2      = select[2] # function 2, dg(q)

        ######## directional (CAI)
        m_test  = t[2] 
        m_1     = f1[2]
        m_2     = f2[2]

        test    = np.abs(m_test)
        sum     = np.abs(m_1 + m_2)
        diff    = np.abs(m_1 - m_2)
        
        Caiflag = (test - sum == 0) or (test - diff == 0)

        if not Caiflag:
            counter = counter + 1

        ####### anisotropic (Andrea)
        ltest  = t[1]
        l1     = f1[1]
        l2     = f2[1]
        rule = l1 + l2 - ltest
        m = min(l1, l2)

        Andrea_flag = (rule <= 2*m) and (0 <= rule) and (rule % 2 == 0)
        print(e, "Cai ",  Caiflag, "Andrea: " , Andrea_flag)

        if not Andrea_flag:
            counter = counter + 1

    print('number of times the sparsity rule failed: ', counter)
    
# now just one list with simplified indeces
def simple_index(nz, L):
    sim_ind = []

    # iterate over the non-zeros
    for data in nz:
        # extract indices
        select  = data[0]
        t       = select[0] # test 
        f       = select[1] # function 1, f(p)
        g       = select[2] # function 2, dg(q)
        # extract value
        val     = data[1]

        # compute simple indices
        t_ind = ind(t[0], t[1], t[2], L)
        f_ind = ind(f[0], f[1], f[2], L)
        g_ind = ind(g[0], g[1], g[2], L)

        # append 
        result = [t_ind, f_ind, g_ind, val]

        sim_ind.append(result) 
    
    print("finished computing the list with a simple index.")
    return sim_ind

# produces the dense operator as a list of matrices
def dense_op(si, n):
    '''
    dense is a list of matrices, the convention is
    that every matrix will be associated with a 
    specific test function, then, the fist coordinate
    associated with f(p), and the second one to dg(q) 

    '''
    # initialize 
    size = n**3
    shape = (size, size)
    dense = []
    for i in range(0, n**3):
        dense.append(np.zeros(shape))

    # fill
    for element in si:
        # extract values, should be 
        # in accordance with convention in 
        # simple index
        t   = element[0]
        f   = element[1]
        g   = element[2]
        val = element[3] 
        
        # insert
        dense[t][f][g] = val # notice the index convention
    
    print("finished computing the dense tensor (list of dense matrices).")
    return dense


# produces the sparse operator, as a list of sparse matrices
def sparse_op(do):
    sparse = []
    
    for dense_matrix in do:
        print("check for symmetry: ", np.max(np.abs(dense_matrix - dense_matrix.T)))
        sparse.append(csr_matrix(dense_matrix))
    
    print("finished computing the sparse tensor (list of sparse matrices).")
    return sparse

# main funtion
def main():
    n = 3
    L = n - 1       # max value l can take?
    tol = 0.0001    # tolerance for the nonzeros

    file_name = 'results/test.pkl'
    
    print("analyzing for file with name: ", file_name)
    op = load_operator(file_name)   # load operator pkl
    nz = non_zeros(op, tol)         # extract non zeros
    analyse(nz)                     # analyse sparsity 
    
    si = simple_index(nz, L)        # with simple index
    do = dense_op(si, n)            # dense operator
    so = sparse_op(do)              # sparse operator
    
   
    # compute the size of the sparse operator
    print('sparse operator length: ', len(so))

    # show the number of non-zeros
    print('number of non-zeros per matrix: ')
    for slice in so:
        print(slice.nnz)
 
    # save it 
    sparse_name = "sparse_operators/test.pkl" 
    save_sparse_op(sparse_name, so)

    return 0
    
# execute main funtion
if __name__ == "__main__":
    main()
