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
        t       = select[0]
        f1      = select[1]
        f2      = select[2]

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
    return 19
    
# now just one list with simplified indeces
def simple_index(nz, L):
    sim_ind = []

    # iterate over the non-zeros
    for data in nz:
        # extract indices
        te  = data[0] # test 
        tr  = data[1] # trial
        val = data[2]

        # compute simple indices
        t = ind(te[0],te[1], te[2], L)
        p = ind(tr[0],tr[1], tr[2], L)
        u = ind(tr[3],tr[4], tr[5], L)

        # append 
        result = [t, p, u, val]
        sim_ind.append([data, result]) 
    
    return sim_ind

# produces the dense operator as a list of matrices
def dense_op(si, n):
    # initialize 
    size = n**3
    shape = (size, size)
    dense = []
    for i in range(0, n**3):
        dense.append(np.zeros(shape))

    # fill
    for element in si:
        # extract values 
        t   = element[1][0]
        p   = element[1][1]
        u   = element[1][2]
        val = element[1][3] 
        
        # insert
        dense[t][p][u] = val
        '''
        NOTE: multiply by -1 to
        correct mistake from before
        dense[t][p][u] = (-1)*val
        '''
    return dense


# produces the sparse operator, as a list of sparse matrices
def sparse_op(do):
    sparse = []
    
    for dense_matrix in do:
        sparse.append(csr_matrix(dense_matrix))

    return sparse

# main funtion
def main():
    n = 2
    L = n - 1 # max value l can take?
    tol = 0.03 # tolerance for the nonzeros

    file_name = 'e=analytical.pkl'
    # file_name = 'analytic_energy.pkl'
    
    print(file_name)
    op = load_operator(file_name)   # operator pkl
    nz = non_zeros(op, tol)         # non zeros
    analyse(nz)
    # finish here
    return 0

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
    sparse_name = "sparse_operator.pkl" 
    save_sparse_op(sparse_name, so)

    return 0
    
# execute main funtion
if __name__ == "__main__":
    main()
