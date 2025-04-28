import multiprocessing
import time
import pickle
import sympy as sp

# import pieces from operator
from kern import kernel
from operator_1 import operator_parallel, load_quad

# create iterable
def create_param_iterable(n):
    param = []

    # iterate for all test functions
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):
                # iterate over all p
                for k1 in range(0, n):
                    for l1 in range(0, n):
                        for m1 in range(-l1, l1+1):
                            # iterate over all q    
                            for k2 in range(0, n):
                                for l2 in range(0, n):
                                    for m2 in range(-l2, l2+1): 
                                        # create the select
                                        select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]

                                        # skip conservation laws
                                        mass = [k,l,m] == [0,0,0]
                                        px   = [k,l,m] == [0,1,-1]
                                        py   = [k,l,m] == [0,1,0]
                                        pz   = [k,l,m] == [0,1,1]
                                        e    = [k,l,m] == [1,0,0] 
                                        flag = mass or px or py or pz or e

                                        if not flag:
                                            param.append(select)

    return param

# given a weight iterates over different trial functions
def trial_iterator(data, n, r):
    # Create a manager for shared data
    manager = multiprocessing.Manager()
    shared_data = manager.list(data)

    # Define the list of parameters
    params = create_param_iterable(n)

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_parallel, [(select, shared_data) for select in params])

    r.append(results)

# produce collision matrix
def weight_iteration(n, r, fr):
    # Load data once 
    quad = load_quad()
    print("quadrature size: ", len(quad))

    # produce the kernel once
    kern = kernel(energy)

    # defined the shared data
    sd = [quad, kern]
                
    # compute it and time it
    start = time.time()
    trial_iterator(sd, n, r)
    end = time.time()

    # save the partial result
    with open(fr, 'wb') as file:
        pickle.dump(r, file)

    # Calculate elapsed time
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.6f} seconds") 
    print()

# main function
if __name__ == "__main__":
    # select the degrees of freedom
    n = 2

    # where to save result:
    file_result = 'results.pkl'
    
    # results to be stored here
    results = []

    '''
    Choose the energy
    '''
    # radial symbol
    r = sp.symbols('r')
    # energy = (1/2)*r**2       # non-relativistic
    energy = sp.sqrt(1+r**2)  # relativistic
    # energy   = r**3             # polynomial
    # energy = 1 + 0.43991322*r**2 - 0.0338162*r**4

    # compute the tensor
    weight_iteration(n, results, file_result)

    # print the result
    print(results)

    # save the result
    with open(file_result, 'wb') as file:
        pickle.dump(results, file)
