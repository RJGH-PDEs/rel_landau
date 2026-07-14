import multiprocessing
import time
import pickle
import sympy as sp

# import pieces from operator
from kern import kernel
from landau import operator_parallel, load_quad
from sparse_rules import andrea, cai
from naming import operator_tag, save_with_meta
from quadrature import load_quad_order

# create iterable
def create_param_iterable(n, rel, sparse=True):
    param = []

    # iterate for all test functions
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):

                # iterate over all f(p)
                for k1 in range(0, n):
                    for l1 in range(0, n):
                        for m1 in range(-l1, l1+1):
               
                            # iterate over all dg(q)    
                            for k2 in range(0, n):
                                for l2 in range(0, n):
                                    for m2 in range(-l2, l2+1): 

                                        # create the select
                                        select = [[k,l,m], [k1,l1,m1], [k2,l2,m2]]

                                        # skip conservation laws
                                        # mass & momentum vanish for ANY kernel (write-up Prop. 1,
                                        # eq. 455): grad(phi_p)-grad(phi_q)=0 for constant/linear phi.
                                        mass = [k,l,m] == [0,0,0]
                                        px   = [k,l,m] == [0,1,-1]
                                        py   = [k,l,m] == [0,1,0]
                                        pz   = [k,l,m] == [0,1,1]
                                        # energy (1,0,0) ~ r^2 is conserved ONLY non-relativistically
                                        # (then u = grad(E_p)-grad(E_q) proportional to p-q, killed by
                                        # S). Relativistically u = p/E_p - q/E_q, so (1,0,0) is NOT
                                        # conserved by Phi_simple and must be computed. See TODO Part 7.
                                        e    = ([k,l,m] == [1,0,0]) and (not rel)

                                        flag = mass or px or py or pz or e

                                        # sparse=True: prune provably-zero entries with the
                                        # directional (cai) and anisotropic (andrea) selection
                                        # rules. sparse=False: compute the full (dense) tensor,
                                        # e.g. to validate the rules via sparse.py's analyse().
                                        if sparse:
                                            keep = (not flag) and andrea(select) and cai(select)
                                        else:
                                            keep = not flag

                                        if keep:
                                            param.append(select)
    print("number of coefficients to compute: ", len(param), "(sparse)" if sparse else "(dense)")
    return param

# parallel iterator
def parallel(sd, n, rel, sparse=True):
    '''
    Arguments:
        - sd: the shared data. [quadrature, kernel]
        - n:  corresponds to max k, l
        - rel: relativistic flag (controls the energy conservation-law skip)
        - sparse: if True, prune provably-zero entries with the cai/andrea rules;
                  if False, compute the full dense tensor.
    '''
    # obtain the list of parameters
    params = create_param_iterable(n, rel, sparse)

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_parallel, [(select, sd) for select in params])

    # r.append(results) 
    return results

# produce collision matrix
def parallel_setup(n, energy, rel, sparse=True):
    # load quadrature
    quad = load_quad()
    print("quadrature size: ", len(quad))

    # produce the symbolic kernel, dependent on the energy
    verbose = False
    kern    = kernel(energy, verbose, rel)

    # define the shared data
    sd = [quad, kern] # this has to be in accordance to the unpacking at "operator_parallel"
                
    # compute it and time it
    start = time.time()
    r = parallel(sd, n, rel, sparse)
    end = time.time()

    # Calculate elapsed time
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.6f} seconds") 
    print()
    return r

def compute_col_tensor():
    # relativistic flag
    rel     = True
    # conservative flag
    cons    = False
    # sparse flag: True -> prune zeros with the cai/andrea rules; False -> full dense tensor
    sparse  = True

    # select the degrees of freedom
    n       = 3

    # where the result will be saved. The tag is built from the run config by
    # naming.operator_tag so the producer and sparse.py/time_ev.py stay in sync.
    # The quadrature order is read from the actual quadrature file, not the
    # constants, so the name can't misreport the order that was used.
    n_lag, n_leb = load_quad_order()
    meta      = {'rel': rel, 'cons': cons, 'sparse': sparse,
                 'n': n, 'n_lag': n_lag, 'n_leb': n_leb}
    tag       = operator_tag(rel, cons, sparse, n, n_lag, n_leb)
    file_name = f'results/{tag}.pkl'
    
    # load the energy
    if rel:
        if cons: 
            # open the numerical energy
            with open('../cheby/eh.pkl', 'rb') as f:
                energy = pickle.load(f)
        else: 
            # radial symbol
            r = sp.symbols('r')
            energy = sp.sqrt(1+r**2)  # relativistic

    else:
        # radial symbol
        r = sp.symbols('r')
        energy = (1/2)*r**2

    # compute the tensor, results will contain it
    result = parallel_setup(n, energy, rel, sparse)

    # print the result
    print(result)

    # save the result (self-describing: {'meta', 'data'})
    save_with_meta(file_name, result, meta)
    print("the result has been saved at ", file_name)

# main function
if __name__ == "__main__":
    compute_col_tensor()
    # n = 3
    # create_param_iterable(n)
