import sys
sys.path.append("../src")
import numpy as np

import C_calculation

"""
This script processes the results of the calculations to
determine a suitable value of C and saves the processed
results for later use. Run as:

$ python process_C_results.py

"""


def determine_N_for_recovery(Ns, fv_beta, frac_var_tol=0.01):
    """
    Determine minimum N for which majority and all recovery occurs for a specific beta. 
    Majority and all refers to whether recovery occurfs or a majority of training 
    set replicates, or for all replicates. Parameter frac_var_tol sets the
    maximum error tolerance allowed to deem the coefficients exactly recovered.
    """
    N_conditions = np.zeros((len(Ns), 4), dtype=bool)
    for i, N in enumerate(Ns):
        fv_i = fv_beta[i]
        fv_less_than_tol = (fv_i <= frac_var_tol).flatten()
        fv_total_ltt = np.count_nonzero(fv_less_than_tol)
        fv_maj_ltt = (fv_total_ltt / len(fv_less_than_tol)) > 0.5
        fv_all_ltt = np.all(fv_less_than_tol)
        N_conditions[i] = [fv_maj_ltt, fv_all_ltt]
    
    maj_sat = np.argwhere(np.all(N_conditions[:, :2], axis=1))
    min_maj_sat = np.min(maj_sat)
    N_maj = Ns[min_maj_sat]

    all_sat = np.argwhere(np.all(N_conditions[:, 2:], axis=1))
    min_all_sat = np.min(all_sat)
    N_all = Ns[min_all_sat]
    return N_maj, N_all, N_conditions


if __name__ == "__main__":
    frac_var_tol = 0.0001
    
    # determine C required for recovery in each tested case
    C_alls = np.zeros((len(tested), 25))  # 5 beta samples x 5 training set replicates = 25
    lqk = []
    k = 0
    for L, q, K in tested:
        print(L, q, K)
        lqk.append([L, q, K])
        M = q**L
        f = np.load("../results/%i_%i_%i_rn.npy" %(L, q, K), allow_pickle=True).item()
        Ns = f['N']
        betas = f['beta']
        S_beta = f['S_beta']
        fv = f['fv']

        for i, beta in enumerate(betas):
            S = S_beta[i]
            fv_beta = fv[:, i, :]
            fz_beta = fz[:, i, :]
            try:
                N_maj, N_all, N_conditions = determine_N_for_recovery(Ns, fv_beta , fz_beta, 
                                                                      frac_zero_tol=frac_zero_tol, 
                                                                      frac_var_tol=frac_var_tol)
            except ValueError:
                continue
            C0_maj = N_maj / (S*np.log10(M))
            C0_all = N_all / (S*np.log10(M))
            print(("L: %i, q: %i, K: %i, C0_maj, C_all: %.2f, %.2f" % (L, q, K, C_maj, C_all)))
            C_alls[k, i] = C0_all
            if i >= C_alls.shape[1]-1:
                break       
        k += 1
        
    # Determine number of cases recovered at a range of settings C between 0 and 3
    Cs = np.linspace(0, 3, 1000)
    frac_recovered = np.zeros((C_alls.shape[0], len(Cs)))
    nall = 0
    for j in range(C_alls.shape[0]):
        call = C_alls[j]
        call = call[np.nonzero(call)]
        nall += call.shape[0]
        for i, c in enumerate(Cs):
            lt = call < c
            fr = np.sum(lt) / lt.shape[0]
            frac_recovered[j, i] = fr
            
    recovery_dict = {'Cs': Cs, 'frac_recovered': frac_recovered}
    np.save('../results/C_recovery.npy', recovery_dict)