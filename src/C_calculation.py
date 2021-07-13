import numpy as np
import utils
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split

import gnk_model


"""
Code for determining a suitable value of the C constant.
"""

C_VAL = 2.62  # value used in future calculations, saved here for convenience


def sample_rn_beta(L, q, K):
    """
    Returns the Fourier coefficients of a sample GNK fitness function
    with Random neighborhoods.
    """
    Vrn = gnk_model.sample_random_neighborhoods(L, K)
    lam = gnk_model.calc_beta_var_single_q(L, q, Vrn)
    sample = np.random.randn(q**L) * np.sqrt(lam).reshape(1, q**L)
    return sample


def single_lasso_run(X_train, y_train, alpha, beta):
    """
    Runs the LASSO algorithm for a given dataset (X_train, y_train), and 
    regularization parameter (alpha). Calculates the error of the resulting
    Fourier coefficients compared to the true coefficients (beta), 
    and returns the fraction variace explained by the estimated coefficients
    (calculated as the L2 error of estimated coefficiets, divided by the
    L2 norm of the true coefficients.)
    """
    M = X_train.shape[1]
    lasso = Lasso(alpha=alpha, fit_intercept=True)
    lasso.fit(X_train, y_train)
    beta_est = lasso.coef_
    beta_est[0] = lasso.intercept_ * np.sqrt(M)
    diff = np.abs(beta - beta_est)**2
    l2_err = np.sum(diff)
    frac_variance = l2_err / np.sum(beta**2)
    return frac_variance
    

def lasso_test_single_N(X, y, beta, N, training_replicates=5):
    """
    Runs the LASSO test for replicates of training sets of with N samples, given
    the complete fitness data (X, y) and Fourier coefficients (beta). The
    number of replicates is given by training_replicates. For each replicate,
    the optimal regularization parameter is determined, which is returned
    along with the optimal error of each replicate.
    """
    M = X.shape[1]
    frac_variances = np.zeros(training_replicates)
    frac_zeros = np.zeros(training_replicates)
    alphas = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    alpha_bests = np.zeros(training_replicates)
    for j in range(training_replicates):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N)
        fv_best = 1
        alpha_best = None
        for alpha in alphas:
            frac_variance = single_lasso_run(X_train, y_train, alpha, beta)
            if frac_variance < fv_best:
                fv_best = frac_variance
                alpha_best = alpha
        frac_variances[j] = fv_best
        alpha_bests[j] = alpha_best
    return frac_variances, alpha_bests


def lasso_test(L, q, betas, Ns,
               training_replicates=5,  
               verbose=False, noise_var=0):
    """
    Runs the LASSO test for given sequence length (L), alphabet size (q),
    list of sample Fourier coefficients (betas), and list of training set 
    sizes (Ns). Returns an array of errors and optimal regularization 
    parameters for each training set size, Fourier coefficient sample, 
    and training set replicate.
    """
    M = q**L
    X = utils.fourier_basis_recursive(L, q)
    num_betas = len(betas)
    frac_variances = np.zeros((len(Ns), num_betas, training_replicates))
    frac_zeros = np.zeros((len(Ns), num_betas, training_replicates))
    alphas = np.zeros((len(Ns), num_betas, training_replicates))
    ys = np.array([np.dot(X, beta) + np.sqrt(noise_var) * np.random.randn(M) for beta in betas])
    
    for i, N in enumerate(Ns):
        for j, beta in enumerate(betas):
            y = ys[j]
            fv_N, alphas_N = lasso_test_single_N(X, y, beta, N, 
                                                 training_replicates=training_replicates
                                                )
            frac_variances[i, j] = fv_N
            alphas[i, j] = alphas_N
            if verbose:
                print("%i / %i Ns and %i / %i beta replicates" %(i+1, len(Ns), j+1, num_betas))
    return frac_variances, alphas


def run_LK(L, q, K, verbose=False, num_Ns=50, 
           min_factor=1, max_factor=5,
           num_betas=5):
    """
    Runs the LASSO test for a given sequence length (L), alphabet size (q), and
    setting of the K parameter. This method first calculates the expected
    and maximum sparsity of GNK models with Random neighborhoods for L, q, and K. 
    It then determines which training set sizes to use for the tests, which 
    are evenly distributed between min_factor*mean_sparsity*log10(q^L) and
    max_factor*max_sparsity. This range usually includes the value
    required for exact recovery of GNK fitness functions when min_factor
    and max_factor are set to their default values of 1 and 5, respectively,
    though sometimes these parameters must be adjusted to ensure recovery. This
    method then samples Fourier coefficients of GNK fitness functions, runs the 
    tests, and saves the results to the 'results' folder.
    """
    M = q**L
    s_bound = gnk_model.calc_max_rn_sparsity(L, q, K)
    s_mean = gnk_model.calc_mean_rn_sparsity(L, q, K)
    min_n = min_factor*s_mean*np.log10(M)
    Ns = np.linspace(min_n, np.min([max_factor*s_bound, M-0.01*M]), num_Ns).astype(int)
    Ns = np.unique(Ns)
    S_beta = np.zeros(num_betas)
    betas = np.zeros((num_betas, M))
    for i in range(num_betas):
        betas[i] = sample_rn_beta(L, q, K)
        S_beta[i] = np.count_nonzero(betas[i])
    

    fv, alphas = lasso_test(L, q, betas, Ns,
                                training_replicates=5, 
                                verbose=verbose)
    results = {'N': Ns,
               'beta': betas,
               'S_beta': S_beta,
               'fv': fv,
               'alpha': alphas
              }
    np.save('../results/%i_%i_%i_rn.npy' %(L, q, K), results)
