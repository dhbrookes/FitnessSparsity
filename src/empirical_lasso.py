import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from collections import Counter
from tqdm import tqdm

import utils
import gnk_model

"""
Code for running LASSO experiments on empirical fitness functions, the results of
which are shown in Figure 4D.
"""


def run_lasso_experiment(X, y, alpha, ns, savefile, save_example_at=None, 
                         example_savefile=None, replicates=50):
    """
    Runs the LASSO algorithm for a given data set (X, y), regularization parameter (alpha),
    and ns, a list of the number of data points to subsample from the data. At each value in
    ns, data are randomly subsampled and and the LASSO algorithm is run on the subsampled 
    data. This process is repeated a number of times given by 'replicates'. For each 
    replicate at each value in ns, the mse and pearson correlation between the predicted
    and true y values are saved in a dictionary, which is also returned.
    """
    ns = np.array(np.sort(list(ns))).astype(int)
    mse = np.zeros((len(ns), replicates))
    pearson = np.zeros((len(ns), replicates))
    print("Running LASSO tests...")
    for i, n in enumerate(tqdm(ns)):
        for j in tqdm(range(replicates)):
            model = Lasso(alpha=alpha)
            X_train, _, y_train, _ = train_test_split(X, y, train_size=n)
            model.fit(X_train, y_train)
            pred = model.predict(X)
            pearson[i, j] = pearsonr(y, pred)[0]
            mse[i, j] = np.mean((pred-y)**2)
            if save_example_at is not None and n == save_example_at and j==0:
                np.save(example_savefile, np.array([y, pred]))
    results_dict = {'n': ns, 'pearson': pearson, 'mse': mse, 'alpha': alpha}
    np.save(savefile, results_dict)
    return results_dict


def determine_alpha(X, y, n, replicates=10):
    """
    Determines the optimal regularization parameter for n data points randomly subsampled from
    a given data set (X, y).
    """
    alphas = [5e-8, 1e-8, 5e-7, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
    opt_vals = np.zeros(replicates)
    for j in range(replicates):
        model = LassoCV(alphas=alphas, n_jobs=10)
        X_train, _, y_train, _ = train_test_split(X, y, train_size=n)
        model.fit(X_train, y_train)
        opt_vals[j] = model.alpha_
    cts = Counter(opt_vals)
    opt_alpha = cts.most_common(1)[0][0]
    return opt_alpha