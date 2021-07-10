import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.special import binom
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gnk_model
import C0_calculation
import utils
import structure_utils
import data_utils

plt.style.use('seaborn-deep')
rcParams['savefig.dpi'] = 500
rcParams['lines.linewidth'] = 1.0
rcParams['axes.grid'] = True
rcParams['axes.spines.right'] = True
rcParams['axes.spines.top'] = True
rcParams['grid.color'] = 'gray'
rcParams['grid.alpha'] = 0.2
rcParams['axes.linewidth'] = 0.5
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'
rcParams['xtick.major.pad']='2'
rcParams['ytick.major.pad']='2'
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'
rcParams['xtick.major.size']='2'
rcParams['ytick.major.size']='2'
rcParams['xtick.major.width']='0.5'
rcParams['ytick.major.width']='0.5'

"""
This script produces the second row of Figure 4D, which displays the results
of various tests on the His3p(small) empirical fitness function.
"""

q = 2
L = 11

# get WH coefficients
beta = data_utils.calculate_his3p_wh_coefficients()

# Calculate fraction variance explained by empirical coefficients
beta_mag = beta**2
beta_mag /= np.sum(beta_mag)
bm_sorted = sorted(beta_mag, reverse=True)
bm_fv = np.zeros(len(bm_sorted))
for i in range(len(bm_sorted)):
    bm_fv[i] = np.sum(bm_sorted[:i])

# get contact map and calculate neighborhoods
bin_cm_sub = data_utils.get_his3p_binarized_contact_map()
V = structure_utils.contact_map_to_neighborhoods(bin_cm_sub)

# calculate the coefficient variances corresponding to neighborhoods
gnk_beta_var_ = gnk_model.calc_beta_var(L, q, V)
gnk_beta_var = gnk_beta_var_/np.sum(gnk_beta_var_) # normalize beta
gnk_sparsity = np.count_nonzero(gnk_beta_var)
percent_var = 100*bm_fv[gnk_sparsity]
num_samples = int(np.ceil(gnk_sparsity*C0_calculation.C0_VAL*np.log10(q**L)))
print("Sparsity of His3p(small) Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover GNK: %s" % num_samples)
print("Percent variance explained by largest %i empirical coefficients: %.3f" % (gnk_sparsity, percent_var))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained(gnk_beta_var_, samples=1000, up_to=76)

# Load LASSO results
results_dict = np.load("../results/his3p_small_lasso_results.npy", allow_pickle=True).item()
ns = results_dict['n']
pearson = results_dict['pearson']
mean_r = np.mean(pearson**2, axis=1)
std_r = np.std(pearson**2, axis=1)
idx = np.where(ns==num_samples)[0][0]
r2_val = mean_r[idx]
print("LASSO R^2 at GNK predicted number of samples: %.3f" % r2_val)

# Load example results
example_results = np.load("../results/his3p_small_lasso_example.npy")
y = example_results[0]
pred = example_results[1]
r2_test = pearsonr(pred, y)[0]**2
m, b = np.polyfit(y ,pred, 1)
minmin = np.min([np.min(pred), np.min(y)])
maxmax = np.max([np.max(pred), np.max(y)])
