import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from scipy.special import binom
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gnk_model
import C_calculation
import utils
import structure_utils
import data_utils
import plot_utils

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['seaborn-deep', 'plots/paper.mplstyle'])

"""
This script produces the second row of Figure 4D, which displays the results
of various tests on the His3p(small) empirical fitness function, and prints
quantities related to this analysis. Run as:

$ python figure4_bottom.py

"""

q = 2
L = 11

# get WH coefficients
emp_beta = data_utils.calculate_his3p_small_wh_coefficients()

# Calculate fraction variance explained by empirical coefficients
beta_mag_sq = emp_beta**2 / np.sum(emp_beta**2)  # normalize sum of squares to one
bm_fv = utils.calc_frac_var_explained(emp_beta)

# get gnk_coefficients
gnk_beta_var_, V = data_utils.calculate_his3p_small_gnk_wh_coefficient_vars(return_neighborhoods=True)
gnk_beta_var = gnk_beta_var_/np.sum(gnk_beta_var_) # normalize sum of variances to one
gnk_sparsity = np.count_nonzero(gnk_beta_var)
pv_at_gnk = 100*bm_fv[gnk_sparsity]
pred_num_samples = int(np.ceil(gnk_sparsity*C_calculation.C_VAL*np.log10(q**L)))
print("Sparsity of His3p(small) Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover His3p(small) GNK: %s" % pred_num_samples)
print("Percent variance explained by largest %i His3p(small) empirical coefficients: %.3f" % (gnk_sparsity, pv_at_gnk))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained_from_beta_var(gnk_beta_var_, samples=1000, up_to=100)

# Load LASSO results
results_dict = np.load("../results/his3p_small_lasso_results.npy", allow_pickle=True).item()
example_results = np.load("../results/his3p_small_lasso_example.npy")

########################
### Make large panel ###
########################

fig = plt.figure(figsize=(15, 3))
gs = fig.add_gridspec(1,5)

# plot neighborhoods
ax = fig.add_subplot(gs[0, 0])
plot_utils.plot_neighborhoods(ax, V, L, 
                              data_utils.HIS3P_POSITIONS, 
                              label_rotation=60, s=120)

# plot beta comparison
ax = fig.add_subplot(gs[0, 1:3])
num_coeffs = int(np.sum([binom(L, i) for i in range(7)])) 

his3p_kwargs = {
    'max_order': 6,
    'order_lbl_offset': 20,
    'order_lbl_height': 0.35,
    'arrow1_xy': (4, 0.39),
    'arrow1_text_xy': (38, 0.48),
    'arrow2_xy': (32, 0.34),
    'arrow2_text_xy': (54, 0.42),
    'yticks': (-0.4, -0.2, 0, 0.2, 0.4),
    'yticklabels': ('0.40', '0.20', '0', '0.20', '0.40')
}

max_order = 6
num_coeffs = int(np.sum([binom(L, i) for i in range(max_order+1)]))
plot_utils.plot_beta_comparison(ax, L, num_coeffs, 
                                beta_mag_sq, 
                                gnk_beta_var, 
                                use_order_labels=True, 
                                **his3p_kwargs)



# plot percent variance explained
ax = fig.add_subplot(gs[0, 3])
plot_utils.plot_percent_variance_explained(ax, 100*bm_fv, 
                                           100*gnk_fv_mean, 
                                           100*gnk_fv_std,
                                           gnk_sparsity,
                                           xlim=100,
                                           xticks=(0, 25, 50, 75, 100)
                                          )

# plot lasso results
ax = fig.add_subplot(gs[0, 4])
plot_utils.plot_lasso_results(ax, results_dict, pred_num_samples)

axins = inset_axes(ax, width="75%", height="95%",
                   bbox_to_anchor=(0.5, 0.25, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)

his3p_inset_kwargs = {
    'xlim': (-0.1, 1.3),
    'ylim': (-0.5, 1.5),
    'xticks': (-0.5, 0, 0.5, 1, 1.5),
    'yticks': (0, 0.5, 1),
    'text_xy': (0.6, -0.25)
}

plot_utils.plot_lasso_example_inset(axins, example_results, **his3p_inset_kwargs)


plt.tight_layout()
plt.savefig("plots/figure4_his3p.png", dpi=500, bbox_inches='tight', transparent=True)