import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
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
This script produces Figure S1, which displays the results
of various tests on the His3p(big) empirical fitness function, and prints
quantities related to this analysis. Run as:

$ python figure_s1.py

"""

qs = data_utils.HIS3P_BIG_QS
L = 11

# get WH coefficients
emp_beta = data_utils.calculate_his3p_big_fourier_coefficients()

# Calculate fraction variance explained by empirical coefficients
beta_mag_sq = emp_beta**2 / np.sum(emp_beta**2)  # normalize sum of squares to one
bm_fv = utils.calc_frac_var_explained(emp_beta)
    
# get contact map and calculate neighborhoods
bin_cm_sub = data_utils.get_his3p_binarized_contact_map()
V = structure_utils.contact_map_to_neighborhoods(bin_cm_sub)

# calculate the coefficient variances corresponding to neighborhoods
gnk_beta_var_ = gnk_model.calc_beta_var(L, qs, V)
gnk_beta_var = gnk_beta_var_ / np.sum(gnk_beta_var_) # normalize beta
gnk_sparsity = np.count_nonzero(gnk_beta_var)
pv_at_gnk = 100*bm_fv[gnk_sparsity]
M = np.prod(qs)
num_samples = int(np.ceil(gnk_sparsity*C_calculation.C_VAL*np.log10(M)))
print("Sparsity of His3p(big) Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover His3p(big) GNK: %s" % num_samples)
print("Percent variance explained by largest %i His3p(big) empirical coefficients: %.3f" % (gnk_sparsity, pv_at_gnk))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained_from_beta_var(gnk_beta_var_, samples=1000, up_to=1000)

# Load LASSO results
results_dict = np.load("../results/his3p_big_lasso_results.npy", allow_pickle=True).item()
example_results = np.load("../results/his3p_big_lasso_example.npy")

# Determine how many coefficients of each order:
num_per = [0]* (L+1)
all_U = list(utils.powerset(range(L)))
for U in all_U:
    sz = len(U)
    if len(U) == 0:
        n = 1
    else:
        n = np.prod([qs[k-1]-1 for k in U])

    num_per[sz] += n
up_to = [np.sum([num_per[i] for i in range(j+1)]) for j in range(L+1)]


########################
### Make large panel ###
########################

fig = plt.figure(figsize=(12, 3))
gs = fig.add_gridspec(1,4)



# plot beta comparison
his3p_big_kwargs = {
    'max_order': 4,
    'order_lbl_offset': 20,
    'order_lbl_height': 0.35,
    'arrow1_xy': (4, 0.39),
    'arrow1_text_xy': (38, 0.48),
    'arrow2_xy': (96, 0.34),
    'arrow2_text_xy': (100, 0.42),
    'yticks': (-0.4, -0.2, 0, 0.2, 0.4),
    'yticklabels': ('0.40', '0.20', '0', '0.20', '0.40')
}

ax = fig.add_subplot(gs[0, 0:2])
num_coeffs = 4297
ticks = [0] + up_to
ticks = [t for t in ticks if t <= num_coeffs]
plot_utils.plot_beta_comparison(ax, L, num_coeffs, beta_mag_sq, gnk_beta_var, 
                                use_order_labels=True, 
                                ticks=ticks, **his3p_big_kwargs)



# plot variance explained
ax = fig.add_subplot(gs[0, 2])
plot_utils.plot_percent_variance_explained(ax, 100*bm_fv, 
                                           100*gnk_fv_mean, 
                                           100*gnk_fv_std,
                                           gnk_sparsity,
                                           xlim=800,
                                           xticks=(0, 200, 400, 600, 800)
                                          )

##########################
### plot LASSO results ###
##########################

ax = fig.add_subplot(gs[0, 3])

plot_utils.plot_lasso_results(ax, results_dict, num_samples, 
                   xlim=20000, print_name='His3p(big)')
axins = inset_axes(ax, width="75%", height="95%",
                   bbox_to_anchor=(0.5, 0.15, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)

his3p_big_inset_kwargs = {
    'xlim': (-0.1, 1.7),
    'ylim': (-0.6, 1.6),
    'xticks': (0, 0.5, 1, 1.5),
    'yticks': (-0.5, 0, 0.5, 1, 1.5),
    'text_xy': (0.9, -0.3)
}

plot_utils.plot_lasso_example_inset(axins, example_results, **his3p_big_inset_kwargs)

plt.tight_layout() 
plt.savefig("plots/supp_fig_his3p_big.png", dpi=500, bbox_inches='tight')