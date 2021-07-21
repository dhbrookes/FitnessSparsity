import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
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
This script produces the first row of Figure 4D, which displays the results
of various tests on the mTagBFP empirical fitness function, and prints
quantities related to this analysis. Run as:

$ python figure4_top.py

"""

q = 2
L = 13

# get WH coefficients
emp_beta = data_utils.calculate_mtagbfp_wh_coefficients()

# Calculate fraction variance explained by empirical coefficients
beta_mag_sq = emp_beta**2 / np.sum(emp_beta**2)  # normalize sum of squares to one
bm_fv = utils.calc_frac_var_explained(emp_beta)

# get contact map and calculate neighborhoods
bin_cm_sub = data_utils.get_mtagbfp_binarized_contact_map()
V = structure_utils.contact_map_to_neighborhoods(bin_cm_sub)

# calculate the coefficient variances corresponding to neighborhoods
gnk_beta_var_ = gnk_model.calc_beta_var(L, q, V)
gnk_beta_var = gnk_beta_var_/np.sum(gnk_beta_var_) # normalize beta
gnk_sparsity = np.count_nonzero(gnk_beta_var)
pv_at_gnk = 100*bm_fv[gnk_sparsity]
num_samples = int(np.ceil(gnk_sparsity*C_calculation.C_VAL*np.log10(q**L)))
print("Sparsity of mTagBFP Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover mTagBFP GNK: %s" % num_samples)
print("Percent variance explained by largest %i mTagBFP empirical coefficients: %.3f" % (gnk_sparsity, pv_at_gnk))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained_from_beta_var(gnk_beta_var_, samples=1000, up_to=76)

# Load LASSO results
results_dict = np.load("../results/mtagbfp_lasso_results.npy", allow_pickle=True).item()
ns = results_dict['n']
pearson = results_dict['pearson']
mean_r = np.mean(pearson**2, axis=1)
std_r = np.std(pearson**2, axis=1)
idx = np.where(ns==num_samples)[0][0]
r2_val = mean_r[idx]
print("LASSO R^2 at mTagBFP GNK predicted number of samples: %.3f" % r2_val)

# Load example results
example_results = np.load("../results/mtagbfp_lasso_example.npy")


########################
### Make large panel ###
########################

fig = plt.figure(figsize=(15, 3))
gs = fig.add_gridspec(1,5)

# plot neighborhoods
ax = fig.add_subplot(gs[0, 0])
plot_utils.plot_neighborhoods(ax, V, L, 
                              data_utils.MTAGBFP_POSITIONS, 
                              label_rotation=60, s=90)

# plot beta comparison
ax = fig.add_subplot(gs[0, 1:3])
num_coeffs = int(np.sum([binom(13, i) for i in range(6)])) # up to 5th order interactions
plot_utils.plot_beta_comparison(ax, L, num_coeffs, beta_mag_sq, gnk_beta_var, use_order_labels=True)


# plot percent variance explained
colors = sns.color_palette('Set1', n_colors=2)

ax = fig.add_subplot(gs[0, 3])
plot_utils.plot_percent_variance_explained(ax, 100*bm_fv, 
                                           100*gnk_fv_mean, 
                                           100*gnk_fv_std,
                                           gnk_sparsity,
                                           xlim=75,
                                           xticks=(0, 25, 50, 75)
                                          )


###########################
### plot LASSO results  ###
###########################


range1 = [0.45, 1]
range2 = [0, 0.08]
height_ratio = ((range1[1]-range1[0]) / (range2[1]-range2[0]))

gs_ = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 4], height_ratios=[height_ratio, 1])

ax1 = plt.subplot(gs_[0])
ax2 = plt.subplot(gs_[1])

colors = sns.color_palette('Set1')
c = colors[1]
# c = colors(0.6)
ax1.errorbar(ns, mean_r, yerr=std_r, lw=0.5, marker='o', markersize=0, 
             c='k', zorder=12, fmt='none', capsize=1, markeredgewidth=0.5)
ax2.errorbar(ns, mean_r, yerr=std_r, c='k', lw=0.5, marker='o', markersize=0, 
             zorder=12, fmt='none', capsize=1, markeredgewidth=0.5)
ax1.plot(ns, mean_r, c=c, lw=1, marker='o', markersize=3, zorder=10)
ax2.plot(ns, mean_r, c=c, lw=1, marker='o', markersize=3, zorder=10)

ax1.set_ylim(range1[0], range1[1])  
ax2.set_ylim(range2[0], range2[1])
ax1.set_xlim([0, 2000])
ax2.set_xlim([0, 2000])

ax1.plot((0, 2000), (range1[0]+0.002, range1[0]+0.002), lw=0.5,c='k', alpha=0.2)
ax2.plot((0, 2000), (range2[1], range2[1]), lw=0.5,c='k', alpha=0.2)

ax1.plot((num_samples, num_samples), (0, r2_val), ls='--', c='k', lw=0.75, zorder=0)
ax1.plot((0, num_samples), (r2_val, r2_val), ls='--', c='k', lw=0.75)
ax1.scatter([num_samples], [r2_val], edgecolor='k', facecolor=colors[0], 
            zorder=11, s=15, linewidth=0.75)

ax2.plot((num_samples, num_samples), (0, r2_val), ls='--', c='k', lw=0.75, zorder=0)
ax2.plot((0, num_samples), (r2_val, r2_val), ls='--', c='k', lw=0.75)

ax2.set_yticklabels(["0", "0.05"])

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - height_ratio*d, 1 + height_ratio*d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - height_ratio*d, 1 + height_ratio*d), **kwargs)

ax1.tick_params(axis='y', which='major', right=False, labelright=False)
ax1.tick_params(axis='x', which='major', top=False, labeltop=False)
ax2.tick_params(axis='y', which='major', right=False, labelright=False)
ax2.tick_params(axis='x', which='major', top=False, labeltop=False)

ax1.set_ylabel("Prediction $R^2$", fontsize=12, labelpad=2)
ax2.set_xlabel("Number of training samples",  fontsize=12, )
ax1.yaxis.set_label_coords(-0.1,0.4)

axins = inset_axes(ax1, width="80%", height="100%",
                   bbox_to_anchor=(0.45, 0.13, .6, .5),
                   bbox_transform=ax1.transAxes, loc=3)

plot_utils.plot_lasso_example_inset(axins, example_results)


plt.subplots_adjust(hspace=0.08)
plt.tight_layout()
plt.savefig("plots/figure4_top.png", dpi=500, bbox_inches='tight', 
            facecolor='white', transparent=False)
