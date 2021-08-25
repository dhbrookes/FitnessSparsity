import sys
sys.path.append("../src")
import numpy as np
import plot_utils
import data_utils
import gnk_model
import structure_utils
import matplotlib.pyplot as plt
from scipy.special import binom

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['seaborn-deep', '../paper.mplstyle'])

"""
This script makes the plots comparing Kernel Density Estimates of empirical coefficients
identified as zero and nonzero by GNK models with Structural neighborhoods.
"""

# get TagBFP coefficients
L = 13
q = 2
mtag_beta = data_utils.calculate_mtagbfp_wh_coefficients()
mtag_beta_mag = mtag_beta**2 / np.sum(mtag_beta**2)  # normalize sum of squares to one
mtag_gnk_beta_var_ = data_utils.calculate_mtagbfp_gnk_wh_coefficient_vars()
mtag_gnk_beta_var = mtag_gnk_beta_var_/np.sum(mtag_gnk_beta_var_)

# get His3p beta coefficients
L = 11
q = 2
his3p_beta = data_utils.calculate_his3p_small_wh_coefficients()
his3p_beta_mag = his3p_beta**2 / np.sum(his3p_beta**2)  # normalize sum of squares to one
his3p_gnk_beta_var_ = data_utils.calculate_his3p_small_gnk_wh_coefficient_vars()
his3p_gnk_beta_var = his3p_gnk_beta_var_/np.sum(his3p_gnk_beta_var_)


# make plots comparing all coefficients:
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
plot_utils.compare_gnk_id_coeffs(axes[0], mtag_beta_mag, mtag_gnk_beta_var, 
                                 bandwidth=0.01, xmax=0.3, nz_bar_height=-3.15)
_, pval = plot_utils.wilcoxon_rank_sum(mtag_beta_mag, mtag_gnk_beta_var)
axes[0].text(0.15, 20, "$p=$" + plot_utils.pp_sci_not(pval))

plot_utils.compare_gnk_id_coeffs(axes[1], his3p_beta_mag, his3p_gnk_beta_var, bandwidth=0.01)
_, pval = plot_utils.wilcoxon_rank_sum(his3p_beta_mag, his3p_gnk_beta_var)
axes[1].text(0.2, 15, "$p=$" + plot_utils.pp_sci_not(pval))

plt.tight_layout()
plt.savefig("plots/supp_fig_coeff_kdes.png", dpi=500)


# make plot comparing coefficients of each order for mTagBFP2
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
L = 13
q = 2
axes = axes.flatten()
start = 1
for r in range(2, 6):
    ax = axes[r-2]
    num = int(binom(L, r))
    emp_r = mtag_beta_mag[start:start+num]
    gnk_r = mtag_gnk_beta_var[start:start+num]
    plot_utils.compare_gnk_id_coeffs(ax, emp_r, gnk_r, bandwidth=0.01, xmax=0.3, nz_bar_height=-3.15)
    _, pval = plot_utils.wilcoxon_rank_sum(emp_r, gnk_r)
    ax.text(0.15, 15, "$p=$" + plot_utils.pp_sci_not(pval))
    ax.set_title("$r=%s$" % r)
    if r > 2:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()
    start += num
    
plt.tight_layout()
plt.savefig("plots/supp_fig_mtagbfp_kde_by_order.png", dpi=500)


# make plot comparing coefficients of each order for His3p
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
L = 11
q = 2
axes = axes.flatten()
start = 1
for r in range(2, 6):
    ax = axes[r-2]
    num = int(binom(L, r))
    emp_r = his3p_beta_mag[start:start+num]
    gnk_r = his3p_gnk_beta_var[start:start+num]
    plot_utils.compare_gnk_id_coeffs(ax, emp_r, gnk_r, bandwidth=0.01, xmax=0.4, nz_bar_height=-3.15)
    _, pval = plot_utils.wilcoxon_rank_sum(emp_r, gnk_r)
    ax.text(0.2, 15, "$p=$" + plot_utils.pp_sci_not(pval))
    ax.set_title("$r=%s$" % r)
    if r > 2:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()
    start += num
plt.tight_layout()
plt.savefig("plots/supp_fig_mtagbfp_kde_by_order.png", dpi=500)