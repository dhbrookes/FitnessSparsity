import sys
sys.path.append("../src")
import matplotlib.pyplot as plt
import numpy as np

import data_utils
import gnk_model
import structure_utils

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['seaborn-deep', 'plots/paper.mplstyle'])

"""
This script produces Figure S2, which shows the sparsity of
GNK fitness functions with Structural Neighborhoods as a function
of the cutoff distance for structural contacts.

$ python figure_s2.py

"""

thresholds = np.arange(1, 7, 0.01)
mtag_sparsity = []
his3p_sparsity = []
for thresh in thresholds: 
    mtag_bin_cm = data_utils.get_mtagbfp_binarized_contact_map(threshold=thresh)
    mtag_V = structure_utils.contact_map_to_neighborhoods(mtag_bin_cm)
    mtag_gnk_beta_var = gnk_model.calc_beta_var(13, 2, mtag_V)
    mtag_sparsity.append(np.count_nonzero(mtag_gnk_beta_var))
    
    his3p_bin_cm = data_utils.get_his3p_binarized_contact_map(threshold=thresh)
    his3p_V = structure_utils.contact_map_to_neighborhoods(his3p_bin_cm)
    his3p_gnk_beta_var = gnk_model.calc_beta_var(13, 2, his3p_V)
    his3p_sparsity.append(np.count_nonzero(his3p_gnk_beta_var))

main_val = 4.5
idx = np.where(np.abs(mtag_sparsity['thresholds']- main_val) < 1e-12 )[0][0]


fig, axes = plt.subplots(1, 2, figsize=(6, 3))

ax = axes[0]
ax.plot(thresholds, mtag_sparsity, lw=2)
val = mtag_sparsity[idx]
ax.plot((main_val, main_val), (0, val), ls='--', c='k', lw=0.75, zorder=0)
ax.plot((0, main_val), (val, val), ls='--', c='k', lw=0.75)
ax.scatter([main_val], [val], edgecolor='k', facecolor='none', zorder=11, s=15, linewidth=0.75)
ax.set_xlim([1, 7])
ax.set_ylim([0, 75])
ax.set_ylabel("Sparsity", fontsize=12)
ax.set_xlabel("Contact cutoff distance ($\AA$)", fontsize=12)
ax.set_xticks([1, 2, 3, 4, 4.5, 5, 6, 7])
ax.set_yticks([0, 20, 40, val, 60, 80])

ax = axes[1]
ax.plot(thresholds, his3p_sparsity, lw=2)
val = his3p_sparsity['sparsity'][idx]
ax.plot((main_val, main_val), (0, val), ls='--', c='k', lw=0.75, zorder=0)
ax.plot((0, main_val), (val, val), ls='--', c='k', lw=0.75)
ax.scatter([main_val], [val], edgecolor='k', facecolor='none', zorder=11, s=15, linewidth=0.75)
ax.set_xlim([1, 7])

ax.set_ylabel("Sparsity", fontsize=12)
ax.set_xlabel("Contact cutoff distance ($\AA$)", fontsize=12)
ax.set_xticks([1, 2, 3, 4, 4.5, 5, 6, 7])
ax.set_yticks([0, 25, 50, val, 100])
ax.set_ylim([0, 120])

plt.tight_layout()
plt.savefig("plots/figure_s2.png", dpi=500, bbox_inches='tight', facecolor='white')