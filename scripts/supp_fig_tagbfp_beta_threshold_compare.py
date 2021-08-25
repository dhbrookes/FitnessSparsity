import sys
sys.path.append("../src")
import numpy as np
import data_utils
import gnk_model
import structure_utils
import plot_utils
import matplotlib.pyplot as plt
from scipy.special import binom

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['seaborn-deep', '../scripts/plots/paper.mplstyle'])

"""
This script makes the plot comparing empirical and GNK coefficients for 
different settings of the structural contact threshold.
"""

L = 13
q = 2
mtag_dists = data_utils.get_mtagbfp_contact_map()
mtag_beta = data_utils.calculate_mtagbfp_wh_coefficients()
mtag_beta_mag = mtag_beta**2 / np.sum(mtag_beta**2)  # normalize sum of squares to one
fig, axes = plt.subplots(4, 2, figsize=(6, 8))
axes = axes.flatten()
num_coeffs = int(np.sum([binom(13, i) for i in range(6)]))
thresholds = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

for i, thresh in enumerate(thresholds): 
    mtag_bin_cm = structure_utils.binarize_contact_map(mtag_dists, threshold=thresh)
    mtag_V = structure_utils.contact_map_to_neighborhoods(mtag_bin_cm)
    mtag_gnk_beta_var_ = gnk_model.calc_beta_var(L, q, mtag_V)
    mtag_gnk_beta_var = mtag_gnk_beta_var_/np.sum(mtag_gnk_beta_var_)
    ax = axes[i]
    if i==0:
        use_order=True
    else:
        use_order=False
    plot_utils.plot_beta_comparison(ax, L, num_coeffs, 
                                    mtag_beta_mag, 
                                    mtag_gnk_beta_var, 
                                    use_order_labels=use_order,
                                    order_label_fontsize=7,
                                    arrow2_text_xy=(83, 0.56)
                                    )
    ax.set_ylabel('')
    if i == 0:
        ax.set_ylabel('Magnitude of Fourier coefficient', fontsize=8)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    ax.set_title("Structural contact threshold: %s $\AA$" % thresh, fontsize=8)
    
plt.tight_layout()
plt.savefig('plots/supp_fig_mtagbfp_beta_threshold_compare.png', dpi=500)