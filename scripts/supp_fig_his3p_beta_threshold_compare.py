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

plt.style.use(['seaborn-deep', '../paper.mplstyle'])

"""
This script makes the plot comparing empirical and GNK coefficients for 
different settings of the structural contact threshold.
"""

L = 11
q = 2
his3p_dists = data_utils.get_his3p_contact_map()
his3p_beta = data_utils.calculate_his3p_small_wh_coefficients()
his3p_beta_mag = his3p_beta**2 / np.sum(his3p_beta**2)  # normalize sum of squares to one
fig, axes = plt.subplots(4, 2, figsize=(6, 8))
axes = axes.flatten()
num_coeffs = int(np.sum([binom(L, i) for i in range(7)]))
thresholds = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

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

for i, thresh in enumerate(thresholds): 
    his3p_bin_cm = structure_utils.binarize_contact_map(his3p_dists, threshold=thresh)
    his3p_V = structure_utils.contact_map_to_neighborhoods(his3p_bin_cm)
    his3p_gnk_beta_var_ = gnk_model.calc_beta_var(L, q, his3p_V)
    his3p_gnk_beta_var = his3p_gnk_beta_var_/np.sum(his3p_gnk_beta_var_)
    ax = axes[i]
    if i==0:
        use_order=True
    else:
        use_order=False
    plot_utils.plot_beta_comparison(ax, L, num_coeffs, 
                                    his3p_beta_mag, 
                                    his3p_gnk_beta_var, 
                                    use_order_labels=use_order,
                                   order_label_fontsize=7,
                                   **his3p_kwargs)
    if i == 0:
        ax.set_ylabel('Magnitude of Fourier coefficient', fontsize=8)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    ax.set_title("Structural contact threshold: %s$\AA$" % thresh, fontsize=8)
    
plt.tight_layout()
plt.savefig('plots/supp_fig_his3p_beta_threshold_compare.png', dpi=500)