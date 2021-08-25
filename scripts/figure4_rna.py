import sys
sys.path.append("../src")
import numpy as np
import RNA
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
import rna_utils
import plot_utils

plt.style.use(['seaborn-deep', '../paper.mplstyle'])


q = 4
L = 8

positions = rna_utils.RNA_POSITIONS

# get Fourier coefficients
emp_beta = rna_utils.calculate_rna_fourier_coefficients()

# Calculate fraction variance explained by empirical coefficients
beta_mag_sq = emp_beta**2 / np.sum(emp_beta**2)  # normalize sum of squares to one
bm_fv = utils.calc_frac_var_explained(emp_beta)

# calculate the coefficient variances corresponding to neighborhoods
gnk_beta_var_, V = rna_utils.calculate_rna_gnk_wh_coefficient_vars(return_neighborhoods=True, 
                                                                   pairs_from_scratch=False)
gnk_beta_var = gnk_beta_var_/np.sum(gnk_beta_var_) # normalize sum of variances to one
gnk_sparsity = np.count_nonzero(gnk_beta_var)
pv_at_gnk = 100*bm_fv[gnk_sparsity]
pred_num_samples = int(np.ceil(gnk_sparsity*C_calculation.C_VAL*np.log10(q**L)))

print("Sparsity of RNA Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover RNA GNK: %s" % pred_num_samples)
print("Percent variance explained by largest %i RNA empirical coefficients: %.3f" % (gnk_sparsity, pv_at_gnk))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained_from_beta_var(gnk_beta_var_, samples=1000, up_to=1500)

# Load LASSO results
results_dict = np.load("../results/rna_lasso_results.npy", allow_pickle=True).item()
example_results = np.load("../results/rna_lasso_example.npy")

########################
### Make large panel ###
########################

fig = plt.figure(figsize=(15, 3))
gs = fig.add_gridspec(1,5)

# plot neighborhoods
ax = fig.add_subplot(gs[0, 0])
plot_utils.plot_neighborhoods(ax, V, L, 
                              positions, 
                              label_rotation=0, s=250)


# plot beta comparison
ax = fig.add_subplot(gs[0, 1:3])

rna_kwargs = {
    'max_order': 4,
    'order_lbl_offset': 100,
    'order_lbl_height': 0.25,
    'arrow1_xy': (15, 0.31),
    'arrow1_text_xy': (110, 0.375),
    'arrow2_xy': (150, 0.28),
    'arrow2_text_xy': (250, 0.34),
    'yticks': (-0.3, -0.15, 0, 0.15, 0.3),
    'yticklabels': ('0.30', '0.15', '0', '0.15', '0.30')
}

# fig, ax = plt.subplots(figsize=(6, 3))
ticks = [np.sum([binom(L, j)*(q-1)**j for j in range(i)]) for i in range(rna_kwargs['max_order']+2)]
num_coeffs = int(ticks[-1])+1
# ticks = [t for t in ticks if t <= num_coeffs+1]
plot_utils.plot_beta_comparison(ax, L, num_coeffs, beta_mag_sq, 
                                gnk_beta_var, 
                                use_order_labels=True,
                                ticks=ticks, **rna_kwargs)



# plot percent variance explained
ax = fig.add_subplot(gs[0, 3])
plot_utils.plot_percent_variance_explained(ax, 100*bm_fv, 
                                           100*gnk_fv_mean, 
                                           100*gnk_fv_std,
                                           gnk_sparsity,
                                           xlim=1500,
                                           xticks=(0, 250, 500, 750, 1000, 1250, 1500)
                                          )

# plot LASSO results
ax = fig.add_subplot(gs[0, 4])
plot_utils.plot_lasso_results(ax, results_dict, pred_num_samples, xlim=25000)
axins = inset_axes(ax, width="75%", height="95%",
                   bbox_to_anchor=(0.5, 0.25, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)

rna_inset_kwargs = {
    'xlim': (-33, -18),
    'ylim': (-33, -18),
    'xticks': (-0.5, 0, 0.5, 1, 1.5),
    'yticks': (0, 0.5, 1),
    'text_xy': (-26, -30),
    'xticks': (-30, -25, -20),
    'yticks': (-30, -25, -20)
}


plot_utils.plot_lasso_example_inset(axins, example_results, **rna_inset_kwargs)

plt.tight_layout()
plt.savefig("plots/figure4_rna.png", dpi=500, bbox_inches='tight', transparent=True)
