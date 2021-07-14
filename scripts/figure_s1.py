import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.special import binom
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gnk_model
import C_calculation
import utils
import structure_utils
import data_utils

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
beta = data_utils.calculate_his3p_big_wh_coefficients()

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
gnk_beta_var_ = gnk_model.calc_beta_var(L, qs, V)
gnk_beta_var = gnk_beta_var_ / np.sum(gnk_beta_var_) # normalize beta
gnk_sparsity = np.count_nonzero(gnk_beta_var)
percent_var = 100*bm_fv[gnk_sparsity]
M = np.prod(qs)
num_samples = int(np.ceil(gnk_sparsity*C_calculation.C_VAL*np.log10(M)))
print("Sparsity of His3p(big) Structural GNK model: %i" % gnk_sparsity)
print("Number of samples to recover GNK: %s" % num_samples)
print("Percent variance explained by largest %i empirical coefficients: %.3f" % (gnk_sparsity, percent_var))

# calculate fraction variance explained by samples of GNK coefficients
gnk_fv_mean, gnk_fv_std = utils.calc_frac_var_explained(gnk_beta_var_, samples=1000, up_to=1000)

# Load LASSO results
results_dict = np.load("../results/his3p_big_lasso_results.npy", allow_pickle=True).item()
ns = results_dict['n']
pearson = results_dict['pearson']
mean_r = np.mean(pearson**2, axis=1)
std_r = np.std(pearson**2, axis=1)
idx = np.where(ns==num_samples)[0][0]
r2_val = mean_r[idx]
print("LASSO R^2 at GNK predicted number of samples: %.3f" % r2_val)

# Load example results
example_results = np.load("../results/his3p_big_lasso_example.npy")
y = example_results[0]
pred = example_results[1]
r2_test = pearsonr(pred, y)[0]**2
m, b = np.polyfit(y ,pred, 1)
minmin = np.min([np.min(pred), np.min(y)])
maxmax = np.max([np.max(pred), np.max(y)])

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
########################
### Make large panel ###
########################
########################

fig = plt.figure(figsize=(12, 3))
gs = fig.add_gridspec(1,4)

colors = sns.color_palette('Set1', n_colors=2)

##########################
## Plot beta comparison ##
##########################

ax = fig.add_subplot(gs[0, 0:2])
num = 4297
plot_gnk_vals = np.sqrt(gnk_beta_var)
plot_bb_vals = np.sqrt(beta**2/np.sum(beta**2))
mv = np.max([np.max(plot_gnk_vals), np.max(plot_bb_vals)])

ax.bar(range(len(plot_bb_vals[:num])), plot_bb_vals[:num], width=5, facecolor=colors[1])
ax.bar(range(len(plot_gnk_vals[:num])), -plot_gnk_vals[:num], width=5, facecolor=colors[0])

ax.plot((-10, num),(0, 0), c='k')
ax.spines['bottom'].set_visible(False)
ticks = [0] + up_to
ticks = [t for t in ticks if t <= num]
ordlbls = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th"]
for i, tick in enumerate(ticks):
    ax.plot((tick, tick), (-1, 1), c='k', ls='--', lw=0.5, alpha=0.5, zorder=0)
    if i > 2 and i < 5:
        ax.text(tick+20, 0.35, "$r=%i$" %i)

ax.annotate("",
            xy=(ticks[2]+75, 0.34), xycoords='data',
            xytext=(100, 0.42), textcoords='data',
            arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                            connectionstyle="arc3,rad=0.15"),
            )

ax.text(100, 0.42, "$r=2$")

ax.annotate("",
            xy=(4, 0.39), xycoords='data',
            xytext=(38, 0.48), textcoords='data',
            arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                            connectionstyle="arc3,rad=0.15"),
            )

ax.text(30, 0.48, "$r=1$")
ax.tick_params(axis='y', which='major', direction='out')
ax.set_xticks([])
ax.set_ylim([-mv, mv])
ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_yticklabels(['0.4', '0.2', '0', '0.2', '0.4'])
ax.set_xlim([-10, num+10])
ax.set_ylabel("Magnitude of Fourier coefficient", labelpad=2, fontsize=12)
ax.grid(False)

###############################
### plot variance explained ###
###############################

ax = fig.add_subplot(gs[0, 2])
ax.plot(range(len(bm_fv)), 100*bm_fv, c=colors[1], label='Empirical', lw=1.5, zorder=1)
ax.plot(range(len(gnk_fv_mean)), 100*gnk_fv_mean, c=colors[0], label='GNK mean', lw=1.5)
ax.fill_between(range(len(gnk_fv_mean)), 100*(gnk_fv_mean-gnk_fv_std), 
                100*(gnk_fv_mean+gnk_fv_std), color=colors[0], alpha=0.4, 
                edgecolor='none', label='GNK std. dev.', zorder=10)

gnk_sparsity = np.count_nonzero(gnk_beta_var)
percent_var = 100*bm_fv[gnk_sparsity]
ax.plot((gnk_sparsity, gnk_sparsity), (0, percent_var), ls='--', c='k', lw=0.75, zorder=0)
ax.plot((0, gnk_sparsity), (percent_var, percent_var), ls='--', c='k', lw=0.75)
ax.scatter([gnk_sparsity], [percent_var], edgecolor='k', facecolor='none', zorder=11, s=15, linewidth=0.75)

ax.set_xlim([0, 800])
ax.set_ylim([0, 101])
ax.set_ylabel("Percent variance explained", labelpad=2, fontsize=12)
ax.set_xlabel("Number of largest coefficients", fontsize=12)

leg = ax.legend(fontsize=10, labelspacing=0.15, bbox_to_anchor=(0.05,0), loc='lower left')
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square', pad=0.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

axes = fig.get_axes()
plt.tight_layout()

##########################
### plot LASSO results ###
##########################

ax = fig.add_subplot(gs[0, 3])
colors = sns.color_palette('Set1')
c = colors[1]

ax.set_xlim([0, 20000])
ax.set_ylim([0, 1])
ax.errorbar(ns[1:-1], mean_r[1:-1], yerr=std_r[1:-1], lw=0.5, marker='o', markersize=0, c='k', zorder=12, fmt='none', capsize=1, markeredgewidth=0.5)
ax.plot(ns[1:-1], mean_r[1:-1], c=c, lw=1, marker='o', markersize=3)

ax.plot((num_samples, num_samples), (0, r2_val), ls='--', c='k', lw=0.75, zorder=0)
ax.plot((0, num_samples), (r2_val, r2_val), ls='--', c='k', lw=0.75)
ax.scatter([num_samples], [r2_val], edgecolor='k', facecolor=colors[0], zorder=11, s=15, linewidth=0.75)

ax.tick_params(axis='y', which='major', right=False, labelright=False)
ax.tick_params(axis='x', which='major', top=False, labeltop=False)
ax.set_ylabel("Prediction $R^2$", fontsize=12)
ax.set_xlabel("Number of training samples", fontsize=12)

axins = inset_axes(ax, width="75%", height="95%",
                   bbox_to_anchor=(0.5, 0.15, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)
axins.set_xlabel("Empirical fitness", fontsize=9, labelpad=1)
axins.set_ylabel("Predicted fitness", fontsize=9, labelpad=1)
axins.scatter(y, pred, s=1, c='k')
axins.set_xticks([0, 0.5, 1, 1.5])
axins.set_yticks([0, 0.5, 1, 1.5])
xs = np.linspace(minmin, maxmax+0.1, 1000)
axins.plot(xs, m*xs+b, c=colors[0], ls='--')
axins.set_ylim([-0.6, 1.6])
axins.set_xlim([-0.1, 1.7])
axins.text(0.9, -0.3, "$R^2=%.2f$" % r2_test, fontsize=7, color=colors[0])
axins.tick_params(axis='y', which='major', right=False, labelright=False, labelsize=8)
axins.tick_params(axis='x', which='major', top=False, labeltop=False, labelsize=8)
axins.grid(False)

plt.tight_layout() 
plt.savefig("plots/figure_s1.png", dpi=500, bbox_inches='tight', facecolor='white', transparent=False)