import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import gnk_model
import utils

plt.style.use('seaborn-deep')
rcParams['savefig.dpi'] = 500
rcParams['lines.linewidth'] = 1.0
rcParams['axes.grid'] = True
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
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
This script produces Figure 2.
"""

fig, axes = plt.subplots(2, 2, figsize=(6, 6))

##################################
### Length dependence (Fig 2A) ###
##################################

ax = axes[0, 0]

Ks = list(range(1, 6))
q = 2
SLs = []
Ls = []
for i, K in enumerate(Ks):
    L_vals = np.arange(K+1, 101, 1)
    Ls.append(L_vals)
    SL = np.zeros(len(L_vals), dtype=int)
    for j, L in enumerate(L_vals):
        SL[j] = gnk_model.calc_max_rn_sparsity(L, q, K)
    SLs.append(SL)
    
colors = sns.color_palette('rocket_r', n_colors=len(Ks))
for i, K in enumerate(Ks[::-1]):
    L_vals = Ls[-(i+1)]
    SL = SLs[-(i+1)]
    ax.plot(L_vals, SL, label="$K=%i$" % K, c=colors[K-1])

ax.set_ylabel("Sparsity", fontsize=14)    
ax.set_xlabel("$L$", fontsize=14)
ax.set_yscale('log')
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])

leg = ax.legend(fontsize=10, labelspacing=0.15)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square', pad=0.05)

#########################################
### Alphabet size dependence (Fig 2B) ###
#########################################

L = 20
qs = np.arange(2, 21, 1)
Ks = [1, 2, 3, 4, 5]

Sq2 = np.zeros((len(Ks), len(qs)), dtype=int)
for i, K in enumerate(Ks):
    for j, q in enumerate(qs):
        Sq2[i, j] = gnk_model.calc_max_rn_sparsity(L, q, K)

ax = axes[0, 1]
colors = sns.color_palette('rocket_r', n_colors=len(Ks))
q_select = np.array([2, 4, 20])
q_select_idx = np.array([j for j in range(len(qs)) if qs[j] in q_select])
for i, K in enumerate(Ks[::-1]):
    ax.plot(qs, Sq2[-(i+1)], label="$K=%i$" % K, c=colors[K-1], zorder=10-i)
    ax.scatter(q_select, Sq2[-(i+1), q_select_idx], edgecolor=colors[K-1], facecolor='w', s=20, zorder=12-i)

ax.set_ylabel("Sparsity", fontsize=14)    
ax.set_xlabel("$q$", fontsize=14)
ax.set_yscale('log')
        
########################################
### Neighborhood dependence (Fig 2C) ###
########################################

L = 20
q = 2
rn_Ks = list(range(1, int(L/2)+1))
an_Ks = rn_Ks
bn_Ks = utils.divisors(L)
S_rn_mean = np.zeros(len(rn_Ks))
S_rn_max = np.zeros(len(rn_Ks))
S_an = np.zeros(len(an_Ks))
S_an_per = np.zeros(len(an_Ks))
S_bn = np.zeros(len(bn_Ks))

for i, K in enumerate(rn_Ks):
    S_rn_mean[i] = gnk_model.calc_mean_rn_sparsity(L, q, K)
    S_rn_max[i] = gnk_model.calc_max_rn_sparsity(L, q, K)

for i, K in enumerate(an_Ks):
    S_an[i] = gnk_model.calc_an_sparsity(L, q, K)
    S_an_per[i] = 1 + L*2**(K-1)
    
for i, K in enumerate(bn_Ks):
    S_bn[i] = gnk_model.calc_bn_sparsity(L, q, K) 
    
ax = axes[1, 0]
colors = sns.color_palette('Set1', n_colors=3)
ax.plot(rn_Ks, S_rn_max, c=colors[1],ls='--', marker='.', label='Uniform Bound', markersize=5)
ax.plot(rn_Ks, S_rn_mean, c=colors[1], marker='.', label='Random Expectation', markersize=5)
ax.plot(an_Ks, S_an_per, c=colors[2], marker='D', label='Adjacent', markersize=3)
ax.plot(bn_Ks, S_bn, c=colors[0], marker='^', label='Block', markersize=3)
ax.set_ylabel("Sparsity", fontsize=14)    
ax.set_xlabel("$K$", fontsize=14)
ax.xaxis.major.formatter._useMathText = True

leg = ax.legend(fontsize=10, labelspacing=0.2)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square', pad=0.1)

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3,2))
ax.yaxis.set_major_formatter(formatter)
plt.tight_layout()  ## keep this here
    
###########################
### C recovery (Fig 2D) ###
###########################

ax = axes[1, 1]
C_recovery_dict = np.load("results/C_recovery.npy", allow_pickle=True).item()
Cs = C_recovery_dict['Cs']
frac_recovered = C_recovery_dict['frac_recovered']
overall = np.mean(frac_recovered, axis=0)
colors = sns.color_palette('Set1', n_colors=5)
for i in range(frac_recovered.shape[0]):
    fr = frac_recovered[i]
    ax.plot(Cs, fr, c='k', alpha=0.2, lw=1)
    ax.plot(Cs, overall, c=colors[0], lw=2.5)
    
ax.plot((2.62, 2.62), (0, 1), c='k', lw=1.5, ls='--')

ax.set_xticks([0, 0.5,1, 1.5, 2, 2.5, 3])
ax.set_xlabel("$C$", fontsize=14)
ax.set_ylabel("Fraction Recovered at $C \cdot S \log(q^L)$", fontsize=12)
ax.set_xlim([0, 3])
ax.set_ylim([-0.01, 1.01])

plt.savefig('plots/gnk_sparsity.png', dpi=300, bbox_inches='tight', facecolor='white', transparent=False)