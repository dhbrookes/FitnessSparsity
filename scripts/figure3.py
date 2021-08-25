import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import gnk_model
import C_calculation

plt.style.use(['seaborn-deep', '../paper.mplstyle'])

"""
This script produces Figure 3. Run as:

$ python figure3.py

"""

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

##################################
### Length dependence (Fig 3A) ###
##################################

ax = axes[0]
Ks = list(range(1, 6))
q = 2
SLs = []
Ls = []
for i, K in enumerate(Ks):
    L_vals = np.arange(K+1, 101, 1)
    Ls.append(L_vals)
    SL = np.zeros(len(L_vals), dtype=int)
    for j, L in enumerate(L_vals):
        SL[j] = C_calculation.C_VAL*L*np.log10(q)*gnk_model.calc_max_rn_sparsity(L, q, K)
    SLs.append(SL)
    
colors = sns.color_palette('rocket_r', n_colors=len(Ks))
for i, K in enumerate(Ks[::-1]):
    L_vals = Ls[-(i+1)]
    SL = SLs[-(i+1)]
    ax.plot(L_vals, SL, label="$K=%i$" % K, c=colors[K-1])

ax.set_ylabel("$N$ for exact recovery", fontsize=14)     
ax.set_xlabel("$L$", fontsize=14, labelpad=2)
ax.set_yscale('log')
ax.set_xlim(0, 102)
ax.set_xticks([0, 25, 50, 75, 100])

leg = ax.legend(fontsize=10, labelspacing=0.15)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square', pad=0.05)


#########################################
### Alphabet size dependence (Fig 2B) ###
#########################################

ax = axes[1]
L = 20
qs = np.arange(2, 21, 1)
Ks = [1, 2, 3, 4, 5]

Sq2 = np.zeros((len(Ks), len(qs)), dtype=int)
for i, K in enumerate(Ks):
    for j, q in enumerate(qs):
        Sq2[i, j] = C_calculation.C_VAL*L*np.log10(q)*gnk_model.calc_max_rn_sparsity(L, q, K)

colors = sns.color_palette('rocket_r', n_colors=len(Ks))
q_select = np.array([2, 4, 20])
q_select_idx = np.array([j for j in range(len(qs)) if qs[j] in q_select])
for i, K in enumerate(Ks[::-1]):
    ax.plot(qs, Sq2[-(i+1)], label="$K=%i$" % K, c=colors[K-1], zorder=10-i)
    ax.scatter(q_select, Sq2[-(i+1), q_select_idx], edgecolor=colors[K-1], facecolor='w', s=20, zorder=12-i)
    
ax.set_ylabel("$N$ for exact recovery", fontsize=14)    
ax.set_xlabel("$q$", fontsize=14, labelpad=1)
ax.set_yscale('log')
ax.set_ylim([10**2, 10**10])

plt.tight_layout()
plt.savefig('plots/figure3.png', dpi=500, bbox_inches='tight')