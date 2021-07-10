import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import gnk_model
import utils

plt.style.use('seaborn-deep')
rcParams['savefig.dpi'] = 500
rcParams['lines.linewidth'] = 1.0
rcParams['axes.grid'] = True
rcParams['axes.spines.right'] = True
rcParams['axes.spines.top'] = True
rcParams['grid.color'] = 'gray'
rcParams['grid.alpha'] = 0.2
rcParams['axes.linewidth'] = 0.5
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

"""
This script produces Figure 1.
"""

L = 9
K = 3
cm = sns.color_palette("crest", as_cmap=True)

Vs = [
    gnk_model.sample_random_neighborhoods(L, K),
    gnk_model.build_adj_neighborhoods_periodic(L, K), 
    gnk_model.build_block_neighborhoods(L, K)
]

fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    V = Vs[i]
    for j in range(L):
        y = [L-j-0.5 for _ in range(len(V[j]))]
        x = [v-0.5 for v in V[j]]
        ax.scatter(x, y, facecolor=cm(0.9), marker='s',s=150)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L+1))
    ax.xaxis.tick_top()

    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator([i+0.5 for i in range(L)]))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.FixedLocator([i+0.5 for i in range(L)]))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter([str(i+1) for i in range(L)]))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(['$V^{[%i]}$' % (L-i) for i in range(L)]))
    ax.set_xlim([0, L])
    ax.set_aspect('equal', adjustable='box')
    
    ax.tick_params(axis='x', which='both', length=0, pad=1)
    ax.tick_params(axis='y', which='both', length=0)
    
plt.tight_layout()
plt.savefig('plots/neighborhood_schemes_.png', dpi=500, facecolor='white', transparent=False)
plt.show()