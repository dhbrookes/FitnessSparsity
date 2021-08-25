import sys
sys.path.append("../src")
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt

import gnk_model
import utils
import plot_utils

plt.style.use(['seaborn-deep', '../paper.mplstyle'])
rcParams['axes.spines.right'] = True
rcParams['axes.spines.top'] = True

"""
This script produces Figure 1. Run as:

$ python figure1.py

"""

L = 9
K = 3
cm = sns.color_palette("crest", as_cmap=True)

Vs = [
    gnk_model.sample_random_neighborhoods(L, K),
    gnk_model.build_adj_neighborhoods(L, K), 
    gnk_model.build_block_neighborhoods(L, K)
]

fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    V = Vs[i]
    plot_utils.plot_neighborhoods(ax, V, L, range(1, L+1), label_rotation=0, s=120)
    
plt.tight_layout()
plt.savefig('plots/figure1.png', dpi=500)