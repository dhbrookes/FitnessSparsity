import sys
sys.path.append("../src")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import C_calculation

plt.style.use(['seaborn-deep', 'plots/paper.mplstyle'])

"""
This script produces Figure S3, which displays a detailed summary of the 
calculations used to determine a suitable value for C. Run as:

$ python figure_s3.py

"""


C_recovery_dict = np.load("../results/C_recovery.npy", allow_pickle=True).item()
Cs = C_recovery_dict['Cs']
frac_recovered = C_recovery_dict['frac_recovered']

Lqs = [(10, 2), (11, 2), (12, 2), (13, 2), 
       (6, 3), (7, 3), (8, 3), 
       (5, 4), (6, 4), (7, 4)]

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes_flat = axes.flatten()
colors = sns.color_palette('rocket_r', n_colors=5)
for j in range(len(Lqs)):
    ax = axes_flat[j]
    L, q = Lqs[j]
    Ks = []
    vals = []
    for i in range(len(C_calculation.TESTED)):
        L_, q_, K = C_calculation.TESTED[i]
        if L == L_ and q == q_:
            if j == 1:
                lbl = '$K=%s$' % K
            else:
                lbl=None
            vals = frac_recovered[i]
            ax.plot(Cs, vals, c=colors[K-1], label=lbl)
    if j == 1:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fancybox=True)
        
    ax.plot((2.62, 2.62), (0, 1), c='k', lw=1, ls='--')
    ax.set_xlim([0, 3])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("$C$")
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.set_ylabel("Fraction Recovered at $C \cdot S \log(q^L)$")
    ax.set_title("$L=%s, \,q=%s$" % (L, q))
    
plt.tight_layout()
plt.savefig("plots/figure_s3.png", dpi=300, bbox_inches='tight', facecolor='w', transparent=False)
plt.show()