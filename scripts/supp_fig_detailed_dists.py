import sys
sys.path.append("../src")
import numpy as np
import data_utils
import plot_utils
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

plt.style.use(['seaborn-deep', '../paper.mplstyle'])

mtag_dists = data_utils.get_mtagbfp_contact_map()
his3p_dists = data_utils.get_his3p_contact_map()


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

plot_utils.plot_detailed_dists(axes[0], mtag_dists, data_utils.MTAGBFP_POSITIONS)
plot_utils.plot_detailed_dists(axes[1], his3p_dists, data_utils.HIS3P_POSITIONS)

plt.tight_layout()
plt.savefig("plots/supp_fig_detailed_dists.png", dpi=500)