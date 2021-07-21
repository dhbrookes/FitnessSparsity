import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import binom
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_neighborhoods(ax, V, L, positions, label_rotation=0, s=120):
    """
    Plots a set of neighborhoods, as done in Figure 1 and Figure 4A.
    """
    colormap = sns.color_palette('crest', as_cmap=True)
    for j in range(L):
        y_ = [L-j-0.5 for _ in range(len(V[j]))]
        x_ = [v-0.5 for v in V[j]]
        ax.scatter(x_, y_, facecolor=colormap(0.9), marker='s',s=s)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L+1))
    ax.xaxis.tick_top()
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator([i+0.5 for i in range(L)]))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.FixedLocator([i+0.5 for i in range(L)]))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(positions))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(['$V^{[%s]}$' % s for s in positions[::-1]]))
    ax.tick_params(axis='x', which='minor', rotation=label_rotation)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    return ax



def plot_beta_comparison(ax, L, num_coeffs,
                         emp_beta_sq_mags, 
                         gnk_beta_var, 
                         use_order_labels=True,
                         max_order=5,
                         order_lbl_offset=25, 
                         order_lbl_height=0.45, 
                         arrow1_xy=(7, 0.5), 
                         arrow1_text_xy=(52, 0.63),
                         arrow2_xy=(44, 0.45), 
                         arrow2_text_xy=(83, 0.54), 
                         ticks=None, 
                         yticks=(-0.5, -0.25, 0, 0.25, 0.5), 
                         yticklabels=('0.5', '0.25', '0', '0.25', '0.5')
                        ):
    """
    Makes plots comparing empirical Fourier coefficient magnitudes with 
    expected GNK magnitudes. These plots are shown in Figure 4B and S1A. 
    Default values are for the top row of Figure 4B.
    """
    colors = sns.color_palette('Set1', n_colors=2)
    plot_gnk_vals = np.sqrt(gnk_beta_var)
    plot_emp_vals = np.sqrt(emp_beta_sq_mags)
    mv = np.max([np.max(plot_gnk_vals), np.max(plot_emp_vals)])

    ax.bar(range(len(plot_emp_vals[:num_coeffs])), plot_emp_vals[:num_coeffs], 
           width=3, facecolor=colors[1])
    ax.bar(range(len(plot_gnk_vals[:num_coeffs])), -plot_gnk_vals[:num_coeffs], 
           width=3, facecolor=colors[0])

    ax.plot((-10, num_coeffs),(0, 0), c='k')
    if ticks is None:
        ticks = [np.sum([binom(L, j) for j in range(i)]) for i in range(L+1)]
        ticks = [t for t in ticks if t <= num_coeffs]
    ordlbls = ["1st", "2nd", "3rd"] + ["%ith" for i in range(3, L+1)]
    for i, tick in enumerate(ticks):
        ax.plot((tick, tick), (-1, 1), c='k', ls='--', lw=0.5, alpha=0.5)
        if i > 2 and i <= max_order:
            if use_order_labels:
                ax.text(tick+order_lbl_offset, order_lbl_height, "$r=%i$" %i)
                
    if use_order_labels:
        ax.annotate("",
                    xy=arrow1_xy, xycoords='data',
                    xytext=arrow1_text_xy, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                                    connectionstyle="arc3,rad=0.15"),
                    )

        ax.text(arrow1_text_xy[0], arrow1_text_xy[1], "$r=1$")

        ax.annotate("",
                    xy=arrow2_xy, xycoords='data',
                    xytext=arrow2_text_xy, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                                    connectionstyle="arc3,rad=0.15"),
                    )

        ax.text(arrow2_text_xy[0], arrow2_text_xy[1], "$r=2$")

    ax.tick_params(axis='y', which='major', direction='out')
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim([-mv, mv])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim([-10, num_coeffs+5])
    ax.set_ylabel("Magnitude of Fourier coefficient", labelpad=2, fontsize=12)
    ax.grid(False)
    return ax


def plot_percent_variance_explained(ax, emp_beta_pv, 
                                    gnk_pv_mean, 
                                    gnk_pv_std,
                                    gnk_sparsity,
                                    xlim=75,
                                    xticks=(0, 25, 50, 75)):
    """
    Plots the percent variance explained by empirical coefficients 
    and GNK coefficients. These plots are shown in Figure 4C and S1B.
    Default values are for the top row of Figure 4C.
    """
    colors = sns.color_palette('Set1', n_colors=2)
    ax.plot(range(len(emp_beta_pv)), emp_beta_pv, c=colors[1], 
            label='Empirical', lw=1.5, zorder=1)
    ax.plot(range(len(gnk_pv_mean)), gnk_pv_mean, 
            c=colors[0], label='GNK mean', lw=1.5)
    ax.fill_between(range(len(gnk_pv_mean)), gnk_pv_mean-gnk_pv_std, 
                    (gnk_pv_mean+gnk_pv_std), color=colors[0], alpha=0.4, 
                    edgecolor='none', label='GNK std. dev.', zorder=10)
    
    pv_at_gnk = emp_beta_pv[gnk_sparsity]
    ax.plot((gnk_sparsity, gnk_sparsity), (0, pv_at_gnk), ls='--', c='k', lw=0.75, zorder=0)
    ax.plot((0, gnk_sparsity), (pv_at_gnk, pv_at_gnk), ls='--', c='k', lw=0.75)
    ax.scatter([gnk_sparsity], [pv_at_gnk], edgecolor='k', facecolor='none', 
               zorder=11, s=15, linewidth=0.75)

    ax.set_xlim([0, xlim])
    ax.set_ylim([0, 100.5])
    ax.set_ylabel("Percent variance explained", labelpad=2, fontsize=12)
    ax.set_xlabel("Number of largest coefficients", fontsize=12)
    ax.set_xticks(xticks)

    leg = ax.legend(fontsize=10, labelspacing=0.15, bbox_to_anchor=(0.05,0), loc='lower left')
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_boxstyle('Square', pad=0.05)
    return ax


def plot_lasso_example_inset(axins, example_results, 
                             xlim=(0, 1.75), 
                             ylim=(0, 1.65), 
                             xticks=(0, 0.5, 1, 1.5), 
                             yticks=(0, 0.5, 1, 1.5),
                             text_xy=(0.9, 0.3)
                            ):
    """
    Plot the inset to the LASSO results shown in Figures 4D and S1C.
    Default values are for the top row of Figure 4D
    """
    colors = sns.color_palette('Set1')
    # Process example results
    y = example_results[0]
    pred = example_results[1]
    r2_example = pearsonr(pred, y)[0]**2
    m, b = np.polyfit(y ,pred, 1)
    minmin = np.min([np.min(pred), np.min(y)])
    maxmax = np.max([np.max(pred), np.max(y)])
    
    axins.scatter(y, pred, s=1, c='k')
    xs = np.linspace(minmin, maxmax+0.1, 1000)
    axins.plot(xs, m*xs+b, c=colors[0], ls='--')
    
    axins.set_xlabel("Empirical fitness", fontsize=8, labelpad=1)
    axins.set_ylabel("Predicted fitness", fontsize=8, labelpad=1)
    axins.set_xticks(xticks)
    axins.set_yticks(yticks)
    axins.set_ylim(ylim)
    axins.set_xlim(xlim)
    axins.text(text_xy[0], text_xy[1], "$R^2=%.2f$" % r2_example, fontsize=8, color=colors[0])
    axins.tick_params(axis='y', which='major', right=False, labelright=False, labelsize=8)
    axins.tick_params(axis='x', which='major', top=False, labeltop=False, labelsize=8)
    axins.grid(False)
    axins.spines['right'].set_visible(True)
    axins.spines['top'].set_visible(True)
    return axins


def plot_lasso_results(ax, lasso_results_dict, 
                       pred_num_samples, 
                       xlim=2000,
                       print_name='His3p(small)'
                      ):
    """
    Makes the plots showing LASSO results. This is used for Figures 4D 
    (bottom row) and Figure S1C. The top row of Figure 4D requires splitting 
    the y-axis, which is done in the plotting script.
    """
    # Process LASSO results
    ns = lasso_results_dict['n']
    pearson = lasso_results_dict['pearson']
    mean_r = np.mean(pearson**2, axis=1)
    std_r = np.std(pearson**2, axis=1)
    idx = np.where(ns==pred_num_samples)[0][0]
    r2_at_pred = mean_r[idx]
    print("LASSO R^2 at %s GNK predicted number of samples: %.3f" % (print_name, r2_at_pred))
    
    colors = sns.color_palette('Set1')
    
    ax.errorbar(ns, mean_r, yerr=std_r, lw=0.5, marker='o', 
                markersize=0, c='k', zorder=12, fmt='none', capsize=1, 
                markeredgewidth=0.5)
    ax.plot(ns, mean_r, c=colors[1], lw=1, marker='o', markersize=3, zorder=10)
    ax.plot((pred_num_samples, pred_num_samples), (0, r2_at_pred), 
            ls='--', c='k', lw=0.75, zorder=0)
    ax.plot((0, pred_num_samples), (r2_at_pred, r2_at_pred), 
            ls='--', c='k', lw=0.75)
    
    ax.scatter([pred_num_samples], [r2_at_pred],
               edgecolor='k', facecolor=colors[0], 
               zorder=11, s=15, linewidth=0.75)

    ax.tick_params(axis='y', which='major', right=False, labelright=False)
    ax.tick_params(axis='x', which='major', top=False, labeltop=False)
    
    ax.set_ylabel("Prediction $R^2$", labelpad=2, fontsize=12)
    ax.set_xlabel("Number of training samples", fontsize=12)
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, 1])
    
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4","0.6", "0.8", "1.0"])
    return ax
    
    
def plot_lasso_example_inset(axins, example_results, 
                             xlim=(0, 1.75), 
                             ylim=(0, 1.65), 
                             xticks=(0, 0.5, 1, 1.5), 
                             yticks=(0, 0.5, 1, 1.5),
                             text_xy=(0.9, 0.3)
                            ):
    """
    Plots the inset to the LASSO results shown in Figures 4D and S1C.
    """
    colors = sns.color_palette('Set1')
    # Process example results
    y = example_results[0]
    pred = example_results[1]
    r2_example = pearsonr(pred, y)[0]**2
    m, b = np.polyfit(y ,pred, 1)
    minmin = np.min([np.min(pred), np.min(y)])
    maxmax = np.max([np.max(pred), np.max(y)])
    
    axins.scatter(y, pred, s=1, c='k')
    xs = np.linspace(minmin, maxmax+0.1, 1000)
    axins.plot(xs, m*xs+b, c=colors[0], ls='--')
    
    axins.set_xlabel("Empirical fitness", fontsize=8, labelpad=1)
    axins.set_ylabel("Predicted fitness", fontsize=8, labelpad=1)
    axins.set_xticks(xticks)
    axins.set_yticks(yticks)
    axins.set_ylim(ylim)
    axins.set_xlim(xlim)
    axins.text(text_xy[0], text_xy[1], "$R^2=%.2f$" % r2_example, fontsize=8, color=colors[0])
    axins.tick_params(axis='y', which='major', right=False, labelright=False, labelsize=8)
    axins.tick_params(axis='x', which='major', top=False, labeltop=False, labelsize=8)
    axins.grid(False)
    axins.spines['right'].set_visible(True)
    axins.spines['top'].set_visible(True)
    return axins