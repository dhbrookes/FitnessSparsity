import sys
sys.path.append("../src")
import data_utils
import lasso_experiments

"""
Runs the LASSO experiments on empirical fitness functions, whose results
are shown in Figure 4D.

Run this script as 
    $ python run_lasso_exp.py <data_name>
where <data_name> is either 'mtagbfp', 'his3p_small', or 'his3p_big'
"""

nm = sys.argv[1]
if nm == 'mtagbfp':
    gnk_val = 575  # predicted number of samples from GNK model 
    ns = [25, 50, 75] + list(np.arange(100, 2050, 50))+[gnk_val]
    X, y = data_utils.load_mtagbfp_data()
    savefile = "../results/mtagbfp_lasso_results.npy"
    example_savefile = "../results/mtagbfp_lasso_example.npy"
    
elif nm == 'his3p_small':
    gnk_val = 660
    ns = [25, 50, 75] + list(np.arange(100, 2050, 50))+[gnk_val]
    alphas = [5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
    X, y = data_utils.load_his3p_small_data()
    savefile = "../results/his3p_small_lasso_results.npy"
    example_savefile = "../results/his3p_small_lasso_example.npy"
        
elif nm == 'his3p_big':
    gnk_val = 6537
    ns = [100, 200, 300, 400] + list(np.arange(500, 20500, 500)) + [gnk_val]
    alphas = [5e-7, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4]
    X, y = data_utils.load_his3p_big_data()
    savefile = "../results/his3p_big_lasso_results.npy"
    example_savefile = "../results/his3p_big_lasso_example.npy"
    

alpha = lasso_experiments.determine_alpha(X, y, gnk_val)
print("Optimal alpha at GNK predicted number of samples: %s" % alpha)
lasso_experiments.run_lasso_exp(X, y, alpha, ns, savefile, 
                                save_example_at=gnk_val, 
                                example_savefile=example_savefile)