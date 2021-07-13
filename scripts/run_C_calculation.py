import sys
sys.path.append("../src")
import argparse

import C_calculation

"""
This script runs the tests required to determine a suitable value of C.

Run this script as

$ python run_C_Calculation.py -L <L> -q <q> -K <K> -n <num_N> -m <max_factor> -i <min_factor> -b <num_betas>

"""

parser = argparse.ArgumentParser()
parser.add_argument("-L", type=int, help="sequence length")
parser.add_argument("-q", type=int, help="alphabet size")
parser.add_argument("-K", type=int, help="GNK K parameter")
parser.add_argument("-n", "--num_N", type=int, default=50, help="number of training set sizes to run the test for")
parser.add_argument("-m", "--max_factor", type=float, default=5, help="factor that increases the maximum training set size if increased")
parser.add_argument("-i", "--min_factor", type=float, default=1, help="factor that decreases the minimum training set size if decreased")
parser.add_argument("-b", "--num_betas", type=float, default=25, help="number of GNK fitness function samples to test")
    
args = parser.parse_args()
L = args.L
q = args.q
K = args.K
num_betas = args.num_betas
num_N = args.num_N
max_factor = args.max_factor
min_factor = args.min_factor

C_calculation.run_LK(L, q, K, verbose=True, num_Ns=num_N, 
                     min_factor=min_factor, max_factor=max_factor,
                     num_betas=num_betas)