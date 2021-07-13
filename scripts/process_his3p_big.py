import sys
sys.path.append("../src")
import data_utils

"""
This script runs the initial processing that are required
before the His3p(big) dataset can be loaded. Run as:

$ python process_his3p_big.py

"""

data_utils.find_his3p_big_sequences()
data_utils.build_his3p_big_fourier()

