import sys
sys.path.append("../src")
import data_utils

"""
This script runs the initial processing for the His3p(big)
data and saves the results. This isn't strictly required
but is recommended because the methods take some time to run.

$ python process_his3p_big.py

"""

data_utils.find_his3p_big_sequences(save=True)
data_utils.build_his3p_big_fourier(save=True)

