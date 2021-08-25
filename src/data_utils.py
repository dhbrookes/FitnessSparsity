import pandas as pd
import numpy as np
import itertools
from Bio import PDB
from tqdm import tqdm
from sklearn.linear_model import Lasso

import structure_utils
import utils
import gnk_model


"""
Utility functions for loading and processing empirical fitness function data
"""

# global variables
MTAGBFP_POSITIONS = [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 227]
HIS3P_BIG_QS = [2, 2, 3, 2, 2, 3, 3, 4, 2, 4, 4]
HIS3P_POSITIONS = [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170]


def load_mtagbfp_data():
    """
    Loads mTagBFP2 blue flourescence fitness data from Poelwijk et. al. (2019). 
    Returns the data as (X, y), where X is a matrix of Walsh-Hadamard encodings 
    of sequences and y is an array of corresponding fitness values. Raw data
    is taken from Supplementary Data 3 of Poelwijk et. al. (2019) at
    https://www.nature.com/articles/s41467-019-12130-8#Sec20
    """
    df = pd.read_csv("../data/mtagbfp_raw_data.csv")
    y = np.array(df['brightness.1'][1:]).astype(float)
    bin_seqs_ = list(df['binary'][1:])
    bin_seqs = []
    for s_ in bin_seqs_:
        s_ = s_[1:-1]
        s = []
        for si in s_:
            if si == '0':
                s.append(0)
            else:
                s.append(1)
        bin_seqs.append(s)
    bin_seqs = np.array(bin_seqs)
    L = len(bin_seqs[0])
    X = utils.walsh_hadamard_from_seqs(bin_seqs)
    return X, y


def load_his3p_small_data():
    """
    Loads His3p fitness data from Pokusaeva et. al. (2019) for binary sequences. 
    Returns the data  as (X, y), where X is a matrix of Walsh-Hadamard encodings 
    of sequences and y is an array of corresponding fitness values. Raw data is 
    taken from https://github.com/Lcarey/HIS3InterspeciesEpistasis/tree/master/Data.
    """
    df = pd.read_csv("../data/his3p_raw_data.csv")
    extract = lambda x: "".join([x[i-1] for i in HIS3P_POSITIONS])
    mut_seqs = df['seq'].apply(extract) # extract subsequences
    seqs_split = mut_seqs.str.split(pat ="\s*", expand = True)
    seqs_split = seqs_split.iloc[:, 1:-1]
    
    # find the two most frequently occuring alphabet elements at each position
    best = []
    for i in range(1, 12): 
        pos = seqs_split[i]
        cts = pos.value_counts()
        best.append(list(cts.iloc[:2].index))
    bin_seqs_ = list(itertools.product((0, 1), repeat=11))
    best_combos = []
    for bs in bin_seqs_:
        seq = "".join([best[i][bs[i]] for i in range(len(bs))])
        best_combos.append(seq)
    num = 0
    bin_match =[]
    y_match = []
    
    # collect fitness values
    for i, s in enumerate(best_combos):
        matches = mut_seqs.loc[mut_seqs == s]
        if len(matches) == 0:
            continue
        fitness = np.log(np.mean(np.exp(df['log_fitness'].loc[matches.index])))  # calculate mean if there are multiple fitness values
        y_match.append(fitness)
        bin_match.append(bin_seqs_[i])
    
    X = utils.walsh_hadamard_from_seqs(np.array(bin_match))
    return X, np.array(y_match)



#######################
### His3p(big) data ###
#######################

"""
Loading and processing the His3p(big) data takes much more time and memory than 
the smaller datasets above, so the pipeline is split into multiple functions, whose
results are saved for further use. In order to load this data from scratch, one 
must run find_his3p_big_sequences(), followed by build_his3p_fourier(), and then
load_his3p_big_fourier().
"""

def find_his3p_big_sequences(save=False):
    """
    Searches through the His3p raw data to find sequences with fitness data that
    correspond to combinations of extant amino acids at each position (i.e. the 
    most frequently occuring amino acids at each position in the data). See page
    4 of Pokusaeva et. al. (2019) for more information about these extant sequences. 
    Returns a dictionary containing the found sequences, corresponding fitness values,
    and indices in the raw data. This takes some time to run, so there is an option
    to save the dictionary into the results fold by setting save=True. 
    """
    df = pd.read_csv("../data/his3p_raw_data.csv")
    qs = HIS3P_BIG_QS
    extract = lambda x: "".join([x[i-1] for i in HIS3P_POSITIONS])
    mut_seqs = df['seq'].apply(extract)
    seqs_split = mut_seqs.str.split(pat ="\s*", expand = True)
    seqs_split = seqs_split.iloc[:, 1:-1]
    best = []
    for i in range(1, 12):
        pos = seqs_split[i]
        cts = pos.value_counts()
        if i == 6:
            best.append(list(cts.iloc[[0,1,3]].index))
        else:
            best.append(list(cts.iloc[:qs[i-1]].index))
    sizes = [list(range(q)) for q in qs]
    int_seqs = list(itertools.product(*sizes))
    best_combos = []
    for bs in int_seqs:
        seq = []
        for i in range(len(bs)):
            seq.append(best[i][bs[i]])
        seq = "".join(seq)
        best_combos.append(seq)
    
    num = 0
    int_match =[]
    y_match = []
    idx_match = []
    print("Finding sequences in data...")
    for i, s in enumerate(tqdm(best_combos)):
        matches = mut_seqs.loc[mut_seqs == s]
        if len(matches) == 0:
            continue
        fitness = np.log(np.mean(np.exp(df['log_fitness'].loc[matches.index])))
        y_match.append(fitness)
        int_match.append(int_seqs[i])
        idx_match.append(i)
        num += 1

    out_dict = {"seq": int_match, "y": y_match, "idx": idx_match}
    if save:
        np.save("../results/his3p_big_data.npy", out_dict)
    return out_dict
    
    
def build_his3p_big_fourier(save=False):
    """
    Converts the His3p(big) sequences into Fourier encodings and returns the matrix
    If save=True, then the resulting matrix (which is ~20GB) will be saved into the results
    folder for fast loading. Will try to load the dict resulting from find_his3p_big_sequences,
    but will otherwise run that method.
    """
    qs = HIS3P_BIG_QS
    try:
        save_dict = np.load("../results/his3p_big_data.npy",allow_pickle=True).item()
    except FileNotFoundError:
        save_dict = find_his3p_big_sequences(save=save)
    int_seqs = save_dict['seq']
    M = np.prod(qs)
    N = len(int_seqs)
    phi = np.zeros((N, M))
    encodings = utils.get_encodings(qs)
    print("Calculating Fourier encoding for each sequence...")
    for i, seq in enumerate(tqdm(int_seqs)):
        phi[i] = utils.fourier_for_seq(seq, encodings) / np.sqrt(M)
    if save:
        np.save('../results/his3p_big_fourier.npy', phi)
    return phi


def load_his3p_big_data(save=False):
    """
    Loads His3p(big) fitness data from Pokusaeva, et. al. (2019). Returns 
    the data  as (X, y), where X is a matrix of Fourier encodings of sequences 
    and y is an array of corresponding fitness values. Will try to load dictionary
    from find_his3p_big_sequences and the Fourier matrix resulting from 
    build_his3p_big_fourier, but will otherwise run those method 
    (if save=True, then the dictionary and matrix will be saved for future use).
    """
    try:
        save_dict = np.load("../results/his3p_big_data.npy",allow_pickle=True).item()
    except FileNotFoundError:
        save_dict = find_his3p_big_sequences(save=save)

    y = np.array(save_dict['y'])
    try:
         X = np.load("../results/his3p_big_fourier.npy")
    except FileNotFoundError:
        X  = build_his3p_big_fourier(save=save)
    return X, y


def _get_contact_map(which_data):
    """
    Returns the contact map corresponding to either the TagBFP 
    (which_data='mtagbfp') or His3p data (which_data='his3p'). 
    """
    
    if which_data == 'mtagbfp':
        name = '3m24'
        pos = MTAGBFP_POSITIONS
    elif which_data == 'his3p':
        name = 'his3_itasser'
        pos = HIS3P_POSITIONS
    structure = PDB.PDBParser().get_structure(name, '../data/%s.pdb' % name)
    chains = structure.get_chains()
    chain1 = next(chains)
    contact_map, resid2idx  = structure_utils.calc_min_dist_contact_map(chain1) 
    pos_in_cm = [resid2idx[p] for p in pos]
    cm_sub = contact_map[pos_in_cm][:, pos_in_cm]
    return cm_sub
    

def _get_binarized_contact_map(which_data, threshold=4.5):
    """
    Returns the binarized contact map corresponding to either the mTagBFP 
    (which_data='mtagbfp') or His3p data (which_data='his3p').
    """
    cm_sub = _get_contact_map(which_data)
    bin_cm = structure_utils.binarize_contact_map(cm_sub, threshold=threshold)
    return bin_cm


def get_mtagbfp_contact_map():
    """
    Returns the contact map of the TagBFP structure, for the positions in
    the empirical fitness function of Poelwijk, et. al. (2019).
    """
    return _get_contact_map('mtagbfp')


def get_his3p_contact_map():
    """
    Returns the contact map of the His3p I-TASSER predicted structure, for the 
    positions in the empirical fitness function of Pokusaeva, et. al. (2019).
    """
    return _get_contact_map('his3p')


def get_mtagbfp_binarized_contact_map(threshold=4.5):
    """
    Returns the binarized contact map of the TagBFP structure, for the positions in
    the empirical fitness function of Poelwijk, et. al. (2019).
    """
    return _get_binarized_contact_map('mtagbfp', threshold=threshold)


def get_his3p_binarized_contact_map(threshold=4.5):
    """
    Returns the binarized contact map of the His3p I-TASSER predicted structure, 
    for the positions in the empirical fitness function of Pokusaeva, et. al. (2019).
    """
    return _get_binarized_contact_map('his3p', threshold=threshold)


def _calculate_wh_coefficients_complete(which_data):
    """
    Calculate the WH coefficients of the complete mTagBFP (which_data='mtagbfp'),
    His3p(small) (which_data='his3p_small') or His3p(big) (which_data='his3p_big') 
    empirical fitness functions.
    """
    alpha = 1e-12
    if which_data == 'mtagbfp':
        X, y = load_mtagbfp_data()
    elif which_data == 'his3p_small':
        X, y = load_his3p_small_data()
    elif which_data == 'his3p_big':
        X, y = load_his3p_big_data()
        alpha = 1e-10  # slightly higher because data is less complete than others
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    beta = model.coef_
    beta[0] = model.intercept_
    return beta

    
def calculate_mtagbfp_wh_coefficients():
    """
    Calculate the WH coefficients of the mTagBFP2 fitness functions.
    """
    return _calculate_wh_coefficients_complete('mtagbfp')


def calculate_his3p_small_wh_coefficients():
    """
    Calculate the Walsh-Hadamard coefficients of the His3p(small) fitness functions.
    """
    return _calculate_wh_coefficients_complete('his3p_small')


def calculate_his3p_big_fourier_coefficients():
    """
    Calculate the Fourier coefficients of the His3p(big) fitness functions.
    """
    return _calculate_wh_coefficients_complete('his3p_big')


def calculate_mtagbfp_gnk_wh_coefficient_vars(return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with Structural
    neighborhoods corresponding to the TagBFP structure. If return_neighborhoods is
    True then the Structural neighborhoods are also returned
    """
    L = 13
    q = 2
    bin_cm = get_mtagbfp_binarized_contact_map()
    V = structure_utils.contact_map_to_neighborhoods(bin_cm)
    gnk_beta_var = gnk_model.calc_beta_var(L, q, V)
    if return_neighborhoods:
        return gnk_beta_var, V
    else:
        return gnk_beta_var


def calculate_his3p_small_gnk_wh_coefficient_vars(return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with 
    Structural neighborhoods corresponding to the His3p structure.
    """
    L = 11
    q = 2
    bin_cm = get_his3p_binarized_contact_map()
    V = structure_utils.contact_map_to_neighborhoods(bin_cm)
    gnk_beta_var = gnk_model.calc_beta_var(L, q, V)
    if return_neighborhoods:
        return gnk_beta_var, V
    else:
        return gnk_beta_var