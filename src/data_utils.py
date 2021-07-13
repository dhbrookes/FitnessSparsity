import pandas as pd
import numpy as np
import utils
import itertools
from Bio import PDB
import structure_utils
from sklearn.linear_model import Lasso, Ridge


"""
Utility functions for loading and processing empirical fitness function data
"""

# hardcoded quantities
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

def find_his3p_big_sequences():
    """
    Searches through the His3p raw data to find sequences with fitness data that
    correspond to combinations of extant amino acids at each position (i.e. the 
    most frequently occuring amino acids at each position in the data). See page
    4 of Pokusaeva et. al. (2019) for more information about these extant sequences. 
    Saves a dictionary containing the found sequences, corresponding fitness values,
    and indices in the raw data.
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
    for i, s in enumerate(best_combos):
        matches = mut_seqs.loc[mut_seqs == s]
        if len(matches) == 0:
            continue
        fitness = np.log(np.mean(np.exp(df['log_fitness'].loc[matches.index])))
        y_match.append(fitness)
        int_match.append(int_seqs[i])
        idx_match.append(i)
        num += 1

    save_dict = {"seq": int_match, "y": y_match, "idx": idx_match}
    np.save("../results/his3p_big_extant_match.npy", save_dict)
    
    
def build_his3p_big_fourier():
    """
    Converts the His3p(big) sequences into Fourier encodings. The resulting
    matrix is quite large (~20GB) and is saved into the results folder.
    """
    qs = [2, 2, 3, 2, 2, 3, 3, 4, 2, 4, 4]
    save_dict = np.load("../results/his3p_big_extant_match.npy",allow_pickle=True).item()
    int_seqs = save_dict['seq']
    M = np.prod(qs)
    N = len(int_seqs)
    phi = np.zeros((N, M))
    encodings = utils.get_encodings(qs)
    for i, seq in enumerate(int_seqs):
        phi[i] = utils.fourier_for_seq(seq, encodings) / np.sqrt(M)
    np.save('../results/his3p_big_fourier.npy', phi)


def load_his3p_big_data():
    """
    Loads His3p(big) fitness data from Pokusaeva, et. al. (2019). Returns 
    the data  as (X, y), where X is a matrix of Fourier encodings of sequences 
    and y is an array of corresponding fitness values.
    """
    save_dict = np.load("../results/his3p_big_extant_match.npy", allow_pickle=True).item()
    y = np.array(save_dict['y'])
    X = np.load("../results/his3p_big_fourier.npy")
    return X, y


def _get_binarized_contact_map(which_data):
    """
    Get the binarized contact map corresponding to either the mTagBFP (which='mtagbfp') 
    or His3p data (which='his3p').
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
    bin_cm = structure_utils.binarize_contact_map(contact_map, threshold=4.5)
    pos_in_cm = [resid2idx[p] for p in pos]
    bin_cm_sub = bin_cm[pos_in_cm][:, pos_in_cm]
    return bin_cm_sub


def get_mtagbfp_binarized_contact_map():
    """
    Returns the binarized contact map of the mTagBFP structure, for the positions in
    the empirical fitness function of Poelwijk, et. al. (2019).
    """
    return _get_binarized_contact_map('mtagbfp')


def get_his3p_binarized_contact_map():
    """
    Returns the binarized contact map of the His3p I-TASSER predicted structure, 
    for the positions in the empirical fitness function of Pokusaeva, et. al. (2019).
    """
    return _get_binarized_contact_map('his3p')


def _calculate_wh_coefficients_complete(which_data):
    """
    Calculate the WH coefficients of the complete mTagBFP (which_data='mtagbfp')
    or His3p small (which_data='his3p') empirical fitness functions.
    """
    if which_data == 'mtagbfp':
        X, y = load_mtagbfp_data()
    elif which_data == 'his3p':
        X, y = load_his3p_small_data()
    model = Lasso(alpha=1e-12)
    model.fit(X, y)
    beta = model.coef_
    beta[0] = model.intercept_
    return beta

    
def calculate_mtagbfp_wh_coefficients():
    """
    Calculate the WH coefficients of the mTagBFP fitness functions.
    """
    return _calculate_wh_coefficients_complete('mtagbfp')


def calculate_his3p_small_wh_coefficients():
    """
    Calculate the Walsh-Hadamard coefficients of the His3p(small) fitness functions
    """
    return _calculate_wh_coefficients_complete('his3p')