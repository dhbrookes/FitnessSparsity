import pandas as pd
import numpy as np
import itertools
import RNA
from Bio import PDB
from tqdm import tqdm
from sklearn.linear_model import Lasso

import structure_utils
import utils
import gnk_model


"""
Utility functions for loading and processing empirical fitness function data
"""

# hardcoded quantities
MTAGBFP_POSITIONS = [20, 45, 63, 127, 143, 158, 168, 172, 174, 197, 206, 207, 227]
HIS3P_BIG_QS = [2, 2, 3, 2, 2, 3, 3, 4, 2, 4, 4]
HIS3P_POSITIONS = [145, 147, 148, 151, 152, 154, 164, 165, 168, 169, 170]
RNA_POSITIONS = [2, 20, 21, 30, 43, 44, 52, 70]


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


#################
#### RNA Data ###
#################



def dna_to_rna(seq):
    """
    Converts DNA sequences to RNA sequences.
    """
    rs = []
    for s in seq:
        if s == 'T':
            rs.append('U')
        else:
            rs.append(s)
    return "".join(rs)


def insert(base_seq, positions, sub_seq):
    """
    Inserts a subsequence into a base sequence
    """
    new_seq = list(base_seq)
    for i, p in enumerate(positions):
        new_seq[p-1] = sub_seq[i]
    return "".join(new_seq)


def get_rna_base_seq():
    """
    Returns the sequence of RFAM: AANN01066007.1 
    """
    base_seq = "CTGAGCCGTTACCTGCAGCTGATGAGCTCCAAAAAGAGCGAAACCTGCTAGGTCCTGCAGTACTGGCTTAAGAGGCT"


def load_rna_data():
    """
    Constructs and returns the data corresponding to the quasi-empirical RNA fitness function
    of the Hammerhead ribozyme HH9. 
    """
    base_seq = get_rna_base_seq()
    base_seq = dna_to_rna(base_seq)
    positions = RNA_POSITIONS
    L = len(positions)
    q = 4
    
    # construct insertion sequences
    nucs = ["A", "U", "C", "G"]
    nucs_idx = {nucs[i]: i for i in range(len(nucs))}
    seqs_as_list = list(itertools.product(nucs, repeat=len(positions)))
    int_seqs = [[nucs_idx[si] for si in s] for s in seqs_as_list]
    seqs = ["".join(s) for s in seqs_as_list]
    
    y = []
    print("Calculating free energies...")
    for s in tqdm(seqs):
        full = insert(base_seq, positions, s)
        (ss, mfe) = RNA.fold(full)
        y.append(mfe)
    
    print("Constructing design matrix...")
    X = utils.fourier_from_seqs(int_seqs, [4]*L)
    return X, np.array(y)
    
    
def pairs_to_neighborhoods(positions, pairs):
    """
    Converts a list of pairs of interacting positions into a set of neighborhoods.
    """
    V = []
    for i, p in enumerate(positions):
        Vp = [i+1]
        for pair in pairs:
            if pair[0] == p:
                Vp.append(positions.index(pair[1]) + 1)
            elif pair[1] == p:
                Vp.append(positions.index(pair[0]) + 1)
        V.append(sorted(Vp))
    return V 


def find_pairs(ss):
    """
    Finds interacting pairs in a RNA secondary structure
    """
    pairs = []
    op = []
    N = len(ss)
    for i in range(N):
        if ss[i] == '(':
            op.append(i)
        elif ss[i] == ')':
            pair = (op.pop(), i)
            pairs.append(pair)
    return pairs


def sample_structures_and_find_pairs(base_seq, positions, samples=10000):
    """
    Samples secondary structures from the Boltzmann distribution 
    and finds pairs of positions that are paired in any of the
    sampled strutures.
    """
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(base_seq, md)
    (ss, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()

    important_pairs = set()
    for s in fc.pbacktrack(10000):
        pairs = find_pairs(s)
        for p in pairs:
            if p[0] in positions and p[1] in positions:
                if p[0] > p[1]:
                    print(p, s)
                important_pairs.add(tuple(p))
    return important_pairs


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

    save_dict = {"seq": int_match, "y": y_match, "idx": idx_match}
    np.save("../results/his3p_big_data.npy", save_dict)
    
    
def build_his3p_big_fourier():
    """
    Converts the His3p(big) sequences into Fourier encodings. The resulting
    matrix is quite large (~20GB) and is saved into the results folder.
    """
    qs = HIS3P_BIG_QS
    save_dict = np.load("../results/his3p_big_data.npy",allow_pickle=True).item()
    int_seqs = save_dict['seq']
    M = np.prod(qs)
    N = len(int_seqs)
    phi = np.zeros((N, M))
    encodings = utils.get_encodings(qs)
    print("Calculating Fourier encoding for each sequence...")
    for i, seq in enumerate(tqdm(int_seqs)):
        phi[i] = utils.fourier_for_seq(seq, encodings) / np.sqrt(M)
    np.save('../results/his3p_big_fourier.npy', phi)


def load_his3p_big_data():
    """
    Loads His3p(big) fitness data from Pokusaeva, et. al. (2019). Returns 
    the data  as (X, y), where X is a matrix of Fourier encodings of sequences 
    and y is an array of corresponding fitness values.
    """
    save_dict = np.load("../results/his3p_big_data.npy", allow_pickle=True).item()
    y = np.array(save_dict['y'])
    X = np.load("../results/his3p_big_fourier.npy")
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
    elif which_data == 'rna':
        X, y = load_rna_data()
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


def calculate_rna_fourier_coefficients(fast=True):
    """
    Calculate Fourier coefficients of the RNA fitness function.
    fast=True loads pre-calculated coefficients and fast=False
    calculates them from scratch.
    """
    if fast:
        return np.load("../results/rna_beta.npy")
    else:
        return _calculate_wh_coefficients_complete('rna')


def calculate_mtagbfp_gnk_wh_coefficient_vars(return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with Structural
    neighborhoods corresponding to the TagBFP structure. If return_neighborhoods is
    True then the Structural neighborhoods are also returned
    """
    L = 13
    q = 2
    mtag_bin_cm = get_mtagbfp_binarized_contact_map()
    mtag_V = structure_utils.contact_map_to_neighborhoods(mtag_bin_cm)
    mtag_gnk_beta_var = gnk_model.calc_beta_var(L, q, mtag_V)
    if return_neighborhoods:
        return mtag_gnk_beta_var, mtag_V
    else:
        return mtag_gnk_beta_var


def calculate_his3p_small_gnk_wh_coefficient_vars(return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with 
    Structural neighborhoods corresponding to the His3p structure.
    """
    L = 11
    q = 2
    his3p_bin_cm = get_his3p_binarized_contact_map()
    his3p_V = structure_utils.contact_map_to_neighborhoods(his3p_bin_cm)
    his3p_gnk_beta_var = gnk_model.calc_beta_var(L, q, his3p_V)
    if return_neighborhoods:
        return his3p_gnk_beta_var, his3p_V
    else:
        return his3p_gnk_beta_var