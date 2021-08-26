import numpy as np
import itertools
import RNA
from tqdm import tqdm
from sklearn.linear_model import Lasso

import utils
import gnk_model

"""
Utility functions for loading and processing the quasi-empirical RNA fitness function.
"""

RNA_POSITIONS = [2, 20, 21, 30, 43, 44, 52, 70]


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
    return base_seq


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
    
    print("Constructing Fourier matrix...")
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


def calculate_rna_fourier_coefficients(save=False):
    """
    Calculates Fourier coefficients of the RNA fitness function. This will try to load them 
    from the results folder, but will otherwise calculate from scratch. If save=True,
    then coefficients will be saved to the results folder.
    """
    try:
        return np.load("../results/rna_beta.npy")
    except FileNotFoundError:
        alpha = 1e-12
        X, y = load_rna_data()
        print("Fitting Fourier coefficients (this may take some time)...")
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        beta = model.coef_
        beta[0] = model.intercept_
        return beta


def calculate_rna_gnk_wh_coefficient_vars(pairs_from_scratch=False, return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with 
    Structural neighborhoods corresponding to RNA secondary structure. If pairs_from_scratch
    is True, then structures are sampled to find paired positions, otherwise pre-calculated
    pairs are used.
    """
    L = 8
    q = 4

    if pairs_from_scratch:
        important_pairs = sample_structures_and_find_pairs(data_utils.get_rna_base_seq(), positions, samples=10000) # uncomment to calculate from scratch
    else:
        important_pairs = {(21, 52), (20, 44), (20, 52), (20, 43)}  # pre-calculated

    # add adjacent pairs
    important_pairs = important_pairs.union({(20, 21), (43, 44)})
    V = data_utils.pairs_to_neighborhoods(positions, important_pairs)

    gnk_beta_var = gnk_model.calc_beta_var(L, q, V)
    if return_neighborhoods:
        return gnk_beta_var, V
    else:
        return gnk_beta_var
