import numpy as np
from itertools import chain, combinations


def divisors(num):
    """Returns all divisors of a given integer"""
    divs = []
    for x in range (1, num):
        if (num % x) == 0:
            divs.append(x)
    return divs


def powerset(iterable):
    """Returns the powerset of a given set"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def complete_graph_evs(q):
    """
    Returns a set of eigenvectors of complete graph of size q as column vectors of a matrix
    """
    x = np.ones(q)
    y = np.eye(q)[0]
    v = x - np.linalg.norm(x, ord=2) * y
    w = v / np.linalg.norm(v, ord=2)
    w = w.reshape(q, 1)
    P = np.eye(q) - 2*np.dot(w, w.T)
    return P


def fourier_basis_recursive(L, q):
    """
    Recursively constructs the Fourier basis corresponding to the H(L, q) hamming graph 
    (i.e. graphs of sequences of length L with alphabet size q). Hamming graphs are the 
    Cartesian product H(L+1, q) = H(L, q) x Kq, and the eigevectors of a cartesian product 
    (A x B) are the Kronecker product of the eigenvectors of A and B.
    """
    Pq = complete_graph_evs(q)
    phi = np.copy(Pq)
    for i in range(L-1):
        phi = np.kron(Pq, phi)
    return phi


def get_encodings(qs):
    """
    Returns a length L list of arrays containing the encoding vectors corresponding 
    to each alphabet element at each position in sequence, given the alphabet size 
    at each position.
    """
    encodings = []
    Pqs = []
    L = len(qs)
    for i in range(L):
        qi = qs[i]
        Pq = complete_graph_evs(qi) * np.sqrt(qi)
        Pqs.append(Pq)
        enc_i = Pq[:, 1:]
        encodings.append(enc_i)
    return encodings


def fourier_for_seq(int_seq, encodings):
    """
    Returns an M x 1 array containing the Fourier encoding of a sequence, 
    given the integer representation of the sequence and the encodings returned 
    by get_encodings, where M = prod(qs) and qs is the alphabet size at each position.
    """
    L = len(int_seq)
    all_U = list(powerset(range(L)))
    all_U = [list(U) for U in all_U]
    epi_encs = []
    enc_1 = encodings[0][int_seq[0]]
    for U in all_U:
        if len(U) > 0 and 0 == U[0]:
            U_enc = enc_1
            U.pop(0)
        else:
            U_enc = np.array([1])
        epi_encs.append(U_enc)
    
    for l in range(1, L):
        enc_l = encodings[l][int_seq[l]]
        for k, U in enumerate(all_U):
            U_enc = epi_encs[k]
            if len(U) > 0 and l==U[0]:
                U_enc = np.kron(U_enc,enc_l)
                U.pop(0)
            epi_encs[k] = U_enc
    all_enc = np.concatenate(epi_encs)
    return all_enc


def fourier_from_seqs(int_seqs, qs):
    """
    Returns an N x M array containing the Fourier encodings of a given list of 
    N sequences with alphabet sizes qs.
    """
    M = np.prod(qs)
    N = len(int_seqs)
    encodings = get_encodings(qs)
    phi = np.zeros((N, M))
    for i, seq in enumerate(int_seqs):
        phi[i] = fourier_for_seq(seq, encodings) / np.sqrt(M)
    return phi


def convert_01_bin_seqs(bin_seqs):
    """Converts an array of {0, 1} binary sequences to {-1, 1} sequences"""
    bin_seqs[bin_seqs == 0] = -1
    return bin_seqs


def walsh_hadamard_from_seqs(bin_seqs):
    """
    Returns an N x 2^L array containing the Walsh-Hadamard encodings of
    a given list of N binary ({0,1}) sequences. This will return the 
    same array as fourier_from_seqs(bin_seqs, [2]*L), but is much
    faster.
    """
    bin_seqs_ = convert_01_bin_seqs(bin_seqs)
    L = len(bin_seqs_[0])
    all_U = list(powerset(range(0, L)))
    M = 2**L
    N = len(bin_seqs)
    X = np.zeros((N, len(all_U)))
    for i, U in enumerate(all_U):
        if len(U) == 0:
            X[:, i] = 1
        else:
            X[:, i] = np.prod(bin_seqs_[:, U], axis=1)
    X = X / np.sqrt(M)
    return X


def calc_frac_var_explained(beta_var, samples=1000, up_to=None):
    """
    Numerically calculates the fraction variance explained by the largest elements in
    samples of a normally distributed random vector with variances given by beta_var. 
    Let S be the number of largest elements to consider. For each value of S between 0
    and up_to (default up_to is len(beta_var)), this method returns the mean and std.
    dev. of the fraction variance explained by the largest S elements in samples
    of the random vector.
    """
    beta_var_nz = beta_var[np.nonzero(beta_var)]
    M = len(beta_var_nz)
    if up_to is None:
        up_to = M
    fv = np.zeros(up_to)
    fv_std = np.zeros(up_to)
    samples = np.random.randn(samples, M) * np.sqrt(beta_var_nz).reshape(1, M)
    sorted_beta_sq = -np.sort(-samples**2, axis=1)
    mags = np.sum(sorted_beta_sq, axis=1)
    for i in range(up_to):
        if i > M:
            fv[i] = 1
            fv_std[i] = 1
        if i == 0:
            continue
        all_fv = np.sum(sorted_beta_sq[:, :i], axis=1) / mags
        fv[i] = np.mean(all_fv)
        fv_std[i] = np.std(all_fv)
    return fv, fv_std