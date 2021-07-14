import numpy as np
import itertools
from itertools import chain, combinations
from scipy.special import binom
from math import factorial
import utils


def get_neighborhood_powerset(V):
    """Returns the union of powersets of a set of neighborhoods"""
    Vs = [sorted(Vk) for Vk in V]
    powersets = [tuple(utils.powerset(Vs[i])) for i in range(len(Vs))]
    T = set().union(*powersets)
    return T


def calculate_sparsity(L, q, V):
    """Calculates sparsity given any neighborhoods V=[V1,V2,...,VL]"""
    T = get_neighborhood_powerset(V)
    sparsity = 0
    for U in T:
        sparsity += (q-1)**len(U)
    return sparsity


def calc_bn_sparsity(L, q, K):
    """Calculates sparsity of Block Beighborhood scheme"""
    sparsity = (L/K)*(q**K - 1) +1
    return sparsity


def calc_an_sparsity(L, q, K):
    """Calculates sparsity of Adjacent Neighborhood scheme"""
    sparsity = 0
    for r in range(K+1):
        nr = (L-K+1)*binom(K-1, r-1) + binom(K-1, r)
        sparsity += nr * (q-1)**r
    return sparsity


def calc_an_sparsity_periodic(L, q, K):
    """Calculates sparsity of Adjacent Neighborhood scheme"""
    sparsity = 1
    for r in range(1, K+1):
        nr = L*binom(K-1, r-1)
        sparsity += nr * (q-1)**r
    return sparsity


def _calc_set_prob(r, L, K):
    """Calculates p(r) for a set of size r, for use with 'calc_mean_rn_sparsity'"""
    if r == 0 or r == 1:
        return 1
    else:
        ar = (factorial(K-1) / factorial(L-1)) * (factorial(L-r) / factorial(K-r))
        br = ar * ((K-r) / (L-r))
        term1 = (1-ar)**r
        term2 = (1-br)**(L-r)
        return 1-term1*term2
  

def calc_mean_rn_sparsity(L, q, K):
    """Calculates expected sparsity of Random Neighborhood scheme"""
    sparsity = 0
    for r in range(K+1):
        pr = _calc_set_prob(r, L, K)
        sparsity += binom(L, r)*pr *(q-1)**r
    return sparsity


def calc_max_rn_sparsity(L, q, K):
    """Calculates an upper bound on the sparsity of the Random Neighborhood scheme"""
    bd = 1+L*(q-1)
    for r in range(2, K+1):
        bd += L * binom(K, r) * (q-1)**r
    return bd


def build_adj_neighborhoods(L, K):
    """Build neighborhoods according to the Adjacent Neighborhood scheme"""
    V = []
    M = (K-1)/2
    for i in range(L):
        bi = np.floor(np.max([1, (i+1)-M])).astype(int)
        ai = np.floor(np.min([L, (i+1)+M])).astype(int)
        Vi = list(range(bi, ai+1))
        V.append(Vi)
    return V


def build_adj_neighborhoods_periodic(L, K, symmetric=True):
    """Build Adjacent Neighborhoods with periodic boundary conditions"""
    V = []
    M = (K-1)/2
    for i in range(L):
        if symmetric:
            start = np.floor(i-M)
        else:
            start = i
        Vi = [((start + j) % L)+1 for j in range(K)]
        V.append(Vi)
    return V


def build_block_neighborhoods(L, K):
    """Build neighborhoods according to the Block Neighborhood scheme"""
    assert L % K == 0
    V = []
    block_size = int(L/K)
    for j in range(L):
        val = int(K*np.floor(j / K))
        Vj = list(range(val+1, val+K+1))
        V.append(Vj)
    return V


def sample_random_neighborhoods(L, K):
    """Sample neighborhoods according to the Random Neighborhood scheme"""
    V = []
    for i in range(L):
        indices = [j+1 for j in range(L) if j != i]
        Vi = list(np.random.choice(indices, size=K-1, replace=False))
        Vi.append(i+1)
        V.append(sorted(Vi))
    return V


def calc_beta_var(L, qs, V):
    """
    Calculates the variance of beta coefficients for a given sequence length, L, 
    list of alphabet sizes, and neighborhoods V. The returned coefficients are ordered
    by degree of epistatic interaction.
    """
    if type(qs) is int:
        qs = [qs]*L
    all_U = list(utils.powerset(range(1, L+1)))
    z = np.prod(qs)
    beta_var_U = []
    facs = []
    for j, Vj in enumerate(V):
        fac = 1
        for k in Vj:
            fac *= 1/qs[k-1]
        facs.append(fac)
    
    for i, U in enumerate(all_U):
        sz = np.prod([qs[k-1]-1 for k in U])
        bv = 0
        for j, Vj in enumerate(V):
            Uset = set(U)
            Vj_set = set(Vj)
            if Uset.issubset(Vj_set):
                bv += facs[j]
        bv *= z
        bv_expand = bv*np.ones(int(sz))
        beta_var_U.append(bv_expand)
    beta_var = np.concatenate(beta_var_U)
    return beta_var
