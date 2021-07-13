import numpy as np
from Bio import PDB


"""
Various utility functions for working with PDB structures.
"""


def binarize_contact_map(contact_map, threshold=8.0):
    """Returns binary version of contact map."""
    return np.less(contact_map, threshold)


def calc_min_dist(res1, res2):
    """Returns the minimum distance between any two atoms in res1 and res2."""
    md = -1
    for atom1 in res1:
        for atom2 in res2:
            dist = atom1-atom2  # biopython defines this operator 
            if md == -1 or dist < md:
                md = dist
    
    return md


def calc_min_dist_contact_map(chain1, chain2=None):
    """
    Calculates the minimum distance between any two atoms in every pair of residues 
    in chain1 and (optionally) chain2. Returns the array of distances along with a 
    dictionary that maps residue IDs to indices in the array.
    """
    resid_to_idx1 = {}
    if chain2 is None:
        chain2 = chain1
        one_chain = True
    else:
        one_chain = False
        resid_to_idx2 = {}
    l1 = len(chain1)
    l2 = len(chain2)
    contact_map = np.ones((l1, l2)) * -1
    for i, res1 in enumerate(chain1):
        resid_to_idx1[int(res1.id[1])] = i
        for j, res2 in enumerate(chain2):
            if one_chain:
                if j < i:
                    continue
            contact_map[i, j] = calc_min_dist(res1, res2)
            if one_chain:
                contact_map[j, i] = contact_map[i, j]
                resid_to_idx1[int(res2.id[1])] = j
            else:
                resid_to_idx2[int(res2.id[1])] = j
            
    # Set missing distances larger than maximum distance in structure.
    missing_value = np.amax(contact_map) + 1.0
    contact_map[np.less(contact_map, 0)] = missing_value
    if one_chain:
        return contact_map, resid_to_idx1
    else:
        return contact_map, resid_to_idx1, resid_to_idx2


def contact_map_to_neighborhoods(contact_map):
    """Convert a binarized contact map to a set of neighborhoods"""
    L = contact_map.shape[0]
    V = []
    for i in range(L):
        Vi = [i+1]
        for j in range(L):
            if i == j:
                continue
            elif contact_map[i, j]:
                Vi.append(j+1)
        V.append(sorted(Vi))
    return V