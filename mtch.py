from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

def match_batch(desc_ids, conc_ids, X_descs, X_concs):
    """
    Matches descriptions to conclusions using cosine similarity and Hungarian algorithm.

    Parameters:
        desc_ids (list[int]): List of indices of descriptions
        conc_ids (list[int]): List of indices of conclusions
        X_descs (ndarray): Precomputed TF-IDF matrix for all descriptions
        X_concs (ndarray): Precomputed TF-IDF matrix for all conclusions

    Returns:
        list of tuples: (desc_id, conc_id) pairs matched optimally
    """
    # Get vectors for this batch
    desc_vecs = X_descs[desc_ids]
    conc_vecs = X_concs[conc_ids]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(desc_vecs, conc_vecs)

    # Convert to cost matrix for minimization
    cost_matrix = -sim_matrix

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map back to original IDs
    matched_pairs = [(desc_ids[i], conc_ids[j]) for i, j in zip(row_ind, col_ind)]
    return matched_pairs
