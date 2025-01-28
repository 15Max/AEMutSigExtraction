import numpy as np 
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def compute_match(Signatures : pd.DataFrame, Signatures_true : pd.DataFrame, index : int) -> pd.DataFrame:
    """
    Compute the cosine similarity between the extracted signatures and the true signatures and return a dataframe with the similarity values.

    Parameters:
    Signatures (pd.DataFrame): Extracted signatures of shape 96 x k
    Signatures_true (pd.DataFrame): True signatures

    Returns:
    match_df (pd.DataFrame): Dataframe with columns 'Extracted', 'True', and 'Similarity' showing the similarity values between the extracted and true signatures.
    """


    if not isinstance(Signatures, pd.DataFrame):
        Signatures = pd.DataFrame(Signatures)
    
    if not isinstance(Signatures_true, pd.DataFrame):
        Signatures_true = pd.DataFrame(Signatures_true)

    
    cost = cosine_similarity(Signatures.T, Signatures_true.T)


    row_ind, col_ind = linear_sum_assignment(1 - cost)


    Signatures_sorted = Signatures.iloc[:, row_ind]
    Signatures_true_sorted = Signatures_true.iloc[:, col_ind]

    simils = np.diag(cosine_similarity(Signatures_sorted.T, Signatures_true_sorted.T))

    match_df = pd.DataFrame({
        f'Extracted_{index}': Signatures_sorted.columns,
        f'True_{index}' : Signatures_true_sorted.columns,
        f'Similarity_{index}' : simils
    })

    mean_similarity = np.mean(match_df[f'Similarity_{index}'])

    return match_df, mean_similarity



def compute_all_matches(all_signatures : np.ndarray, cosmic : pd.DataFrame, n_runs :int ) -> pd.DataFrame:
    """
    Compute the cosine similarity between the extracted signatures and the true signatures and return a dataframe with the similarity values.

    Parameters:
    all_signatures (np.ndarray): Extracted signatures of shape 96 x k
    cosmic (pd.DataFrame): True signatures

    Returns:
    match_df (pd.DataFrame): Dataframe with columns 'Extracted', 'True', and 'Similarity' showing the similarity values between the extracted and true signatures.
    """
    all_matches = pd.DataFrame()
    for i in range(0, all_signatures.shape[1], n_runs):
    
        signature_block = all_signatures[:, i:i+4]

        match, _ = compute_match(signature_block, cosmic, index = i//n_runs)

        all_matches = pd.concat([all_matches, match.iloc[:,1:]],  axis=1)

    return all_matches
