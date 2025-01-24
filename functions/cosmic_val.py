import numpy as np 
import pandas as pd


from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def compute_match(Signatures : pd.DataFrame, Signatures_true : pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cosine similarity between the extracted signatures and the true signatures and return a dataframe with the similarity values.

    Parameters:
    Signatures (pd.DataFrame): Extracted signatures of shape 96 x k
    Signatures_true (pd.DataFrame): True signatures of shape 96 x k

    Returns:
    match_df (pd.DataFrame): Dataframe with columns 'Extracted', 'True', and 'Similarity' showing the similarity values between the extracted and true signatures.
    """


    if not isinstance(Signatures, pd.DataFrame):
        Signatures = pd.DataFrame(Signatures)
    
    if not isinstance(Signatures_true, pd.DataFrame):
        Signatures_true = pd.DataFrame(Signatures_true)

    
    # Since the matrices are 96 x k, we need to transpose them to k x 96
    cost = cosine_similarity(Signatures.T, Signatures_true.T)

    row_ind, col_ind = linear_sum_assignment(cost)

    Signatures_sorted = Signatures.iloc[:, col_ind]
    Signatures_true_sorted = Signatures_true.iloc[:, row_ind]

    simils = np.diag(cosine_similarity(Signatures_sorted.T, Signatures_true_sorted.T))

    match_df = pd.DataFrame({
        'Extracted' : Signatures_sorted.columns,
        'True' : Signatures_true_sorted.columns,
        'Similarity' : simils
    })

    return match_df