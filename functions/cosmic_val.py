import numpy as np 
import pandas as pd


from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def compute_match(Signatures : pd.DataFrame, Signatures_true : pd.DataFrame) -> pd.DataFrame:
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

    print(Signatures_sorted)
    print(Signatures_true_sorted)
    print("Shape of signatures_sorted: ", Signatures_sorted.shape)
    print("Shape of signatures_true_sorted: ",Signatures_true_sorted.shape)

    simils = np.diag(cosine_similarity(Signatures_sorted.T, Signatures_true_sorted.T))

    match_df = pd.DataFrame({
        'Extracted' : Signatures_sorted.columns,
        'True' : Signatures_true_sorted.columns,
        'Similarity' : simils
    })

    return match_df