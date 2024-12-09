import scipy.spatial as sp
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls



def signature_matching(estimated_signatures : np.ndarray, real_signatures: np.ndarray) -> tuple:
    '''
    This function calculates the similarity between the estimated and real signatures and uses it to 
    match the sets of estimated and real signatures. This is achieved via the Hungarian algorithm 
    which tries to maximize the similarity between the signatures.
    
    Parameters:
    estimated_signatures : np.ndarray, the estimated signature matrix. Each row corresponds to a signature.
    real_signatures : np.ndarray, the real signature matrix. Each row corresponds to a signature.

    Returns:
    similarity_matrix.T)[:,col_idx].T : np.ndarray, the columns of the transposed similarity matrix corresponding to the optimal matching.
    col_idx : np.ndarray, the indices of the optimal matching.
    '''

    # similarity_matrix[i,j] = similarity between estimated signature i and real signature j
    similarity_matrix = 1 - sp.distance.cdist(estimated_signatures, real_signatures, 'cosine') 

    # Check for invalid values in the similarity matrix
    if np.any(np.isinf(similarity_matrix) | np.isnan(similarity_matrix)):
        raise ValueError("infinity or NaN values detected in the similarity matrix. Please check your input data.\n\n Similarity matrix:\n", similarity_matrix)
    
    # Hungarian algorithm: find the  matching that maximizes the sum over all pairs of similarities
    # Good application of the algorithms since basis matrices are M X K with M being fixed 96 (SBS) and K being usually small
    _, col_idx  = linear_sum_assignment(-similarity_matrix.T)

    return ((similarity_matrix.T)[:,col_idx].T , col_idx)


def reconstruction_error(data : pd.DataFrame, signature_matrix : np.ndarray, mean : bool = True, max_iterations : int = 30) -> float:
    '''
    This function calculates the reconstruction error of the data matrix given the signature matrix. 
    The reconstruction error is defined as the Frobenius norm of the difference between the data matrix 
    and the product of the signature matrix and the coefficient matrix. If mean is set to True, the 
    reconstruction error is normalized by the number of elements in the data matrix.
    
    Parameters:
    data : np.ndarray, the data matrix. Each row corresponds to a sample.
    signature : np.ndarray, the signature matrix. Each row corresponds to a signature.
    mean : bool, whether to normalize the reconstruction error by the number of elements in the data matrix.

    Returns:
    reconstruction_error : float, the reconstruction error.
    '''

    # Solve Ax = b with non-negative least squares
    exposures = data.apply(lambda x: nnls(A = signature_matrix, b = x, maxiter = max_iterations)[0], axis = 0)

    # Calculate the reconstruction error
    reconstruction_error = np.linalg.norm(data.to_numpy() - np.dot(signature_matrix, exposures))

    if mean:
        reconstruction_error /= np.prod(data.shape)

    return reconstruction_error


if __name__ == "__main__":
    
    # Test the reconstruction error function

    np.random.seed(0)

    # Generate random data matrix

    data = np.random.rand(100, 100) 

    data = pd.DataFrame(data)
    
    # Generate random signature matrix

    signature_matrix = np.random.rand(10, 100)

    signature_matrix = signature_matrix.T

    # Calculate the reconstruction error

    error = reconstruction_error(data, signature_matrix)

    print("Reconstruction error:", error)

    # Test the signature matching function

    # Generate random estimated signature matrix

    estimated_signatures = np.random.rand(10, 100)

    # Generate random real signature matrix

    real_signatures = np.random.rand(10, 100)

    # Match the estimated and real signatures

    matrix, col_idx = signature_matching(estimated_signatures, real_signatures)

    print("Optimal matching:", col_idx)
    print("Similarity matrix:", matrix)