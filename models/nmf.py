import numpy as np


def NMF(catalog_matrix, num_sign, tol = 1e-3, max_iter = 10e8):
    """
    Performs Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    catalog_matrix (numpy.ndarray): Catalog matrix of shape m x n where m is the number of SBS mutation types (96) and n is the number of patiens
    num_sign (int): Number of signatures to be extracted
    tol (float): Tolerance level for convergence
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).

    
    Returns:
    S (numpy.ndarray): Basis matrix of shape (m, num_sign).
    E (numpy.ndarray): Weight matrix of shape (num_sign, n).
    losses (list): List of loss values at each iteration.
    """
    
    m,n = catalog_matrix.shape
    losses = []
    # Initialize the signature and exposure matrices randomly
    S = np.random.rand(m, num_sign)
    E = np.random.rand(num_sign, n)

    # Compute the loss (Frobenius norm squared)
    loss = np.linalg.norm(catalog_matrix - S@E, ord = 'fro')**2
    losses.append(loss)
    
    diff = float('inf')
    n_iter = 0
    early_stop = False
    
    while(diff > tol and n_iter < max_iter and early_stop == False):
        n_iter += 1 


        E = E*(np.divide(S.T@catalog_matrix, S.T@S@E))
        S = S*(np.divide(catalog_matrix@(E.T), S@E@(E.T)))
        
        loss= np.linalg.norm(catalog_matrix - S@E)**2
        losses.append(loss)

        diff = abs(losses[-1] - losses[-2])
        

        # Add a print statement to see the progress
        if n_iter%100 == 0:
            print(f"Iteration: {n_iter}, Loss: {losses[-1]}")

    return(S, E, losses)
    


if __name__ == '__main__':

    
    print("############ Testing the NMF function ############")

    # Test the NMF function
    catalog_matrix = np.random.rand(96, 532) # Ovary cancer dataset dummy matrix
    num_sign = 4
    S, E, losses = NMF(catalog_matrix, num_sign)
    print(S.shape, E.shape, len(losses))




    print("############ End of testing ############")

    # Expected output: (96, 5) (5, 100) 1