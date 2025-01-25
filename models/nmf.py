import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF as nmf_sklearn
import os

def nmf(catalog_matrix, num_sign, tol = 1e-6, max_iter = 10e8):
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

    # Make sure catalog_matrix is a numpy array

    if not isinstance(catalog_matrix, np.ndarray):
        # Convert to numpy array
        catalog_matrix = np.array(catalog_matrix)
    
    m,n = catalog_matrix.shape
    losses = []
    # Initialize the signature and exposure matrices randomly
    S = np.random.rand(m, num_sign)
    E = np.random.rand(num_sign, n)

    # Compute the loss (Frobenius norm squared)
    loss = np.linalg.norm(catalog_matrix - S@E, ord = 'fro')
    losses.append(loss)
    
    diff = float('inf')
    n_iter = 0
    early_stop = False
    
    while(diff > tol and n_iter < max_iter and early_stop == False):
        n_iter += 1 


        E = E*(np.divide(S.T@catalog_matrix, S.T@S@E))
        S = S*(np.divide(catalog_matrix@(E.T), S@E@(E.T)))
        
        loss= np.linalg.norm(catalog_matrix - S@E, ord = 'fro')
        losses.append(loss)

        diff = abs(losses[-1] - losses[-2])
        

        # if n_iter%100 == 0:
        #     print(f"Iteration: {n_iter}, Loss: {losses[-1]}")

    return(S, E, losses)


def nmf_custom(V, k, max_iter=1000, tol=1e-6):
    """
    Perform Non-negative Matrix Factorization (NMF) using multiplicative updates.
    
    Parameters:
    - V: Input matrix (m x n)
    - k: Number of components
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence
    
    Returns:
    - W: Basis matrix (m x k)
    - H: Coefficient matrix (k x n)
    """

    if not isinstance(V, np.ndarray):
        # Convert to numpy array
        V = np.array(V)

    m, n = V.shape
    
    # Initialize W and H with random non-negative values
    W = np.abs(np.random.randn(m, k))
    H = np.abs(np.random.randn(k, n))

    losses = []
    
    for iteration in range(max_iter):
        # Update H
        numerator = W.T @ V
        denominator = W.T @ W @ H
        H *= numerator / (denominator + 1e-10)  # Add small value to avoid division by zero
        
        # Update W
        numerator = V @ H.T
        denominator = W @ H @ H.T
        W *= numerator / (denominator + 1e-10)
        
        # Check for convergence
        reconstruction_error = np.linalg.norm(V - W @ H, 'fro')
        
        losses.append(reconstruction_error)

        if reconstruction_error < tol:
            print(f"Converged after {iteration} iterations.")
            break
    
    return W, H, losses




def NMF_mult_tol(X, rank, tol = 1e-6, relative_tol = True, mse = False, max_iter = 10e8, F_0 = None, G_0 = None):
    """
    Performs Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    tol (float): Tolerance level for convergence (default is 1e-3).
    relative_tol (bool): If True, the tolerance is relative, otherwise absolute (default is True).
    mse (bool): If True, returns Mean Squared Error, otherwise Frobenius Norm (default is False).
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).
    F_0 (numpy.ndarray): If given initial basis matrix of shape (p, rank) (default is None).
    G_0 (numpy.ndarray): If given initial weight matrix of shape (rank, n) (default is None).

    
    Returns:
    F (numpy.ndarray): Basis matrix of shape (p, rank).
    G (numpy.ndarray): Weight matrix of shape (rank, n).
    loss (list): List of loss values at each iteration.
    F_0 (numpy.ndarray): Initial basis matrix.
    G_0 (numpy.ndarray): Initial weight matrix.
    n_iter (int): Number of iterations performed.
    """
    
    
    if not isinstance(X, np.ndarray):
        # Convert to numpy array
        X = np.array(X)

    
    p,n = X.shape


    if F_0 is None:
       F_0 = np.random.rand(p, rank)
    if G_0 is None:  
       G_0 = np.random.rand(rank,n)
        
    F = F_0
    G = G_0

    frob_norm = np.linalg.norm(X - F@G)
    denominator_mse = (n*p) if mse else 1
    loss_0 = frob_norm/denominator_mse
    loss = [loss_0]
    rel_diff = float('inf')
    n_iter = 0
    while(rel_diff>tol):
        n_iter += 1 
        
        G = G*(np.divide(F.T@X, F.T@F@G))
        F = F*(np.divide(X@(G.T), F@G@(G.T)))
        
        frob_norm =  np.linalg.norm(X - F@G)**2
        loss_val = frob_norm/(n*p) if mse else frob_norm
        loss.append(loss_val)
        denominator = loss[-2] if relative_tol else 1
        rel_diff = abs((loss[-1] - loss[-2]))/denominator 
        if n_iter >= max_iter:
            break

    return F, G, loss, F_0, G_0, n_iter 
    


if __name__ == '__main__':

    
    print("############ Testing the NMF function ############")

    # Test the NMF function on a dummy dataset
    catalog_matrix = np.random.rand(96, 532) # Ovary cancer dataset dummy matrix
    num_sign = 4
    S, E, losses_NMF = NMF(catalog_matrix, num_sign)
    print(S.shape, E.shape, len(losses_NMF))

    rec = np.round(S@E,2)

    plt.plot(losses_NMF)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration Ovary dummy dataset')

    S, E, losses_NMF_tol, _, _, _ = NMF_mult_tol(catalog_matrix, num_sign, relative_tol = True)
    print(S.shape, E.shape, len(losses_NMF))
    
    rec_mult_tol = np.round(S@E, 2)

    plt.plot(losses_NMF_tol)

    model = nmf_sklearn(n_components = num_sign, init = 'random', max_iter = int(10e8))

        
    S = model.fit_transform(catalog_matrix)
    E = model.components_
    loss_sklearn = model.reconstruction_err_


    plt.plot(loss_sklearn)


    S, E, loss_custom = nmf_custom(catalog_matrix, num_sign)

    plt.plot(loss_custom)

    plt.legend(['NMF', 'NMF_mult_tol', 'NMF_sklearn', 'NMF_custom'])

    plt.show()

    # Print the last loss values

    print("Loss value of the last iteration of NMF: ", losses_NMF[-1])
    print("Loss value of the last iteration of NMF_mult_tol: ", losses_NMF_tol[-1])
    print("Loss value of the last iteration of NMF_sklearn: ", loss_sklearn)
    print("Loss value of the last iteration of NMF_custom: ", loss_custom[-1])


    # Test the NMF function on the real dataset

    os.chdir("..")

    catalog_matrix = pd.read_csv("data/catalogues_Ovary_SBS.tsv", sep = '\t')

    # Extract 418 patients from the dataset

    catalog_matrix = catalog_matrix.iloc[:, 1:419].values

    num_sign = 4
    S, E, losses_NMF = NMF(catalog_matrix, num_sign)
    print(S.shape, E.shape, len(losses_NMF))

    rec = S@E

    plt.plot(losses_NMF)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration Ovary real dataset')

    S, E, losses_NMF_tol, _, _, _= NMF_mult_tol(catalog_matrix, num_sign, relative_tol = True)

    rec_mult_tol = S@E

    plt.plot(losses_NMF_tol)

    model = nmf_sklearn(n_components = num_sign, init = 'random', max_iter = int(10e8), solver = 'mu', tol = 1e-6)

        
    S = model.fit_transform(catalog_matrix)
    E = model.components_
    loss_sklearn = model.reconstruction_err_


    plt.plot(loss_sklearn)


    
    S, E, loss_custom = nmf_custom(catalog_matrix, num_sign)

    plt.plot(loss_custom)

    rec_custom = S@E

    plt.legend(['NMF', 'NMF_mult_tol', 'NMF_sklearn', 'NMF_custom'])

    plt.show()

    # Print the last loss values

    print("Loss value of the last iteration of NMF: ", losses_NMF[-1])
    print("Loss value of the last iteration of NMF_mult_tol: ", losses_NMF_tol[-1])
    print("Loss value of the last iteration of NMF_sklearn: ", loss_sklearn)
    print("Loss value of the last iteration of NMF_custom: ", loss_custom[-1])

    print("############ End of testing ############")
