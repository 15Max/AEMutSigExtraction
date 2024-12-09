import numpy as np

def NMF_mult(X, rank, tol=1e-3, max_iter=1e8, F_0=None, G_0=None):
    """
    Performs Non-negative Matrix Factorization using multiplicative updates.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    tol (float): Tolerance level for convergence (default is 1e-3).
    max_iter (int): Maximum number of iterations (default is 1e8).
    F_0 (numpy.ndarray): Initial basis matrix of shape (p, rank) (optional).
    G_0 (numpy.ndarray): Initial weight matrix of shape (rank, n) (optional).

    Returns:
    F (numpy.ndarray): Basis matrix of shape (p, rank).
    G (numpy.ndarray): Weight matrix of shape (rank, n).
    loss (list): Loss values at each iteration.
    n_iter (int): Number of iterations performed.
    """
    p, n = X.shape
    F = F_0 if F_0 is not None else np.random.rand(p, rank)
    G = G_0 if G_0 is not None else np.random.rand(rank, n)

    loss = [np.linalg.norm(X - F @ G)**2]
    rel_diff = float('inf')
    n_iter = 0

    while rel_diff > tol and n_iter < max_iter:
        n_iter += 1
        
        # Update rules
        G *= np.divide(F.T @ X, F.T @ F @ G, out=np.zeros_like(G), where=F.T @ F @ G != 0)
        F *= np.divide(X @ G.T, F @ G @ G.T, out=np.zeros_like(F), where=F @ G @ G.T != 0)
        
        # Compute loss
        current_loss = np.linalg.norm(X - F @ G)**2
        loss.append(current_loss)
        
        # Compute relative difference
        rel_diff = abs(loss[-1] - loss[-2]) / loss[-2]

    return F, G, loss, n_iter