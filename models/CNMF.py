import numpy as np
from sklearn.cluster import KMeans

def cvx_update(XtX, G, W):
    """
    Computes a single convex NMF update.

    Parameters:
    XtX (numpy.ndarray): Precomputed X.T @ X matrix.
    G (numpy.ndarray): Weight matrix of shape (n, rank).
    W (numpy.ndarray): Encoding matrix of shape (n, rank).

    Returns:
    G (numpy.ndarray): Updated weight matrix.
    W (numpy.ndarray): Updated encoding matrix.
    """
    XtXW = XtX @ W
    XtXG = XtX @ G
    GWtXtXW = G @ W.T @ XtXW
    XtXWGtG = XtXW @ G.T @ G
    G *= np.sqrt(np.divide(XtXW, GWtXtXW, out=np.zeros_like(G), where=GWtXtXW != 0))
    W *= np.sqrt(np.divide(XtXG, XtXWGtG, out=np.zeros_like(W), where=XtXWGtG != 0))
    return G, W

def cvx_nmf_init(X, rank, init="random"):
    """
    Initializes the encoding and weight matrices for Convex NMF.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    init (str): Initialization method, either "random" or "kmeans" (default is "random").

    Returns:
    G (numpy.ndarray): Initial encoding matrix.
    W (numpy.ndarray): Initial weight matrix.
    """
    _, n = X.shape
    if init == "kmeans":
        kmeans = KMeans(n_clusters=rank).fit(X.T) # kmeans = KMeans(n_clusters=rank, n_init=10).fit(X.T)
        H = np.eye(rank)[kmeans.labels_]
        n_vec = H.sum(axis=0)
        G = H + 0.2 * np.ones((n, rank))
        W = G @ np.diag(1 / n_vec)
    else:  # Default to random initialization
        W = np.random.rand(n, rank)
        G = np.random.rand(n, rank)
    return G, W

def convex_nmf(X, rank, init="random", tol=1e-3, max_iter=10e8):
    """
    Performs Convex Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    init (str): Initialization method, either "random" or "kmeans" (default is "random").
    tol (float): Tolerance level for convergence (default is 1e-3).
    max_iter (int): Maximum number of iterations (default is 10e8).

    Returns:
    G (numpy.ndarray): Encoding matrix of shape (n, rank).
    W (numpy.ndarray): Weight matrix of shape (n, rank).
    loss (list): List of loss values at each iteration.
    n_iter (int): Number of iterations performed.
    """
    p, n = X.shape
    G, W = cvx_nmf_init(X, rank, init)
    XtX = X.T @ X

    loss = [np.linalg.norm(X - X @ W @ G.T)**2]
    rel_diff = float('inf')
    n_iter = 0

    while rel_diff > tol and n_iter < max_iter:
        n_iter += 1
        G, W = cvx_update(XtX, G, W)
        current_loss = np.linalg.norm(X - X @ W @ G.T)**2
        loss.append(current_loss)
        rel_diff = abs(loss[-1] - loss[-2]) / loss[-2]

    return G, W, loss, n_iter