import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function to plot Kmedoids clusters reduced with PCA
def plot_clusters(reduced_signatures, labels, medoid_indices, LATENT_DIM, title = "AENMF Signatures"):

    plt.figure(figsize=(10, 6))
    # Plot all points colored by cluster
    for cluster in range(LATENT_DIM):
        cluster_points = reduced_signatures[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    # Highlight medoids
    medoid_points = reduced_signatures[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1], 
                color='black', marker='+', s=100, label="Medoids")

    plt.title(title)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend()
    plt.grid()
    plt.show()



def plot_clusters_with_cosmic(reduced_signatures, labels, medoid_indices, matched , cosmic, LATENT_DIM, title = "AENMF Signatures"):

    plt.figure(figsize=(10, 6))
    # Plot all points colored by cluster
    for cluster in range(LATENT_DIM):
        cluster_points = reduced_signatures[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    # Highlight medoids
    medoid_points = reduced_signatures[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1], 
                color='black', marker='x', s=100, label="Medoids")

    # Add COSMIC signatures

    plt.title(title)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.grid()
    plt.show()