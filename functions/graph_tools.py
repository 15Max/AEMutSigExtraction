import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches


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
                color='black', marker='x', s=100, label="Medoids")

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
                color='black', marker='+', s=100, label="Medoids")

    # Add COSMIC signatures

    plt.title(title)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.grid()
    plt.show()



def base_plot_signature(array, axs, index, ylim=1):

    color = ((0.196,0.714,0.863),)*16 + ((0.102,0.098,0.098),)*16 + ((0.816,0.180,0.192),)*16 + \
            ((0.777,0.773,0.757),)*16 + ((0.604,0.777,0.408),)*16 + ((0.902,0.765,0.737),)*16
    
    color = list(color)

    width = max(array.shape)
    x = np.arange(width)
    if axs == None:
        f,axs = plt.subplots(1,figsize=(20,10))
    bars = axs.bar(x, array, edgecolor='black', color=color)

    plt.ylim(0, ylim)
    plt.yticks(fontsize=10)
    axs.set_xlim(-0.5, width) 
    axs.set_ylabel('Probability of mutation \n', fontsize=12)
    axs.set_xticks([])
    axs.set_xticks(x)  
    axs.set_xticklabels(index, rotation=90, fontsize=7)
    
    for tick_label, tick_color in zip(axs.get_xticklabels(), color):
        tick_label.set_color(tick_color)  




def plot_signature(signatures, name='DeNovo_Signatures'):
    """
    Visualize mutational signatures directly in a Jupyter Notebook.
    
    Parameters:
    - signatures: DataFrame containing mutational signatures (96 rows x n columns).
    - name: Base name for the plot titles (default: 'DeNovo_Signatures').
    """
    # Define the mutation types (96 trinucleotide contexts)
    index = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T','C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T',
             'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T',
             'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T', 'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T',
             'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T', 'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T',
             'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T',
             'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T',
             'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T',
             'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 
             'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T', 'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T',
             'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T', 'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T',
             'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T',
             'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']

    n_signatures = signatures.shape[1]  # Number of signatures
    
    for signature in range(n_signatures):
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.set_style('darkgrid')
        
        # Extract the signature values
        s = signatures.loc[index].values[:, signature]
        
        base_plot_signature(s, axs=ax, index=index, ylim=max(s) + 0.05)
        
        # Define legend patches
        l1 = mpatches.Patch(color=(0.196, 0.714, 0.863), label='C>A')
        l2 = mpatches.Patch(color=(0.102, 0.098, 0.098), label='C>G')
        l3 = mpatches.Patch(color=(0.816, 0.180, 0.192), label='C>T')
        l4 = mpatches.Patch(color=(0.777, 0.773, 0.757), label='T>A')
        l5 = mpatches.Patch(color=(0.604, 0.777, 0.408), label='T>C')
        l6 = mpatches.Patch(color=(0.902, 0.765, 0.737), label='T>G')
        
        # Add title and legend
        ax.text(0.01, 0.94, f'AENMF-SBS{chr(64 + signature + 1)}\n', transform=ax.transAxes, 
                fontsize=15, fontweight='bold', va='top', ha='left')
        ax.legend(handles=[l1, l2, l3, l4, l5, l6], loc='upper center', ncol=6, 
                  bbox_to_anchor=(0.5, 1.1), fontsize=18)
        
        # Display the plot in the notebook
        
        plt.show()