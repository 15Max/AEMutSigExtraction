# Autoencoders for mutational signatures extraction

## Introduction 
- What are mutational signatures? (problem statement)
- Goal of the project
-Index of contents with motivation for each section



## Data
- Description of the data
- Data preprocessing (maybe) 
- data augmentation (maybe)

## [NMF](references/AENMF.pdf)
- What is NMF?
- Mathematical formulation
- c-nmf (particular case of NMF)

Non-negative matrix factorization (NMF) is a tool for unsupervised learning that factorizes a non-negative data matrix into a product of two non-negative matrices of lower dimension.

Suppose we have a non-negative matrix $V$ of size $M \times N$, where $ M$ is the number of features and $N$ is the number of samples. NMF decoposes $V$ into two non-negative matrices: 
- the basis matrix $ H\in \mathbb{R}^{M \times K}_+$ 
- the weight matrix  $ W\in \mathbb{R}^{K \times N}_+$ 

The shared
dimension, $K$ , of the factor matrices, is typically chosen to be much smaller than the dimensions of the
input matrix, making NMF a dimensionality reduction technique.


<img src="images/NMF.png" alt="NMF" width="400"/>


### [Relationship between c-NMF and Autoencoders](references/AENMF.pdf)

## Autoencoders


## [AE-NMF](references/AENMF.pdf)

## Relationship between AE-NMF and PCA

## [Denoising Sparse Autoencoder](references/Denoising.pdf)



## [MUSE-XAE](references/MUSE-XAE.pdf)




## Denoising Sparse Muse-XAE (maybe)



## Results


## Conclusion




## References



  
