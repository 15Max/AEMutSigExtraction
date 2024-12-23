# Autoencoders for mutational signatures extraction

## Introduction 
- What are mutational signatures? (problem statement)
- Goal of the project
-Index of contents with motivation for each section

Mutational signatures are distinct patterns of mutations that result from specific biological processes or exposures. 
When we refer to mutations we are talking about changes in the DNA sequence of a cell that can be caused by different factors, including environmental exposures, defective DNA repair mechanisms, endogenous (internal) processes or even inherited from our parents.
Some of these mutations can be linked to the development of cancer. 
This is why we are interested in identifyng common patterns of mutations that can help us understand the underlying factors that lead to cancer.

Mutational signatues are 

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



  
