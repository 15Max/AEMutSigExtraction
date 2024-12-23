# Autoencoders for mutational signatures extraction

## Introduction (M)
- What are mutational signatures? (problem statement)
- Goal of the project
-Index of contents with motivation for each section

Mutational signatures are distinct patterns of mutations that result from specific biological processes or external environmental factors. 
When we refer to mutations we are talking about changes in the DNA sequence of a cell that can be caused by different factors, including environmental exposures, defective DNA repair mechanisms, endogenous (internal) processes or even inherited from our parents.
Some of these mutations can be linked to the development of cancer. 
This is why we are interested in identifyng common patterns of mutations that can help us understand the underlying factors that lead to cancer.

Mutational signatues are 

## Data (M)
- Description of the data
  - GEL / Qualcosa di diverso
  - Dati sintetici (?), sicuramente in fase di sviluppo
- Data preprocessing: data loading, augmentation (?), normalization

## [NMF](references/AENMF.pdf) (A)
- What is NMF?
- How is NMF used in the mutational signature context (extraction of Signature & Exposure matrix and their meaning)
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



## [What are AE and their relationship with c-NMF](references/AENMF.pdf) (A)

## [Relationship between AE-NMF and PCA](references/AENMF.pdf) (A)

### Brief recap on PCA 
### Mathematical relationship between PCA and AE-NMF

## [Denoising Sparse Autoencoder](references/Denoising.pdf) (M)

### Denoising technique
### Sparse technique

## [MUSE-XAE](references/MUSE-XAE.pdf) (N)

### Bootstrapping (data aug)
### Poissong Likelihood in loss + early stopping 
### while the third term represents the logarithm of the minimum volume constraint (??) (vediamo se metterlo) $$\beta$$
### K-means to extract best decoder weights -> Consensus matrices
### Signature assignment
### De novo extraction scenario on synt data (1)
### De novo extraction on real data (GEL/WGS)
### Analysis of how data aug improves results on various metrics
### (Confronto generale con i nostri AEs ed NMF)
### T-sne (depending on what we use for dataset)

## Denoising Sparse Muse-XAE (maybe) (N)


## Results (N + M + A)


## Conclusion (N + M + A)




## References (N + M + A)



  
