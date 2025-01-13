# Autoencoders for mutational signatures extraction

## Introduction 
Mutational signatures are distinct patterns of mutations that result from specific biological processes or external environmental factors.

When we refer to mutations, we mean changes in the DNA sequence of an alles, one of the two copies of a gene that we inherit from our parents.

When analyzing mutations in normal tissue, we compare them to the reference genome, which represents the most common sequence of the human genome. Mutations can be classified into two main categories:

- **Germline mutations**: Inherited from our parents and present in every cell of the body.
- **Somatic mutations**: Acquired during a person's lifetime and not inherited.
Since cancer is driven by the accumulation of mutations, identifying common patterns across different patients is crucial for understanding tumor development.

In mutational signature analysis, we specifically focus on somatic mutations, which arise during a patient’s lifetime rather than being inherited. These mutations can be caused by different factors, including environmental exposures, defective DNA repair mechanisms or endogenous (internal) processes.

Mutational processes can be broadly classified into two categories:
- **Endogenous mutational processes** (e.g. DNA replication errors, defective DNA repair mechanisms)
- **Exogenous mutational processes** (e.g. exposure to ultraviolet light, tobacco smoke)

Understanding mutational signatures is particularly useful for devising personalized patient treatment, as different mutations are linked to specific causes of tumorigenesis. 
### Mutational contexts
In this project we'll focus on the analysis of single base substitutions (SBS), which involve the mutation of a single nucleotide in the DNA sequence within a specific context. The context refers to the bases immediately before and after the mutated base. 

If two consecutive bases are mutated, we refer to them as double base substitutions (DBSs). Other types of mutations, such as insertions and deletions (indels), exist but will not be considered in this project.

Single base substitutions can be classified into six types, depending on the type of base substitution:
- C>A (cytosine to adenine)
- C>G (cytosine to guanine)
- C>T (cytosine to thymine)
- T>A (thymine to adenine)
- T>C (thymine to cytosine)
- T>G (thymine to guanine)

The context of a mutation is crucial, as the surrounding bases can influence the likelihood of a mutation occurring. 
This results in 96 possible combinations (6 types of base substitutions $\times$ 4 possible bases before $\times$ 4 possible bases after the mutated base).
Thus, an SBS mutation is classified based on its specific trinucleotide context.

If we define a probability vector $p_i$, where each element represents the probability of observing mutation type $i$ in a given signature. The
matrix $W$ contains these probabilities for each mutational signature, ensuring that the sum of the elements in each column is equal to 1.

The following image shows an example of a mutational signature associated with U-V light mutational processes.

<img src="images/MutSig.png" alt="NMF" width="700"/>

A catalog of known mutational sginatures can be found in the [COSMIC database](https://cancer.sanger.ac.uk/signatures), which is used as a reference for comparing and identifying new signatures.
## Project overview TODO:check this part
Signatures can be represented as a matrix factorization problem, where the data matrix, consisting of non-negative counts of mutations in each trinucleotide context, is factorized into two non-negative matrices: the signature matrix and the exposure matrix. 
Throught this project we want to extract these two lower rank matrices, with a special focus on autoencoders.

TODO: fix this part when we have decided on dataset/data augmetation/synthetic data

The data we used is....

Firstly, as a benchmark, we used NMF and convex-NMF to extract the signature and exposure  matrices, these are the most common methods used for this task. 
Then, we explored the use of autoencoders, starting with a shallow autoencoder with non-negativity constraints.
Since the activation function is identity, this autoencoder is equivalent to PCA.
We also imìnvestigated the use of a shallow denoising sparse autoencoder,with non-negativity constraints to ensure the extracted weigths were coherent with the problem at hand. This model should be more robust to noise and overfitting, and provide a more sparse representation of the data.

#TODO : check this part, fix it when decided what to do
Finally, we implemented a more complex autoencoder, MUSE-XAE, which incorporates bootstrapping, Poisson likelihood in the loss function, early stopping, and k-means to extract the best decoder weights. 
The results should provide insights into the performance of autoencoders compared to the classical NMF methods, and potentially identify non-linear relationships between the data that are not captured by NMF.
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
- TODO: Add a bit more mathematical notation, after we decide what to use, to avoid inconsistencies

Denoising sparse autoencoders are a combination of two techniques: denoising autoencoders and sparse autoencoders.
### Denoising autoencoders
Denoising autoencoders are trained to reconstruct the original input from a corrupted version of it. This is done by adding noise to the input data and training the model to recover the original data. The idea behind this technique is to make the autoencoder more robust to noise and improve its generalization capabilities.
In our case, we decided to add random Gaussian noise to the input count matrices before the training procedure.
### Sparse autoencoder
Sparse autoencoders are designed to learn a sparse representation of the input data. This means the model is encouraged to use only a small number of neurons in the hidden layer, leading to a more compact and efficient representation. This helps reduce overfitting and enhances generalization.
To enforce sparsity, we incorporated L1 regularization in the loss function, which penalizes large activations in the hidden layer.

TODO: add what we did in the project

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



  
