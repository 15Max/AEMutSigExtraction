### Project outline

The project can be separated in two parts. The first is an introductory one: we explain what is a mutational signature, what is the technique that is usually employed to extract them (NMF) and how we can build an autoencoder to mimic a similar process (AE-NMF) and comment on the results.

We're going to test them on either/both the dummy dataset and the PCAWG dataset. What we expect (and what we saw) is that the results are very similar, with a small advantage for NMF. _A case can be made for the study of the different ways one can implement AE-NMF._ 

The great results greenlights a further study on more complex autoencoder, that should introduce features that solve some issues or we think makes sense. Ideally we could go over all the elements that compose an autoencoder.

##### Architecture
How many layers? What activation functions? Dropout? How many neurons? Batch normalization? etc...

##### Training method
Data augmentation? What loss do we use? Introduce regularization? How many signatures?

My idea was something like this: highlight an issue $\to$ propose a solution

- There might be some non linearity $\to$ Introduce a more complex architecture
- For count data some other losses might be more appropriate $\to$ Change the loss (i.e. NPLL, Huber...)
- We have little input data $\to$ Data augmentation
- Now we might be overfitting $\to$ Sparsity + Denoising + Dropout

Something like that. Here we're missing the optimizer type, need to study better (but usually Adam is SOA so not needed ?). Num of layers, activation functions, num of neurons? One can think of optimizing this via grid search or whatever... Maybe for activation functions we can read something and motivate our choice. As for the "hard architecture" I think it should be small since data wise the input matrix is quite small (so maybe we stick to what they did in AE-NMF or MUSE). "Macro" training method should be MUSE like (allows us to find a "good" number of signatures without knowing it a priori).


##### Results comparison

What they did in AE-NMF is good, we should do that, so analyze both the reconstruction error and compare the found consensus signatures with the known ones in COSMIC.

##### CG idea

If we have a prior knowledge of the mutational signatures inside the data, and we fix this knowledge inside the architecture (by fixing the initial weights), then we can try and extract further signatures by beginning from this state. (Say we know there're 4 mut sign, we freeze the weights so that they're the same as the signature and exposure matrices obtained from NMF, then we increase the latent space to allow for X more signatures and setting the new weights to 0. Train and check what happens. Do we find new signatures? Does it stay the same?)

##### Structure

1) Architecture (num layers, num neurons, activations)
2) Weight init
3) Data aug
4) Denoising sparse dropout
5) Losses
6) Optimizers
7) Hyper param tuning

##### TO DO

1) Fix the way we compute the test error: for NMF we fix the signature and compute the best exposure. With AENMF? Investigate
2) Unify muse and denoising
3) Finish developing the correct training cycle