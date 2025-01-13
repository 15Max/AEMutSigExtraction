import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment  # Faster LAP solver
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


class HybridLoss(nn.Module):
    '''
    Class for the Hybrid Loss function. The Hybrid Loss function is a combination of the Negative Poisson Log-Likelihood
    Loss and the Minimum Volume Regularizer. The Negative Poisson Log-Likelihood Loss is the reconstruction loss of the
    model (input is count data), while the Minimum Volume Regularizer is a regularizer that enforces the exposure matrix
    to be sparse (i.e. the latent representation to be sparse therefore "more interpretable").

    Parameters:
    beta (float): The weight of the regularizer
    '''
    def __init__(self, beta=0.001):
        super(HybridLoss, self).__init__()
        self.beta = beta
        self.eps = 1e-6 # Small value for numerical stability (avoid log(0))

    def forward(self, x, x_hat, decoder_weights, reg_enc_loss=0):
        """
        Method to compute the Hybrid Loss.

        Parameters:
        x (torch.Tensor): The input data
        x_hat (torch.Tensor): The reconstructed data
        decoder_weights (torch.Tensor): The weights of the decoder (signature matrix)
        reg_enc_loss (float): The regularization loss of the encoder (significant only during refitting)

        Returns:
        total_loss (float): The total loss of the model
        """
        # Negative Poisson Log-Likelihood Loss (NPLL)
        npll_loss = torch.sum(x_hat - x * torch.log(x_hat + self.eps))  # 🔹 Add stability

        # Apply Minimum Volume Regularization (as constraint)
        gram_matrix = torch.mm(decoder_weights, decoder_weights.T)
        identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        sign, log_det_value = torch.slogdet(gram_matrix + identity)  # 🔹 Use numerically stable determinant

        total_loss = npll_loss + (self.beta * log_det_value) + reg_enc_loss

        return total_loss



class Encoder(nn.Module):
    '''
    Class of the Encoder model. The Encoder model is a simple feedforward neural network that takes the input data
    and encodes it into a latent representation.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    l_1 (int): The number of neurons in the first layer of the encoder
    k (int): The number of latent features
    refit (bool): Whether to use the refitting mechanism (for signature extraction)
    refit_penalty (float): The penalty for the refitting mechanism (for signature extraction)
    '''
    def __init__(self, input_dim, l_1, k, refit=False, refit_penalty=1e-3):
        super(Encoder, self).__init__()
        
        self.refit = refit
        self.refit_penalty = refit_penalty
        self.activation = F.relu if refit else F.softplus

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, l_1),      # fc1
            nn.BatchNorm1d(l_1),            # bn1
            nn.Softplus(),                  # activation    
            
            nn.Linear(l_1, l_1 // 2),       # fc2
            nn.BatchNorm1d(l_1 // 2),       # bn2
            nn.Softplus(),                  # activation
            
            nn.Linear(l_1 // 2, l_1 // 4),  # fc3
            nn.BatchNorm1d(l_1 // 4),       # bn3
            nn.Softplus()                   # activation
        )

        # Adding the last layer (latent) separately to access its weights easily
        if refit:
            self.last_layer = nn.Linear(l_1 // 4, k)  # Latent layer
            self.activation_fn = nn.ReLU()
        else:
            self.last_layer = nn.Linear(l_1 // 4, k)
            self.activation_fn = nn.Softplus()

    def forward(self, x):
        '''
        Method to forward pass the input data through the model.

        Parameters:
        x (torch.Tensor): The input data to pass through the model

        Returns:
        x (torch.Tensor): The output of the model
        reg_loss (float): The regularization loss (significant only during refitting)
        '''
        x = self.encoder(x)  # Pass through encoder layers
        x = self.last_layer(x)  # Last layer (Linear)
        x = self.activation_fn(x)  # Activation

        reg_loss = 0

        if self.refit:

            # Compute L1 regularization on the activation, note that in the paper it is said that the regularization
            # happens on both the output of the encoder and the weights of the last layer of the encoder, but as of now
            # (commit 094f902) the regularization is only applied to the output of the encoder.

            reg_loss = self.refit_penalty * torch.norm(x, p=1)
            reg_loss += self.refit_penalty * torch.norm(self.last_layer.weight, p=1)  # Add the l1 penalty also on the weights

            return x, reg_loss

        return x, reg_loss


class Decoder(nn.Module):
    '''
    Class for the Decoder model. The Decoder model is a simple linear layer that takes the latent features and
    reconstructs the data.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    z (int): The number of latent features
    constraint (str): The constraint to apply on the decoder weights, can be 'pg' or 'abs'
    '''
    def __init__(self, input_dim : int,  z : int, constraint = 'pg'):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(z, input_dim, bias=False)
        self.constraint = constraint
        self.z = z

    def forward(self, x):
        '''
        Method to forward pass the input data through the model.

        Parameters:
        x (torch.Tensor): The input data to pass through the model

        Returns:
        x (torch.Tensor): The output of the model
        self.fc1.weight.data (torch.Tensor): The weight matrix of the model, which is the signature matrix
        '''
        x = self.fc1(x)  # Linear activation


        x = F.softplus(x)
        
        if(self.constraint == 'pg'):
            # Enforce non-negativity constraint on weights via clamping
            self.fc1.weight.data = nn.Parameter(torch.clamp(self.fc1.weight, min=0))
        elif(self.constraint == 'abs'):
            # Enforce non-negativity constraint on weights via absolute value
            self.fc1.weight.data = nn.Parameter(torch.abs(self.fc1.weight))   
        else:
            raise ValueError("Invalid constraint type. Choose 'pg' or 'abs'.")

        # print the size of the weight matrix

        # print("WEIGHT MATRIX SIZE: ", self.fc1.weight.data.size())

        return x, self.fc1.weight.data

class HybridAutoencoder(nn.Module):
    '''
    Hybrid Autoencoder model that combines the Encoder and Decoder models. The model is trained using the HybridLoss
    which is a combination of the Negative Poisson Log-Likelihood Loss and the Minimum Volume Regularizer.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    l_1 (int): The number of neurons in the first layer of the encoder
    z (int): The number of latent features
    refit (bool): Whether to use the refitting mechanism
    refit_penalty (float): The penalty for the refitting mechanism
    constraint (str): The constraint to apply on the decoder weights, can be 'pg' or 'abs'

    '''

    # In the paper it is assumed that the regularizer term is always the Minimum Volume Regularizer so we won't use the other regularizers
    # Also, it is assumed for the matrix to be passed as n x 96, so we will transpose the matrix before passing it to the model

    def __init__(self, input_dim : int = 96, l_1 : int = 128, z : int = 17, refit : bool = False, refit_penalty : float = 1e-3, constraint : str = 'pg'):
        super(HybridAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.l_1 = l_1
        self.z = z
        self.refit = refit
        self.refit_penalty = refit_penalty
        self.constraint = constraint

        self.encoder = Encoder(input_dim, l_1, z, refit, refit_penalty) 
        self.decoder = Decoder(input_dim, z, constraint)                

    def forward(self, x):
        '''
        Method to forward pass the input data through the model.

        Parameters:
        x (torch.Tensor): The input data to pass through the model

        Returns:
        reconstruction (torch.Tensor): The reconstructed data
        exposures (torch.Tensor): The exposure matrix
        signature_matrix (torch.Tensor): The signature matrix
        total_loss (float): The total loss of the model (Note, this is to be added to the loss function and is used only during refitting)
        '''
        exposures, reg_loss_enc = self.encoder(x)
        reconstruction, signature_matrix = self.decoder(exposures)

        total_loss = 0
        if reg_loss_enc is not None:
            total_loss += reg_loss_enc

        return reconstruction, exposures, signature_matrix, total_loss


    def assign_decoder_weights(self, weights):
        '''
        A method to assign the decoder weights (signature matrix) to the model.
        '''
        self.decoder.fc1.weight.data = weights

    
    def return_decoder_weights(self):
        '''
        A method to return the decoder weights (signature matrix) from the model.
        '''
        
        return self.decoder.fc1.weight.data
    
    def return_encoder_model(self):
        '''
        A method to return the encoder model from the HybridAutoencoder model.
        '''
        
        return self.encoder


def MUSE_optimal_model():
    pass 


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Generate n random positve floats matrices of dimension m x k

    n = 1   # Number of matrices
    m = 96  # SBS96
    k = 4   # Number of patients
    z = 2   # Number of signatures

    # Each Signature matrix is compposed of the m mutations and k patients

    X_og = np.random.poisson(lam=10, size=(m, k))  # Poisson-distributed integer values

    X_augmented = []
    for i in range(n):
        X = np.random.poisson(lam=10, size=(m, k))
        X = pd.DataFrame(X)
        X_augmented.append(X)

    # Let X_augmented be a pandas DataFrame

    X_augmented = pd.concat(X_augmented, axis=1)


    print("X_aug shape: ", X_augmented.shape)
    print("X_og shape: ", X_og.shape)

    AE = HybridAutoencoder(
        input_dim = 96,
        l_1 = 128,
        z = z,
        refit = False,
        constraint = 'abs'
    )


    X_og = pd.DataFrame(X_og).T
    X_augmented = X_augmented.T


    error, S, E, train_losses, val_losses = train_model_(
        model = AE,
        X_aug_multi_scaled = X_augmented,
        X_scaled = X_og,
        signatures = 8,
        epochs = 100,
        batch_size = 32,
        save_to = 'test',
        iteration = 10,
        patience = 30
    )

    print(error)
    print("Shape of S: ", S.shape)
    print("Shape of E: ", E.shape)

    print("S: ", S)
    print("E: ", E)

    print("Original matrix: ", X_og)

    print("Reconstructed matrix: ", np.dot(S, E))

    import matplotlib.pyplot as plt

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

