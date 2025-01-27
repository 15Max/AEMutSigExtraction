import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import pandas as pd
import numpy as np

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    
class dsae(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization
    Additional constraints can be added to the model:
    - Projected Gradient (pg) 
    - Absolute Values (abs)
    """
    def __init__(self, input_dim, latent_dim, constraint = 'pg', xavier = False):
        '''
        Constructor for the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

        Parameters:
        - input_dim: Dimension of the input data
        - latent_dim: Dimension of the latent space
        - constraint: Constraint type ('pg' for projected gradient, 'abs' for absolute values)
        - xavier: If True, the weights are initialized using Xavier initialization
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint 
        self.xavier = xavier


        if self.constraint == 'pg':
            self.activation = nn.ReLU()
        elif self.constraint == 'abs':
            self.activation = Abs()
        elif self.constraint == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError('Constraint not recognized. Choose between pg, abs or id')


        ''' Encoder and Decoder layers '''
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            self.activation
        )

        ''' Initialize weights using Xavier initialization '''
        if self.xavier:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    # Ensure non-negative weights
                    module.weight.data.clamp_(min=0)



    def forward(self, x):
        '''
        Forward pass of the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

        Parameters:
        - x: Input data

        Returns:
        - x: Output data
        '''
        x = self.encoder(x)
        x = self.decoder(x)
    
        return x



def add_noise(data :torch.tensor, mu : float , sigma : float) -> torch.tensor:
    """
    Adds random Gaussian noise to the input Pandas DataFrame.
    
    Args:
        data (pd.DataFrame): Input data as a Pandas DataFrame.
        mu (float): Mean of the noise.
        sigma (float): Standard deviation of the noise.
    
    Returns:
        pd.DataFrame: Noisy data as a Pandas DataFrame.
    """
    # Generate noise with the same shape as the input data
    noise = np.random.normal(loc=mu, scale=sigma, size=data.shape)
    
    # Add noise to the data
    noisy_data = data + noise
    
    return noisy_data


def train_dsae(model, training_data, criterion, optimizer, l1_lambda=1e-12, mu = 0, sigma = 1, tol=1e-3, relative_tol=True, max_iter = 10_000_000):

    training_data_tensor = torch.tensor(training_data.values, dtype=torch.float32)
    training_noisy_tensor = add_noise(training_data_tensor, mu, sigma)

    training_loss = []
    diff = float('inf')
    iters = 0

    while diff > tol and iters < max_iter:
        optimizer.zero_grad()
        output = model(training_noisy_tensor)
        loss = criterion(output, training_data_tensor)
        l1_penalty = l1_lambda * torch.norm(model.encoder[0].weight, p=1)
        loss += l1_penalty

        loss.backward()
        optimizer.step()

        # Clamp the weights to non-negative values
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.clamp_(min=0)
        
        training_loss.append(loss.item())

        if len(training_loss) > 1:
            if relative_tol:
                diff = abs(training_loss[-1] - training_loss[-2]) / training_loss[-2]
            else:
                diff = abs(training_loss[-1] - training_loss[-2])
        
        
        iters += 1
    
    enc_weights = model.encoder[0].weight.data.T
    dec_weights = model.decoder[0].weight.data.T

    if torch.any(enc_weights < 0):
        raise ValueError("Negative values present in the encoder weights")
    if torch.any(dec_weights < 0):
        raise ValueError("Negative values present in the decoder weights")
    
    exposure = training_data @ enc_weights
    signature = dec_weights

    return model, training_loss, signature.T, exposure.T