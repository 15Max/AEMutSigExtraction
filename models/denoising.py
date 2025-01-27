import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import pandas as pd
import numpy as np

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class dsae(nn.Module):
    def __init__(self, input_dim, latent_dim, constraint, xavier= False):
        """
        Initializes the denoising autoencoder.
        
        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space 
            constraint (str): Constraint type ('pg' for projected gradient, 'abs' for absolute values).
        """
        super(dsae, self).__init__()
        
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint
        self.xavier = xavier
    
        if self.constraint == 'pg':
            self.activation = nn.ReLU()
        elif self.constraint == 'abs':
            self.activation = Abs()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation 
        )

        # Decoder
        
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.input_dim),
                self.activation
        )
       
        if self.xavier:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                

    def forward(self, x):
        x = self.encoder(x)  
        x = self.decoder(x)  
        return x

def add_noise(data: pd.DataFrame, mu: float, sigma: float) -> pd.DataFrame:
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


def train_dsae(model, training_data, criterion, optimizer, tol = 1e-3, relative_tol = True, max_iter = 100_000_000, l1_lambda=1e-12):
    training_data_tensor = torch.tensor(training_data.values, dtype = torch.float32)

    training_loss = []
    diff = float('inf')

    iters = 0
    while diff > tol and iters < max_iter: # Convergence criterion
        optimizer.zero_grad() 
        output = model(training_data_tensor)
  
        loss = criterion(output, training_data_tensor)
        l1_penalty = l1_lambda * torch.norm(model.encoder[0].weight, 1) #L1 regularization
        loss += l1_penalty
        
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())

        if len(training_loss) > 1:
            if relative_tol:
                diff = abs(training_loss[-1] - training_loss[-2])/training_loss[-2]
            else:
                diff = abs(training_loss[-1] - training_loss[-2])

        # Go to next iteration
        iters += 1
    
    signature = training_data @ model.enc_weight.data
    exposure = model.dec_weight.data

    return model, training_loss, signature, exposure

