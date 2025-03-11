import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class aenmf(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization

    """
    def __init__(self, input_dim, latent_dim):
        '''
        Constructor for the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

        Parameters:
        - input_dim: Dimension of the input data
        - latent_dim: Dimension of the latent space
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.activation = nn.Identity()
        

        ''' Encoder and Decoder layers '''
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            self.activation
        )

        ''' Initialize weights with random values '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=0.0, b=1.0) # Random initialization between 0 and 1

                # Ensure non-negative weights
                module.weight.data.clamp_(min=0)



    def forward(self, x):
        '''
        Forward pass of the Autoencoder for Non-negative Matrix Factorization

        '''
        x = self.encoder(x)
        x = self.decoder(x)
    
        return x
    
    
def train_aenmf(model, training_data, criterion, optimizer, tol = 1e-3, relative_tol = True, max_iter = 100_000_000):
    '''
    Function to train the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

    Parameters:
    - model: AENMF model
    - training_data: Training data (Note, we assume to reconstruct X^T = E^TS^T so the input data should have shape n x m (m being 96))
    - criterion: Loss function
    - optimizer: Optimizer
    - tol: Tolerance for convergence
    - relative_tol: If True, the tolerance is relative. If False, the tolerance is absolute.
    - max_iter: Maximum number of iterations
    
    Returns:
    - model: Trained model
    - training_loss: Training loss
    - signature: Signature matrix
    - exposure: Exposure matrix
    '''

    training_data_tensor = torch.tensor(training_data.values, dtype = torch.float32)

    training_loss = []
    diff = float('inf')

    iters = 0
    while diff > tol and iters < max_iter: # Convergence criterion
        optimizer.zero_grad() 
        output = model(training_data_tensor)
        loss = criterion(output, training_data_tensor)
        loss.backward()
        optimizer.step()

        # Clamp the weights to non-negative values
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.clamp_(min=0)  # Ensure non-negative weights

        training_loss.append(loss.item())


        if len(training_loss) > 1:
            if relative_tol:
                diff = abs(training_loss[-1] - training_loss[-2])/training_loss[-2]
            else:
                diff = abs(training_loss[-1] - training_loss[-2])

        
        # Go to next iteration
        iters += 1
    

    # Get the encoder and decoder weights

    enc_weights = model.encoder[0].weight.data.T
    dec_weights = model.decoder[0].weight.data.T

    if(torch.any(enc_weights < 0)):
        raise ValueError("Negative values present in the encoder weights")
    if(torch.any(dec_weights < 0)):
        raise ValueError("Negative values present in the decoder weights")

    exposure = training_data @ enc_weights
    signature = dec_weights 



    return model, training_loss, signature.T, exposure.T