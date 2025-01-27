import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

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
    Adds random gaussian noise to the input data.
    
    Args:
        data (torch.Tensor): Input data.
        mu (float): Mean of the noise.
        sigma (float): Standard deviation of the noise.
    
    Returns:
        torch.Tensor: Noisy data.
    
    """
    noise = sigma * torch.randn_like(data) + mu
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

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_noisy, batch_clean in train_loader:
            # Both `batch_noisy` and `batch_clean` are (96, batch_size)
            batch_noisy, batch_clean = batch_noisy.to(device), batch_clean.to(device)

            optimizer.zero_grad()
            output = model(batch_noisy)  # Forward pass through the model

            loss = criterion(output, batch_clean)  # Compute reconstruction loss
            l1_penalty = l1_lambda * torch.norm(model.encoder[0].weight, p=1)  # L1 penalty on encoder weights
            loss += l1_penalty

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)  # Normalize by the number of batches
        train_losses.append(train_loss)

        # Check for convergence based on training loss
        if len(train_losses) > 1:
            if relative_tol:
                diff = abs(train_losses[-1] - train_losses[-2]) / train_losses[-2]
            else:
                diff = abs(train_losses[-1] - train_losses[-2])

            if diff < tol:
                print(f"Convergence reached at epoch {epoch+1}")
                break

        # Validation Loss (optional, for monitoring only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_noisy, batch_clean in test_loader:
                batch_noisy, batch_clean = batch_noisy.to(device), batch_clean.to(device)
                output = model(batch_noisy)
                val_loss += criterion(output, batch_clean).item()

        val_loss /= len(test_loader)  # Normalize by the number of batches
        val_losses.append(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Extract signature and exposure matrices from the model
    enc_weights = model.encoder[0].weight.data.T  # Shape (96, 4)
    dec_weights = model.decoder[0].weight.data.T  # Shape (4, 96)

    # Compute signature and exposure matrices
    signature = train_loader.dataset.noisy_data @ enc_weights  # Shape (96, 4)
    exposure = dec_weights  # Shape (4, 96)

    return model, train_losses, val_losses, signature, exposure