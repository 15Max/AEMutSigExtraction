import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

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


def train_dsae(model, train_loader, test_loader, criterion, optimizer, device, epochs, l1_lambda=1e-12, tol=1e-3, relative_tol=True, patience=5):
    model.to(device)
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0

        for batch_noisy, batch_clean in train_loader:
            batch_noisy, batch_clean = batch_noisy.to(device), batch_clean.to(device)

            optimizer.zero_grad()
            output = model(batch_noisy)

            loss = criterion(output, batch_clean)
            l1_penalty = l1_lambda * torch.norm(model.encoder[0].weight, p=1)  # L1 penalty on encoder weights
            loss += l1_penalty

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)  # Normalize by the number of batches
        train_losses.append(train_loss)

        # Check for convergence on training loss
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

        # Early stopping with patience (optional, based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Extract signature and exposure matrices from the model
    enc_weights = model.encoder[0].weight.data.T 
    dec_weights = model.decoder[-1].weight.data.T
    
    signature = train_loader.dataset.data @ enc_weights
    exposure = dec_weights

    return model, train_losses, val_losses, signature, exposure