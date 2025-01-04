import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, regularization=1e-12, count = True):
        """
        Initializes the denoising autoencoder.
        
        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space (encoding).
            regularization (float): Regularization factor for the encoder layer.
        """
        super(DenoisingAutoencoder, self).__init__()
        
        
        self.l1_regularization = regularization
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU() #Should we keep the ReLU? or should this be the identity function?
        )

        # Decoder
        if count:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim),
                nn.ReLU()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim),

                nn.Softmax(dim=-1) # dim=1
            )

    def forward(self, x):
        x = self.encoder(x)  # Encode
        x = self.decoder(x)  # Decode
        return x


    def get_encoder(self):
        # Returns the encoder model
        return self.encoder

    def get_decoder(self):
        # Returns the decoder model
        return self.decoder
    


# This noise function is now adapted to frequency usage
# We are going to use count data
def add_noise_freq(data_tensor, noise_factor=0.5):
    """
    Adds random gaussian noise to the input data.
    
    Args:
        data (torch.Tensor): Input data.
        noise_factor (float): Noise factor.
    
    Returns:
        torch.Tensor: Noisy data.
    """
    noise = noise_factor * torch.randn_like(data_tensor)
    noisy_data = data_tensor + noise
    noisy_data = torch.clamp(noisy_data, 0., 1.)
    return noisy_data

def add_noise_count(data_tensor, sigma = 10): # maybe pass sigma directly
    """
    Adds random gaussian noise to the input data.
    
    Args:
        data (torch.Tensor): Input data.
        noise_factor (float): Noise factor.
    
    Returns:
        torch.Tensor: Noisy data.
    """
    #sigma = noise_factor * torch.mean(data_tensor) # Is this a good choice for sigma?
    noise = sigma * torch.randn_like(data_tensor)
    noisy_data = data_tensor + noise
    noisy_data = torch.clamp(noisy_data, min= 0).int() # We need non-negative integers
    return noisy_data
    

def train():
    pass

