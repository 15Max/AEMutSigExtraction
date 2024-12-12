import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, regularization=1e-12):
        """
        Initializes the denoising autoencoder.
        
        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space (encoding).
            regularization (float): Regularization factor for the encoder layer.
        """
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
            # nn.Dropout(regularization)  # Add regularization-like effect, it could be used instead of L1 regularization
        )

        # Regularization (L1)
        self.l1_regularization = regularization

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Softmax(dim=-1) # dim=1
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)

        # Decode
        decoded = self.decoder(encoded)
        return decoded

    def get_encoder(self):
        # Returns the encoder model
        return self.encoder

    def get_decoder(self):
        # Returns the decoder model
        return self.decoder

# Function to initialize and return the model
def build_denoising_autoencoder(input_dim, latent_dim, regularization=1e-12):
    """
    Builds a denoising autoencoder model.

    Args:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space (encoding).
        regularization (float): Regularization factor for the encoder layer.

    Returns:
        autoencoder (DenoisingAutoencoder): PyTorch autoencoder model.
    """
    model = DenoisingAutoencoder(input_dim, latent_dim, regularization)
    return model
