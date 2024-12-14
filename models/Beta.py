import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the VAE
class BetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decoder1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decoder2 = nn.Linear(hidden_dim, input_dim)
        self.beta = beta

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_decoder1(z))
        out = torch.sigmoid(self.fc_decoder2(h))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_divergence


def train(model, training_data, epochs, optimizer, batch_size, device):
    model.train()
    all_signatures = []
    all_exposures = []

    for epoch in range(epochs):
        train_loss = 0
        signatures = []
        exposures = []

        for i in range(0, len(training_data), batch_size):
            x = training_data[i:i + batch_size]
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Collect signature and exposure matrices
            signatures.append(mu.detach().cpu())  # Latent representations (signatures)
            exposures.append(x.detach().cpu())    # Original data (exposures)

        signatures = torch.cat(signatures, dim=0)
        exposures = torch.cat(exposures, dim=0)

        all_signatures.append(signatures)
        all_exposures.append(exposures)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(training_data):.4f}")

    return torch.cat(all_signatures, dim=0), torch.cat(all_exposures, dim=0)

 