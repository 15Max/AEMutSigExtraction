import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
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
        npll_loss = torch.sum(x_hat - x * torch.log(x_hat + self.eps))  # Add stability

        # Apply Minimum Volume Regularization (as constraint)
        gram_matrix = torch.mm(decoder_weights, decoder_weights.T)
        identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        _, log_det_value = torch.slogdet(gram_matrix + identity)  # Use numerically stable determinant

        total_loss = npll_loss + (self.beta * log_det_value) + reg_enc_loss

        return total_loss

class Abs(nn.Module):
    '''
    Custom Abs module to wrap torch.abs for use in nn.Sequential
    '''
    def forward(self, x):
        return torch.abs(x)

class Encoder(nn.Module):
    '''
    Class of the Encoder model. The Encoder model is a simple feedforward neural network that takes the input data
    and encodes it into a latent representation.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    l_1 (int): The number of neurons in the first layer of the encoder
    latent_dim (int): The number of latent features
    refit (bool): Whether to use the refitting mechanism (for signature extraction)
    refit_penalty (float): The penalty for the refitting mechanism (for signature extraction)
    '''
    def __init__(self, input_dim, l_1, latent_dim):
        super(Encoder, self).__init__()
    
        self.latent_activation =  F.softplus

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

        self.last_layer = nn.Linear(l_1 // 4, latent_dim)  # Latent layer

        self.initialize_weights()

    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=0.0, b=1.0)

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
        x = self.latent_activation(x)  # Activation

        return x


class Decoder(nn.Module):
    '''
    Class for the Decoder model. The Decoder model is a simple linear layer that takes the latent features and
    reconstructs the data.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    latent_dim (int): The number of latent features
    constraint (str): Different activation functions
    '''
    def __init__(self, input_dim : int,  latent_dim : int, constraint = 'pg'):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, input_dim, bias=False)
        self.constraint = constraint
        self.latent_dim = latent_dim

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=0.0, b=1.0)
                module.weight.data.clamp_(min=0)

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


        if(self.constraint == 'pg'):
            x = nn.ReLU(x)
        elif (self.constraint == 'abs'):
            x = Abs(x)
        elif (self.constraint == 'identity'):
            pass # just pretty, does nothings
        else:
            raise ValueError('Constraint not recognized. Choose between pg, abs or identity')
 
        self.fc1.weight.data = nn.Parameter(torch.abs(self.fc1.weight))   

        return x, self.fc1.weight.data

class HybridAutoencoder(nn.Module):
    '''
    Hybrid Autoencoder model that combines the Encoder and Decoder models. The model is trained using the HybridLoss
    which is a combination of the Negative Poisson Log-Likelihood Loss and the Minimum Volume Regularizer.

    Parameters:
    input_dim (int): The input dimension of the model (96)
    l_1 (int): The number of neurons in the first layer of the encoder
    latent_dim (int): The number of latent features
    refit (bool): Whether to use the refitting mechanism
    refit_penalty (float): The penalty for the refitting mechanism
    constraint (str): The constraint to apply on the decoder weights, can be 'pg', 'abs' or 'identity'

    '''

    # In the paper it is assumed that the regularizer term is always the Minimum Volume Regularizer so we won't use the other regularizers
    # Also, it is assumed for the matrix to be passed as n x 96, so we will transpose the matrix before passing it to the model

    def __init__(self, input_dim : int = 96, l_1 : int = 128, latent_dim : int = 17, constraint : str = 'identity'):
        super(HybridAutoencoder, self).__init__()

        self.encoder = Encoder(input_dim, l_1, latent_dim) 
        self.decoder = Decoder(input_dim, latent_dim, constraint)                


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
        exposures = self.encoder(x)
        reconstruction, signature_matrix = self.decoder(exposures)

        return reconstruction, exposures, signature_matrix


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
    


def train_model_for_extraction(model: HybridAutoencoder,
                X_aug_multi_scaled: pd.DataFrame,
                X_scaled: pd.DataFrame,           
                signatures: int,
                epochs: int,
                batch_size: int,
                save_to: str,
                iteration: int,
                patience: int = 30): 
    '''
    Function to train the Hybrid Autoencoder model.

    Parameters:
    model (HybridAutoencoder): The model to train
    X_aug_multi_scaled (pd.DataFrame): The augmented data to train on (dimension should be (n x augmentations) x 96)
    X_scaled (pd.DataFrame): The original data to validate on (dimension should be n x 96)
    signatures (int): The number of signatures to learn 
    epochs (int): The number of epochs to train for
    batch_size (int): The batch size for training
    save_to (str): The directory to save the model
    iteration (int): The iteration number
    patience (int): The patience for early stopping

    Returns:
    error (float): The error of the model
    S (np.ndarray): The signature matrix (as 96 x signatures)
    train_losses (list): The training losses
    val_losses (list): The validation losses
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_aug_multi_scaled.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)

    # print("X_train_tensor shape: ", X_train_tensor.shape)

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)
    criterion = HybridLoss(beta=0.001)
    optimizer = optim.Adam(model.parameters())

    best_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_to, f'best_model_{signatures}_{iteration}.pt')

    # for visualization
    train_losses = []
    val_losses = []

    # Training Loop with Mini-Batches
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0  # Track epoch loss
        
        for batch in train_loader:
            batch_X = batch[0]  # Get input batch
            optimizer.zero_grad()

            output, _ , signature_mat = model(batch_X)

            
            # print("SIGNATURE MATRIX (DECODER WEIGHTS) SIZE: ", signature_mat.size())
            # print("EXPOSURE MATRIX (LATENT REPR) SIZE: ", exposures_mat.size())

            loss = criterion(x=batch_X, x_hat=output, decoder_weights=signature_mat)
            loss.backward()
            optimizer.step()

            # Enforce non-negativity constraint on weights via clamping

            for module in model.encoder.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.clamp_(min=0) 
            
            for module in model.decoder.modules():    
                if isinstance(module, nn.Linear):
                    module.weight.data.clamp_(min=0)



            epoch_loss += loss.item() * batch_X.size(0)  # Accumulate batch loss

        epoch_loss /= len(train_dataset)  # Compute average epoch loss
        train_losses.append(epoch_loss)  # Store training loss

        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_output, _ , val_sign = model(X_val_tensor)
            val_loss = criterion(x=X_val_tensor, x_hat=val_output, decoder_weights=val_sign).item()
        
        val_losses.append(val_loss)  # Store validation loss


        #if (epoch + 1) % 1000 == 0:
            #print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.6f} - Validation Loss: {val_loss:.6f}")

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            
            # Ensure the directory exists before saving
            if not os.path.exists(save_to):
                os.makedirs(save_to)  # Create the directory if it doesn't exist
            torch.save(model.state_dict(), best_model_path)  # Save best model
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load Best Model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Extract Encoder
    encoder = model.return_encoder_model()

    # Compute Error Metric
    with torch.no_grad():
        E = encoder(X_val_tensor)  # Get encoded features
        E = E.cpu().detach().numpy()
        S = model.return_decoder_weights()
        S = S.cpu().detach().numpy()
    

    # Reconstruction Error as frobenius norm
    error = np.linalg.norm(X_scaled.values - np.dot(E, S.T), ord='fro')

    return error, S, E.T, train_losses, val_losses 