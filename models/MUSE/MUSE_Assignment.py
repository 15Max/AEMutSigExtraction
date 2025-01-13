import numpy as np 
import os
import torch 
from .MUSE_Ae import *
from data_handling import data_normalization, data_augmentation
from sklearn.cluster import KMeans


def train_model_for_extraction(model: HybridAutoencoder,
                Signature_matrix: np.ndarray,
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
    Signature_matrix (np.ndarray): The signature matrix to use as decoder weights
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
    E (np.ndarray): The exposure matrix  (as signatures x n)
    train_losses (list): The training losses
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_scaled = data_normalization(X_scaled)
    X_noise = data_augmentation(X_scaled, augmentation=1)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_noise.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)


    # print("X_train_tensor shape: ", X_train_tensor.shape)

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)

    # Assert that model.refit is True
    assert model.refit == True

    # Normalize the signature matrix

    Signature_matrix = Signature_matrix / Signature_matrix.sum(axis=0)

    model.assign_decoder_weights(signature_matrix=Signature_matrix)

    # Freeze the decoder weights

    for param in model.decoder.parameters():
        param.requires_grad = False


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

            output, _ , _, l1_penalty = model(batch_X)
            
            # print("SIGNATURE MATRIX (DECODER WEIGHTS) SIZE: ", signature_mat.size())
            # print("EXPOSURE MATRIX (LATENT REPR) SIZE: ", exposures_mat.size())

            loss = criterion(x=batch_X, x_hat=output, decoder_weights=Signature_matrix)
            loss += l1_penalty  # Add L1 penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)  # Accumulate batch loss

        epoch_loss /= len(train_dataset)  # Compute average epoch loss
        train_losses.append(epoch_loss)  # Store training loss

        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_output, val_exposures, val_sign, l1_val = model(X_val_tensor)
            val_loss = criterion(x=X_val_tensor, x_hat=val_output, decoder_weights=val_sign).item()
            val_loss += l1_val
            
        val_losses.append(val_loss)  # Store validation loss

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.6f} - Validation Loss: {val_loss:.6f}")

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
        E, _ = encoder(X_val_tensor)  # Get encoded features
        E = E.cpu().detach().numpy()
        S = model.decoder.fc1.weight.cpu().detach().numpy()  # Get decoder weight S
    
    # Reconstruction Error as Frobenius Norm
    error = np.linalg.norm(X_scaled.values - np.dot(E, S.T))

    return error, encoder, train_losses, val_losses  
