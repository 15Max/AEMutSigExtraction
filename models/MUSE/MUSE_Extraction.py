import numpy as np 
import torch 
from .MUSE_Ae import *
from sklearn.cluster import KMeans

class KMeans_with_matching:
    def __init__(self, X, n_clusters, max_iter=100):
        """
        Optimized KMeans with Hungarian Matching.

        Parameters:
        - X (numpy array): Data points.
        - n_clusters (int): Number of clusters.
        - max_iter (int): Maximum number of iterations.
        """
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n = X.shape[0]

        # Step 1: Initialize Clusters Using KMeans++
        if self.n_clusters > 1:
            model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10).fit(self.X)
            self.C = model.cluster_centers_

            print("C Shape during init: ", self.C.shape)
            if self.C.shape[0] != self.n_clusters:
                raise ValueError(f"Error: Expected {self.n_clusters} cluster centers, but got {self.C.shape[0]}")
        else:
            self.C = self.X[np.random.choice(self.n, size=1), :]


        self.C_prev = np.copy(self.C)
        self.C_history = [np.copy(self.C)]    

    def fit_predict(self):
        """Runs the optimized KMeans + Hungarian Matching Algorithm."""

        if self.n_clusters == 1:
            # If only one cluster, assign everything to it
            cost = cosine_distances(self.X, self.C)  # Compute cosine distance
            row_ind, colsol = linear_sum_assignment(cost)  # Optimal matching
            self.partition = np.zeros(self.n)  # Assign all to cluster 0
            self.C[0, :] = np.mean(self.X, axis=0)  # Update cluster center
            return pd.DataFrame(self.C).T, self.partition
        

        for k_iter in range(self.max_iter):
            # Step 1: Compute (n, k) cosine distance matrix (faster)
            cost = cosine_distances(self.X, self.C)


            self.partition = np.argmin(cost, axis=1)  # Assigns each point to its nearest cluster

            # Step 3: Update cluster centers
            for i in range(self.n_clusters):
                assigned_points = self.X[self.partition == i]
                if len(assigned_points) > 0:
                    self.C[i, :] = np.mean(assigned_points, axis=0)

            self.C_history.append(np.copy(self.C))

            # Step 4: Early stopping if cluster centers do not change
            if np.allclose(self.C, self.C_prev, atol=1e-6):
                break
            self.C_prev = np.copy(self.C)

        return pd.DataFrame(self.C).T, self.partition



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

            output, _ , signature_mat, _ = model(batch_X)
            
            # print("SIGNATURE MATRIX (DECODER WEIGHTS) SIZE: ", signature_mat.size())
            # print("EXPOSURE MATRIX (LATENT REPR) SIZE: ", exposures_mat.size())

            loss = criterion(x=batch_X, x_hat=output, decoder_weights=signature_mat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)  # Accumulate batch loss

        epoch_loss /= len(train_dataset)  # Compute average epoch loss
        train_losses.append(epoch_loss)  # Store training loss

        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_output, _ , val_sign, _ = model(X_val_tensor)
            val_loss = criterion(x=X_val_tensor, x_hat=val_output, decoder_weights=val_sign).item()
        
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
        S = model.return_decoder_weights()
        S = S.cpu().detach().numpy()
    
    # Reconstruction Error as Frobenius Norm
    error = np.linalg.norm(X_scaled.values - np.dot(E, S.T))

    return error, S, train_losses, val_losses  
