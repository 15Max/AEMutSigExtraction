import torch
import torch.nn as nn
import torch.nn.functional as F


def OrthogonalRegularizer(beta, W):
   
    Gram = torch.mm(W, W.t())
    I = torch.eye(Gram.size(0))

    loss = beta * torch.norm(Gram - I, p='fro')

    return loss

def L2Regularizer(beta, W):
    loss = beta * torch.norm(W, p=2)
    return loss


def MinimumVolumeRegularizer(W, dim, beta):
    Gram = torch.mm(W, W.T)  # W * W^T
    I = torch.eye(dim)  # Identity matrix
    det = torch.det(Gram + I)  # Compute determinant

    print("Dim of the class: ", dim)
    print("Dim obtained from matrix: ", W.size(0))  


    # Numerical stability fix (add a small epsilon)
    log_det = torch.log(det + 1e-6) / torch.log(10.0)

    # Constraint: Scale down W based on log determinant
    return W / (1 + beta * log_det)


class Encoder(nn.Module):
    def __init__(self, input_dim, l_1, k, refit=False, refit_penalty=1e-3):
        super(Encoder, self).__init__()
        
        self.refit = refit
        self.refit_penalty = refit_penalty
        self.activation = F.relu if refit else F.softplus

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

        # Adding the last layer separately to access its weights easily
        if refit:
            self.last_layer = nn.Linear(l_1 // 4, k)  # Latent layer
            self.activation_fn = nn.ReLU()
        else:
            self.last_layer = nn.Linear(l_1 // 4, k)
            self.activation_fn = nn.Softplus()

    def forward(self, x):
        x = self.encoder(x)  # Pass through encoder layers
        x = self.last_layer(x)  # Last layer (Linear)
        x = self.activation_fn(x)  # Activation

        reg_loss = None
        if self.refit:

            # Compute L1 regularization on the activation, note that in the paper it is said that the regularization
            # happens on both the output of the encoder and the weights of the last layer of the encoder, but as of now
            # (commit 094f902) the regularization is only applied to the output of the encoder. 

            reg_loss = self.refit_penalty * torch.norm(x, p=1)
            return x, reg_loss

        return x, None


class Decoder(nn.Module):
    def __init__(self, input_dim, beta = None, z = None):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(z, input_dim, bias=False)

        
        self.beta = beta
        self.z = z

    def forward(self, x):

        x = self.fc1(x)  # Linear activation

        # Enforce non-negativity constraint on weights
        self.fc1.weight.data.clamp_(min=0)  # Clamp weights to be ≥ 0


        if self.beta is not None:
            reg_loss = MinimumVolumeRegularizer(W = self.fc1.weight, dim = self.z, beta = self.beta)
            return x, reg_loss
        return x, None

class HybridAutoencoder(nn.Module):
    # In the paper it is assumed that the regularizer term is always the Minimum Volume Regularizer so we won't use the other regularizers

    def __init__(self, input_dim=96, l_1=128, z=17, beta=0.001, refit=False, refit_penalty=1e-3):
        super(HybridAutoencoder, self).__init__()

        self.encoder = Encoder(input_dim, l_1, z, refit, refit_penalty)
        self.decoder = Decoder(input_dim, z, beta)

    def forward(self, x):
        exposures, reg_loss_enc = self.encoder(x)
        reconstruction, reg_loss_dec = self.decoder(exposures)

        total_loss = 0
        if reg_loss_enc is not None:
            total_loss += reg_loss_enc
        if reg_loss_dec is not None:
            total_loss += reg_loss_dec

        return reconstruction, exposures, total_loss




        
    