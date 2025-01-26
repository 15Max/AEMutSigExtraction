import torch
import torch.nn.functional as F
import torch.nn as nn


# Custom Abs module to wrap torch.abs for use in nn.Sequential
class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    

class aenmf(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization
    Additional constraints can be added to the model:
    - Projected Gradient (pg) 
    - Absolute Values (abs)
    """
    def __init__(self, input_dim, latent_dim, constraint = 'pg', xavier = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint  # Guarantee that the weights are non-negative
        self.xavier = xavier


        if self.constraint == 'pg':
            self.activation = nn.ReLU()
        elif self.constraint == 'abs':
            self.activation = Abs()
        else:
            raise ValueError('Constraint not recognized. Choose between pg and abs')


        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            self.activation
        )

        # Xavier initialization
        if self.xavier:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
    
        return x
    
    
def train_aenmf(model, training_data, criterion, optimizer, tol = 1e-3, relative_tol = True, max_iter = 100_000_000):

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


    signature = training_data @ enc_weights
    exposure = dec_weights 



    return model, training_loss, signature, exposure


# todo: give a more general check to the function