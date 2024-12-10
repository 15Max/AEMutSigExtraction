import torch
import torch.nn.functional as F


class AE_NMF(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization
    Additional constraints can be added to the model:
    - Projected Gradient (pg) 
    - Absolute Values (abs)
    """
    def __init__(self, input_dim, latent_dim, constraint = 'pg'):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint

        # Use xavier initialization for the weights ?
        #self.enc_weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.input_dim, self.latent_dim)))
        #self.dec_weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.latent_dim, self.input_dim)))

        self.enc_weight = torch.nn.Parameter(torch.rand(input_dim, latent_dim))
        self.dec_weight = torch.nn.Parameter(torch.rand(latent_dim, input_dim))

    def forward(self, x):
        if self.constraint == 'pg':
            x = torch.matmul(x, F.relu(self.enc_weight))
            x = torch.matmul(x, F.relu(self.dec_weight))
        elif self.constraint == 'abs':
            x = torch.matmul(x, torch.abs(self.enc_weight))
            x = torch.matmul(x, torch.abs(self.dec_weight))
    
        return x
    
    
def train(model, training_data, criterion, optimizer, tol = 1e-3, relative_tol = True, max_iter = 100_000_000):
    training_data_tensor = torch.tensor(training_data.values, dtype = torch.float32)

    training_loss = [1e10]
    relative_diff = float('inf')

    iter = 0
    while relative_diff > tol and iter < max_iter:
        optimizer.zero_grad()
        output = model(training_data_tensor)
        loss = criterion(output, training_data_tensor)
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
        
        if relative_tol:
            denominator = training_loss[-2]
        else:
            denominator = 1 
        relative_diff = abs(training_loss[-1] - training_loss[-2])/denominator
        
        # Go to next iteration
        iter += 1
        

        with torch.no_grad():
            if model.constraint == 'pg':
                for param in model.parameters():
                    param.clamp_(min=0)
            elif model.constraint == 'abs':
                for param in model.parameters():
                    param.abs_()
          
        
    del training_loss[0]

    # Constraints on final signature and exposure matrices
    if model.constraint == 'pg':    
        enc_mat = (model.enc_weight.data).numpy().clip(min=0)
        sig_mat = (training_data @ enc_mat).to_numpy()
        exp_mat = (model.dec_weight.data).numpy().clip(min=0)
    elif model.constraint == 'abs':
        enc_mat = torch.abs(model.enc_weight).data
        sig_mat = (training_data @ enc_mat).to_numpy()
        exp_mat = (torch.abs(model.dec_weight).data).numpy()

    return (model, training_loss, sig_mat, exp_mat, enc_mat, iter)