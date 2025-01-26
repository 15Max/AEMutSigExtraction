import torch
import torch.nn.functional as F


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

        if xavier:
            self.enc_weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.input_dim, self.latent_dim)))
            self.dec_weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.latent_dim, self.input_dim)))
        else:
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
    

    signature = training_data @ model.enc_weight.data
    exposure = model.dec_weight.data



    return model, training_loss, signature, exposure


# todo: give a more general check to the function