import torch 
import torch.nn as nn
import torch.functional as F


POSITIVE = 'pg'
ABSOLUTE = 'abs'
XAVIER = 'xavier'
RANDOM = 'random'



class AutoEncoder(nn.Module):
    '''
    Superclass for AutoEncoders which will be used as the base class for the other AutoEncoders
    '''

    def __init__(self):

        super().__init__()
        self.weights = None

    def layers_initialization(self, type = 'xavier'):
        
        pass

    def init_weights(self, type):

        pass

    def encode(self,x):
        pass



class AE_NMF(AutoEncoder):
    '''
    Autoencoder for Non-negative Matrix Factorization
    '''

    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers_initialization()

    def layers_initialization(self, constraint):

        if constraint == POSITIVE:
            self.input_layer = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim[0]), nn.ReLU())
            self.output_layer = nn.Sequential(nn.Linear(self.latent_dim[-1], self.output_dim), nn.ReLU())

        self.layers.append(self.input_layer)
        self.layers.append(self.output)
        

    def init_weights(self, type):

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if type == XAVIER:
                    nn.init.xavier_normal_(layer.weight)
                elif type == RANDOM:
                    nn.init.normal_(layer.weight, 0, 1)
        
        self.weights = {layer: layer.weight for layer in self.layers}

    def encode(self,x):
        if self.constraint == POSITIVE:
            x = torch.matmul(x, F.relu(self.enc_weight))
            x = torch.matmul(x, F.relu(self.dec_weight))
        elif self.constraint == ABSOLUTE:
            x = torch.matmul(x, torch.abs(self.enc_weight))
            x = torch.matmul(x, torch.abs(self.dec_weight))

        return x