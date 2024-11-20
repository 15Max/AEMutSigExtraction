import torch
import torch.nn.functional as F


class NMF_AE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, init_enc = None, init_dec = None):
        super(NMF_AE, self).__init__()
        if init_enc is None:
            self.encoder = torch.nn.Parameter(torch.rand(input_dim, latent_dim))
        else:
            self.encoder = torch.nn.Parameter(torch.Tensor(init_enc))

    def forward(self, x):
        x = torch.matmul(x, torch.abs(self.enc_weight))
        x = torch.matmul(x, torch.abs(self.dec_weight))
        return x
    


def train():
    pass
        



    
    

