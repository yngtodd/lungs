import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = encoder(x)
        x = decoder(x)
        return x
