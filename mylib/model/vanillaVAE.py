import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .module.baseModel import *
from .module.utils import *
from .modelbuilder import get_encoder, get_decoder

class VAE(nn.Module):
    def __init__(self, encoder_layer_dims, decoder_layer_dims, mode='conv', 
                    activation=nn.ReLU, act_param=[True]):

        super(VAE, self).__init__()

        assert type(encoder_layer_dims) == list
        assert type(decoder_layer_dims) == list

        self.mode = mode
        self.encoder = get_encoder(self.mode, encoder_layer_dims, activation, act_param)
        self.decoder = get_decoder(self.mode, decoder_layer_dims, activation, act_param)

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def encode_(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode_(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode_(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_(z), mu, logvar