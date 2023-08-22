
import torch
import torch.nn as nn
from functools import partial
from hflayers import Hopfield

from .utils.initialization import get_initializer


def init_lecun(m):
    nn.init.normal_(m.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0])
    nn.init.zeros_(m.bias)


def init_kaiming(m, nonlinearity):
    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
    nn.init.zeros_(m.bias)


@torch.no_grad()
def init_weights(m, activation_function='linear'):
    if activation_function == 'relu':
        if type(m) == nn.Linear:
            init_kaiming(m, nonlinearity='relu')
    elif activation_function == "selu":
        if type(m) == nn.Linear:
            init_lecun(m)
    elif activation_function == 'linear':
        if type(m) == nn.Linear:
            init_lecun(m)


class ContextModule(nn.Module):
    """
    Context module using Modern Hopfield network to enrich protein target embedding using an external memory.
    Author: Johannes Schimunek
    """
    def __init__(self, in_channels, args):
        super(ContextModule, self).__init__()

        self.hopfield = Hopfield(
            input_size=in_channels,                # R
            hidden_size=args['QK_dim'],            # a_1, Dimension Queries, Keys
            stored_pattern_size=in_channels,       # Y
            pattern_projection_size=in_channels,   # Y
            output_size=in_channels,               # a_2, Dim Values / Dim Dot product
            num_heads=args['heads'],
            scaling=args['beta'],
            dropout=args['dropout']
        )

        # Initialization
        #hopfield_initialization = get_initializer('linear')
        hopfield_initialization = partial(init_weights, 'linear')
        self.hopfield.apply(hopfield_initialization)
        self.skip = args['skip']

    def forward(self, embedding, memory):

        s = embedding.unsqueeze(0)
        memory = memory.unsqueeze(0)
        s_h = self.hopfield((memory, s, memory))
        enriched_embedding = s_h
        if self.skip:
            enriched_embedding += s
        return enriched_embedding.squeeze(0)

