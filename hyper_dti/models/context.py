
import torch.nn as nn
from hflayers import Hopfield

from .utils.initialization import get_initializer


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
        hopfield_initialization = get_initializer('linear')
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

