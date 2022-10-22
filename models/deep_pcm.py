import torch
import torch.nn as nn

from settings import constants
from models.context import ContextModule
from models.noise import EncoderHead
from models.mlp import Classifier


class DeepPCM(nn.Module):
    """
    Re-implementation of DeepPCM model from (Kim, P. T., et al., 2021) https://www.mdpi.com/1422-0067/22/23/12882.
    Optionally adds a context module for molecule and/or protein embeddings.
    Author: Emma Svensson
    """

    def __init__(self, args, molecule_encoder, protein_encoder, memory=None):
        super(DeepPCM, self).__init__()
        self.memory = memory

        molecule_dim = constants.MOLECULE_LATENT_DIM[molecule_encoder]
        protein_dim = constants.PROTEIN_LATENT_DIM[protein_encoder]

        self.context = {'molecule': args['molecule_context'], 'protein': args['protein_context']}
        if args['molecule_context']:
            self.molecule_context = ContextModule(in_channels=molecule_dim, args=args['hopfield'])
        if args['protein_context']:
            self.protein_context = ContextModule(in_channels=protein_dim, args=args['hopfield'])

        self.molecule_encoder = EncoderHead(molecule_dim)
        self.protein_encoder = EncoderHead(protein_dim)
        in_channels = molecule_dim + 512 + protein_dim + 512
        hidden_layers = [in_channels, 2048, 1024]

        self.classifier = Classifier(hidden_layers=hidden_layers)

    def forward(self, batch):
        device = next(self.classifier.parameters()).device
        molecule_batch = batch['molecules'].to(device)
        protein_batch = batch['proteins'].to(device)

        if self.memory is not None:
            if self.context['molecule']:
                memory = self.memory.get_molecule_memory(exclude_mids=batch['mids'] if self.train else []).to(device)
                molecule_batch = self.molecule_context(molecule_batch, memory)
            if self.context['protein']:
                memory = self.memory.get_protein_memory(exclude_pids=batch['pids'] if self.train else []).to(device)
                protein_batch = self.protein_context(protein_batch, memory)

        molecule_batch = self.molecule_encoder(molecule_batch.float())
        protein_batch = self.protein_encoder(protein_batch.float())

        x = self.classifier(torch.cat((molecule_batch, protein_batch), dim=1))

        return x.squeeze()

