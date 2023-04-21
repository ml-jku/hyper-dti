
import torch
import torch.nn as nn

from hyper_dti.settings import constants
from hyper_dti.models.context import ContextModule
from hyper_dti.models.noise import EncoderHead
from hyper_dti.models.mlp import Classifier


class DeepPCM(nn.Module):
    """
    Re-implementation of DeepPCM model from (Kim, P. T., et al., 2021) https://www.mdpi.com/1422-0067/22/23/12882.
    Optionally adds a context module for drug and/or target embeddings.
    Author: Emma Svensson
    """

    def __init__(self, args, drug_encoder, target_encoder, memory=None):
        super(DeepPCM, self).__init__()
        self.memory = memory

        drug_dim = constants.DRUG_LATENT_DIM[drug_encoder]
        target_dim = constants.TARGET_LATENT_DIM[target_encoder]

        self.context = {'drug': args['drug_context'], 'target': args['target_context']}
        if args['drug_context']:
            self.drug_context = ContextModule(in_channels=drug_dim, args=args['hopfield'])
        if args['target_context']:
            self.target_context = ContextModule(in_channels=target_dim, args=args['hopfield'])

        self.drug_encoder = EncoderHead(drug_dim)
        self.target_encoder = EncoderHead(target_dim)
        in_channels = drug_dim + 512 + target_dim + 512
        hidden_layers = [in_channels, 2048, 1024]

        self.classifier = Classifier(hidden_layers=hidden_layers)

    def forward(self, batch):
        device = next(self.classifier.parameters()).device
        drug_batch = batch['drugs'].to(device)
        target_batch = batch['targets'].to(device)

        if self.memory is not None:
            if self.context['drug']:
                memory = self.memory.get_drug_memory(exclude_mids=batch['mids'] if self.train else []).to(device)
                drug_batch = self.drug_context(drug_batch, memory)
            if self.context['target']:
                memory = self.memory.get_target_memory(exclude_pids=batch['pids'] if self.train else []).to(device)
                target_batch = self.target_context(target_batch, memory)

        drug_batch = self.drug_encoder(drug_batch.float())
        target_batch = self.target_encoder(target_batch.float())

        x = self.classifier(torch.cat((drug_batch, target_batch), dim=1))

        return x.squeeze()

