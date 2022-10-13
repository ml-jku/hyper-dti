
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

import constants
from models.noise import EncoderHead
from models.context import ContextModule
from models.mlp import Classifier
from models.utils.initialization import get_initializer


class MainFCN(nn.Module):
    """
    Main module for molecular inputs, predicting the drug-target interactions.
    Author: Emma Svensson
    """

    def __init__(self, molecule_encoder, cls_args):
        super(MainFCN, self).__init__()

        self.molecule_encoder = EncoderHead(constants.MOLECULE_LATENT_DIM[molecule_encoder])
        in_channels = constants.MOLECULE_LATENT_DIM[molecule_encoder] + 512
        hidden_layers = [in_channels]
        for _ in range(cls_args['layers']):
            hidden_layers.append(cls_args['hidden_dim'])
        self.classifier = Classifier(hidden_layers=hidden_layers)

    def forward(self, x):
        x = self.molecule_encoder(x)
        x = self.classifier(x)
        return x.squeeze()


class HyperFCN(nn.Module):
    """
    HyperNetwork for protein inputs, predicting the parameters of the main network.
    Author: Emma Svensson
    """

    def __init__(self, protein_encoder, main_architecture, fcn_args, hopfield_args):
        super(HyperFCN, self).__init__()

        in_channels = constants.PROTEIN_LATENT_DIM[protein_encoder]

        self.context = hopfield_args['context_module']
        self.layerNorm = None
        if self.context:
            self.protein_context = ContextModule(in_channels=in_channels, args=hopfield_args)
            if hopfield_args['layer_norm']:
                self.layerNorm = nn.LayerNorm(in_channels, elementwise_affine=False)

        hidden_layers = [in_channels]
        for _ in range(fcn_args['layers']):
            hidden_layers.append(fcn_args['hidden_dim'])
        self.main_architecture = list(main_architecture.items())

        inplace = False
        self.phi = nn.SELU(inplace=inplace) if fcn_args['selu'] else nn.ReLU(inplace=inplace)
        p = 0.25 / 5 if fcn_args['selu'] else 0.25
        dropout = nn.AlphaDropout(p=p, inplace=inplace) if fcn_args['selu'] else nn.Dropout(p=p, inplace=inplace)

        # Shared MLP for dedicated internal embedding
        self.layers = nn.ModuleList()
        for layer in range(len(hidden_layers) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_layers[layer], hidden_layers[layer + 1]),
                    self.phi,
                    dropout
                )
            )

        # Individual decoder heads for each layer of main architecture
        self.heads = nn.ModuleList()
        for param_name, param in main_architecture.items():
            head = nn.Sequential()
            head.add_module('W_p', nn.Linear(hidden_layers[-1], param.nelement()))
            if fcn_args['norm'] == 'learned':
                head.add_module('layer_norm', nn.LayerNorm(param.nelement()))
            self.heads.append(head)

        self.init = fcn_args['init']
        if self.init != 'default':
            self.reset_parameters()

    def reset_parameters(self):
        if 'manual' in self.init:
            init = get_initializer(self.phi.__class__.__name__)
            self.layers.apply(init)
            head_init = get_initializer('linear')
            self.heads.apply(head_init)

        if 'pwi' in self.init:
            with torch.no_grad():
                # principled weight-init (assuming bias=True)
                last_fan_in = None
                for head, (name, par) in zip(self.heads, self.main_architecture):
                    last_layer = head.layer_norm if len(head) > 1 else head.W_p
                    if name.split('.')[-1] == 'weight':   # if bias in main layer
                        last_layer.weight /= (2. * par.shape[1]) ** .5
                        last_layer.bias /= 2. ** .5
                        last_fan_in = par.shape[1]
                    else:
                        last_layer.weight /= (2. * last_fan_in) ** .5
                        last_layer.bias /= 2. ** .5

    def forward(self, x, memory=None):
        batch_size = x.shape[0]

        if self.context:
            x = self.protein_context(x, memory)
            if self.layerNorm:
                x = self.layerNorm(x)

        for layer in self.layers:
            x = layer(x)

        state_dicts = [OrderedDict({}) for _ in range(batch_size)]
        for i, head in enumerate(self.heads):
            param_name, param_values = self.main_architecture[i]
            pred_params = head(x)
            param_size = param_values.size()
            for batch in range(batch_size):
                state_dicts[batch][param_name] = pred_params[batch, :].reshape(param_size)

        return state_dicts


class HyperPCM(nn.Module):
    """
    HyperPCM, full task-specific adaption of models for drug-target interaction prediction.
    Author: Emma Svensson
    """

    def __init__(self, molecule_encoder, protein_encoder, args, memory=None):
        super(HyperPCM, self).__init__()
        assert not (args['hopfield']['context_module'] and memory is None), \
            'A context is required for the context module.'
        self.memory = memory

        self.main = MainFCN(molecule_encoder, args['main_cls']).requires_grad_(False)
        num_params = sum([param.nelement() for param in self.main.parameters()])
        print(f'HyperNet needs to predict {num_params} number of parameters for the main network.')

        self.hyper = HyperFCN(
            protein_encoder,
            main_architecture=self.main.state_dict(),
            fcn_args=args['hyper_fcn'],
            hopfield_args=args['hopfield']
        )
        num_hyper_params = sum([param.nelement() for param in self.hyper.parameters()])
        print(f'The HyperNet itself has {num_hyper_params} number of parameters.')

    def forward(self, batch):
        device = next(self.hyper.parameters()).device
        protein_batch = batch['proteins'].to(device)

        if self.memory is not None:
            memory = self.memory.get_protein_memory(exclude_pids=batch['pids'] if self.train else []).to(device)
        else:
            memory = None

        # Hyper network parameter prediction
        state_dicts = self.hyper(protein_batch, memory)

        outputs = torch.Tensor([]).to(device)
        # Run main sequentially
        for i, state_dict in enumerate(state_dicts):
            protein_output = self.forward_main(batch['molecules'][i].to(device), state_dict)
            if len(batch['molecules'][i]) < 2:
                protein_output = protein_output.unsqueeze(0)
            outputs = torch.cat((outputs, protein_output), 0)

        return outputs  # , state_dicts

    def forward_main(self, molecules, state_dict):
        pred_params = list(state_dict.items())
        l = 0
        for layer in self.main.modules():
            if isinstance(layer, nn.Linear):
                for param, values in layer.named_parameters():
                    layer._parameters[param] = pred_params[l][1]
                    l += 1

        return self.main(molecules)

