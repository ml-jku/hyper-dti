
import torch
import torch.nn as nn
from collections import OrderedDict

from hyper_dti.settings import constants
from hyper_dti.models.noise import EncoderHead
from hyper_dti.models.context import ContextModule
from hyper_dti.models.mlp import Classifier
from hyper_dti.models.utils.initialization import get_initializer


class MainFCN(nn.Module):
    """
    QSAR module for drug inputs, predicting the drug-target interactions.
    Author: Emma Svensson
    """

    def __init__(self, drug_encoder, cls_args):
        super(MainFCN, self).__init__()

        self.noise = cls_args['noise']
        in_channels = constants.DRUG_LATENT_DIM[drug_encoder]
        if self.noise:
            self.drug_encoder = EncoderHead(constants.DRUG_LATENT_DIM[drug_encoder])
            in_channels += 512

        hidden_layers = [in_channels]
        for i in range(cls_args['layers']):
            hidden_layers.append(int(cls_args['hidden_dim'] / (2 ** (i + 1))))
        self.classifier = Classifier(hidden_layers=hidden_layers)

    def forward(self, x):
        if self.noise:
            x = self.drug_encoder(x)
        x = self.classifier(x)
        return x.squeeze()


class HyperFCN(nn.Module):
    """
    HyperNetwork for target inputs, predicting the parameters of the QSAR model.
    Author: Emma Svensson
    """

    def __init__(self, target_encoder, main_architecture, fcn_args, hopfield_args):
        super(HyperFCN, self).__init__()

        in_channels = constants.TARGET_LATENT_DIM[target_encoder]

        self.context = hopfield_args['context_module']
        if self.context:
            self.target_context = ContextModule(in_channels=in_channels, args=hopfield_args)

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
            x = self.target_context(x, memory)

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
    HyperPCM, full task-conditioned prediction of parameters for a QSAR models which in term
    predicts drug-target interactions.
    Author: Emma Svensson
    """

    def __init__(self, drug_encoder, target_encoder, args, memory=None):
        super(HyperPCM, self).__init__()
        assert not (args['hopfield']['context_module'] and memory is None), \
            'A context is required for the context module.'
        self.memory = memory

        self.main = MainFCN(drug_encoder, args['main_cls']).requires_grad_(False)
        num_params = sum([param.nelement() for param in self.main.parameters()])
        print(f'HyperNet needs to predict {num_params} number of parameters for the QSAR model.')

        self.hyper = HyperFCN(
            target_encoder,
            main_architecture=self.main.state_dict(),
            fcn_args=args['hyper_fcn'],
            hopfield_args=args['hopfield']
        )
        num_hyper_params = sum([param.nelement() for param in self.hyper.parameters()])
        print(f'The HyperNet itself has {num_hyper_params} number of parameters.')

        param_size = 0
        buffer_size = 0
        for model in [self.main, self.hyper]:
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Total HyperPCM model size: {:.3f}MB'.format(size_all_mb))

    def forward(self, batch):
        device = next(self.hyper.parameters()).device
        target_batch = batch['targets'].to(device)

        if self.memory is not None:
            memory = self.memory.get_target_memory(exclude_pids=batch['pids']).to(device)
        else:
            memory = None

        # Hyper network parameter prediction
        state_dicts = self.hyper(target_batch, memory)

        outputs = torch.Tensor([]).to(device)
        # Run QSAR models sequentially
        for i, state_dict in enumerate(state_dicts):
            target_output = self.forward_main(batch['drugs'][i].to(device), state_dict)
            if len(batch['drugs'][i]) < 2:
                target_output = target_output.unsqueeze(0)
            outputs = torch.cat((outputs, target_output), 0)

        return outputs  # , state_dicts

    def forward_main(self, drugs, state_dict):
        pred_params = list(state_dict.items())
        l = 0
        for layer in self.main.modules():
            if isinstance(layer, nn.Linear):
                for param, values in layer.named_parameters():
                    layer._parameters[param] = pred_params[l][1]
                    l += 1

        return self.main(drugs)

