import torch
import torch.nn as nn


def get_initializer(non_linearity: str = "relu"):
    non_linearity = non_linearity.lower()
    if non_linearity == "selu":
        # fix pytorch gain for SNN
        non_linearity = "linear"

    def initializer(m):
        if isinstance(m, nn.Linear):
            if non_linearity == 'relu':
                nn.init.kaiming_normal_(m.weight, nonlinearity=non_linearity)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0])
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList) or isinstance(m, nn.ModuleDict):
            return
        elif sum(p.numel() for p in m.parameters()) > 0:
            raise ValueError(f"no initialization for '{m.__class__.__name__}' module")

    return initializer

