import torch.nn as nn

import constants


class Classifier(nn.Module):
    """
    General implementation of simple multi-layer perceptron for single prediction.
    Author: Emma Svensson
    """

    def __init__(self, hidden_layers):
        super(Classifier, self).__init__()

        inplace = False
        self.layers = nn.ModuleList()
        for layer in range(len(hidden_layers)-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_layers[layer], hidden_layers[layer+1]),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(p=0.25, inplace=inplace)
                )
            )

        self.layers.append(
            nn.Sequential(
                nn.Linear(hidden_layers[-1], 1)
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

