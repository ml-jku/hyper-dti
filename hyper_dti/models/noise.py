
import torch
import torch.nn as nn


class EncoderHead(nn.Module):
    """
    Re-implementation of EncoderHead with added noise from DeepPCM model (Kim, P. T., et al., 2021),
    https://www.mdpi.com/1422-0067/22/23/12882.
    Author: Emma Svensson
    """

    def __init__(self, embedding_dim):
        super(EncoderHead, self).__init__()

        inplace = False
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=inplace),
            nn.Dropout(p=0.2, inplace=inplace)
        )

    def forward(self, emb):
        device = next(self.linear.parameters()).device
        noise = torch.normal(mean=0, std=0.01, size=emb.shape).to(device)
        x = torch.add(emb, noise)
        x = self.linear(x)

        return torch.cat((emb, x), dim=1)

