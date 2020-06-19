import torch
from torch import nn


class SIREN(nn.Module):
    """
From https://arxiv.org/pdf/2006.09661.pdf
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return torch.sin(input)  # simply apply already implemented SiLU
