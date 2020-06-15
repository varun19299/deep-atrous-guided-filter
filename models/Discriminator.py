from sacred import Experiment
from typing import TYPE_CHECKING

import torch
from functools import partial
from torch import nn

from models.spectral import SpectralNorm

from config import initialise

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("Disc")
ex = initialise(ex)


def group_norm(num_channels: int, args: "tupperware"):
    return nn.GroupNorm(num_channels=num_channels, num_groups=args.num_groups)


def get_normaliser(args: "tupperware"):
    """
    Batch norm or Group norm
    """
    if args.normaliser == "batch_norm":
        return nn.BatchNorm2d
    elif args.normaliser == "instance_norm":
        return nn.InstanceNorm2d
    elif args.normaliser == "group_norm":
        return partial(group_norm, args=args)
    elif args.normaliser == "layer_norm":
        return nn.LayerNorm


def get_spectral_normaliser(func, args: "tupperware"):
    if args.use_spectral_norm:
        return SpectralNorm(func)
    else:
        return func


class Discriminator(nn.Module):
    def __init__(self, args, source_device=None, target_device=None, use_pool=False):
        super(Discriminator, self).__init__()

        self.args = args

        normaliser = get_normaliser(args)

        spectral_normaliser = partial(get_spectral_normaliser, args=args)

        self.disc = nn.Sequential(
            spectral_normaliser(nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_normaliser(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            normaliser(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            spectral_normaliser(nn.Conv2d(128, 1, kernel_size=1)),
        )

        self.source_device = source_device
        self.target_device = target_device

        assert not (self.source_device and self.target_device) or (
            self.source_device and self.target_device
        ), "XOR of source and target device"

    def forward(self, img):
        if self.source_device:
            img = img.to(self.source_device)

        x = self.disc(img)
        return x.squeeze()


@ex.automain
def main(_run):
    from utils.tupperware import tupperware
    from torchsummary import summary

    args = tupperware(_run.config)
    D = Discriminator(args)
    summary(D, (3, 1024, 2048))
