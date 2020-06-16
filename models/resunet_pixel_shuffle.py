import torch
import torch.nn.functional as F
from torchvision import models
from typing import TYPE_CHECKING
from sacred import Experiment
from functools import partial

import torch.nn as nn

from config import initialise
from utils.tupperware import tupperware
from utils.ops import unpixel_shuffle

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("ResUnet")
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


def convrelu(
    in_channels, out_channels, kernel=3, padding=1, stride=1, normaliser=nn.BatchNorm2d
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, stride=stride),
        normaliser(out_channels),
        nn.ReLU(inplace=True),
    )


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, normaliser):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
        )
        self.upsample = nn.PixelShuffle(up_scale)
        self.residual = ResidualBlock(in_channels, normaliser)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.residual(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, channels, normaliser):
        super(DownSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels * 2, kernel_size=3, padding=1, stride=2
        )
        self.bn1 = normaliser(channels * 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.bn2 = normaliser(channels * 2)

        self.identity = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, stride=2),
            normaliser(channels * 2),
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return self.identity(x) + residual


class ResidualBlock(nn.Module):
    def __init__(self, channels, normaliser):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = normaliser(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = normaliser(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class ResUnet(nn.Module):
    def __init__(self, args: "tupperware"):
        super().__init__()

        self.args = args
        normaliser = get_normaliser(args)

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                3 * args.pixelshuffle_ratio ** 2,
                128,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            normaliser(128),
            nn.ReLU(inplace=True),
        )  # size=(N, 128, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(128, 128, 1, 0, normaliser=normaliser)

        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            ResidualBlock(128, normaliser=normaliser),
        )  # size=(N, 128, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(128, 128, 1, 0, normaliser=normaliser)

        self.layer2 = nn.Sequential(
            DownSampleBlock(128, normaliser=normaliser)
        )  # size=(N, 256, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(256, 256, 1, 0, normaliser=normaliser)

        self.layer3 = nn.Sequential(
            DownSampleBlock(256, normaliser=normaliser)
        )  # size=(N, 512, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(512, 512, 1, 0, normaliser=normaliser)

        self.layer4 = nn.Sequential(
            DownSampleBlock(512, normaliser=normaliser)
        )  # size=(N, 1024, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(1024, 512, 1, 0, normaliser=normaliser)

        self.conv_up3 = nn.Sequential(
            convrelu(512 + 512, 512, 3, 1, normaliser=normaliser),
            convrelu(512, 512, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up2 = nn.Sequential(
            convrelu(256 + 512, 256, 3, 1, normaliser=normaliser),
            convrelu(256, 256, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up1 = nn.Sequential(
            convrelu(128 + 256, 128, 3, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up0 = nn.Sequential(
            convrelu(128 + 128, 128, 3, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
        )

        self.upsample_8_16 = UpsampleBLock(512, 2, normaliser=normaliser)
        self.upsample_16_32 = UpsampleBLock(512, 2, normaliser=normaliser)
        self.upsample_32_64 = UpsampleBLock(256, 2, normaliser=normaliser)
        self.upsample_64_128 = UpsampleBLock(128, 2, normaliser=normaliser)
        self.upsample_128_256 = UpsampleBLock(128, 2, normaliser=normaliser)

        self.conv_original_size0 = convrelu(
            3 * args.pixelshuffle_ratio ** 2, 128, 3, 1, normaliser=normaliser
        )
        self.conv_original_size1 = convrelu(128, 128, 3, 1, normaliser=normaliser)
        self.conv_original_size2 = convrelu(128 + 128, 128, 3, 1, normaliser=normaliser)

        self.get_image = nn.Sequential(
            nn.Conv2d(128, 3 * args.pixelshuffle_ratio ** 2, 1), nn.ReLU()
        )
        self.pixelshuffle_ratio = args.pixelshuffle_ratio

    def forward(self, img):
        # Unpixelshuffle, make smaller
        img = unpixel_shuffle(img, self.pixelshuffle_ratio)

        x_original = self.conv_original_size0(img)
        x_original = self.conv_original_size1(x_original)  # 128 channels

        layer0 = self.layer0(img)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # 1024 channels

        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)

        x = self.upsample_8_16(layer4)  # size=(N, 1024, x.H/16, x.W/16)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)  # size=(N, 1024, x.H/16, x.W/16)

        layer2 = self.layer2_1x1(layer2)
        x = self.upsample_16_32(x)  # size=(N, 1024, x.H/8, x.W/8)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)  # size=(N, 512, x.H/8, x.W/8)

        layer1 = self.layer1_1x1(layer1)
        x = self.upsample_32_64(x)  # size=(N, 512, x.H/4, x.W/4)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)  # size=(N, 512, x.H/4, x.W/4)

        layer0 = self.layer0_1x1(layer0)
        x = self.upsample_64_128(x)  # size=(N, 512, x.H/2, x.W/2)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # size=(N, 256, x.H/2, x.W/2)

        x = self.upsample_128_256(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # size=(N, 128, x.H, x.W)
        x = self.get_image(x)

        # PixelShuffle, make bigger
        img_shuffle = F.pixel_shuffle(x, self.pixelshuffle_ratio)

        return img_shuffle


@ex.automain
def main(_run):
    from torchsummary import summary

    args = tupperware(_run.config)
    model = ResUnet(args)
    model = model.to(args.device)

    summary(model, (3, 256, 512))
