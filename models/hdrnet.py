import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import TYPE_CHECKING

from sacred import Experiment
from config import initialise
from utils.tupperware import tupperware

ex = Experiment("HDRNet")
ex = initialise(ex)

if TYPE_CHECKING:
    from utils.typing_alias import *


class ConvBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        use_bias=True,
        activation=nn.ReLU(inplace=True),
        is_BN=False,
    ):
        super(ConvBlock, self).__init__()
        if is_BN:
            self.conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                inc,
                                outc,
                                kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=use_bias,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(outc)),
                        ("act", activation),
                    ]
                )
            )
        elif activation is not None:
            self.conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                inc,
                                outc,
                                kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=use_bias,
                            ),
                        ),
                        ("act", activation),
                    ]
                )
            )
        else:
            self.conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                inc,
                                outc,
                                kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=use_bias,
                            ),
                        )
                    ]
                )
            )

    def forward(self, input):
        return self.conv(input)


class fc(nn.Module):
    def __init__(self, inc, outc, activation=None, is_BN=False):
        super(fc, self).__init__()
        if is_BN:
            self.fc = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(inc, outc)),
                        ("bn", nn.BatchNorm1d(outc)),
                        ("act", activation),
                    ]
                )
            )
        elif activation is not None:
            self.fc = nn.Sequential(
                OrderedDict([("fc", nn.Linear(inc, outc)), ("act", activation)])
            )
        else:
            self.fc = nn.Sequential(OrderedDict([("fc", nn.Linear(inc, outc))]))

    def forward(self, input):
        return self.fc(input)


class Guide(nn.Module):
    """
    pointwise neural net
    """

    def __init__(self, mode="PointwiseNN"):
        super(Guide, self).__init__()
        if mode == "PointwiseNN":
            self.mode = "PointwiseNN"
            self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, is_BN=True)
            self.conv2 = ConvBlock(
                16, 1, kernel_size=1, padding=0, activation=nn.Tanh()
            )

    def forward(self, x):
        if self.mode == "PointwiseNN":
            guidemap = self.conv2(self.conv1(x))

        return guidemap


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = guidemap.device
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3).to(device) / (H - 1) * 2 - 1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3).to(device) / (W - 1) * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([guidemap, hg, wg], dim=3).unsqueeze(1)

        coeff = F.grid_sample(bilateral_grid, guidemap_guide)

        return coeff.squeeze(2)


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, coeff, full_res_input):
        R = (
            torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True)
            + coeff[:, 3:4, :, :]
        )
        G = (
            torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True)
            + coeff[:, 7:8, :, :]
        )
        B = (
            torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True)
            + coeff[:, 11:12, :, :]
        )

        return torch.cat([R, G, B], dim=1)


class HDRNet(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(HDRNet, self).__init__()
        self.inc = inc
        self.outc = outc

        self.downsample = nn.Upsample(
            size=(256, 512), mode="bilinear", align_corners=True
        )
        self.activation = nn.ReLU(inplace=True)

        # -----------------------------------------------------------------------
        splat_layers = []
        for i in range(4):
            if i == 0:
                splat_layers.append(
                    ConvBlock(
                        self.inc,
                        (2 ** i) * 8,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        activation=self.activation,
                        is_BN=False,
                    )
                )
            else:
                splat_layers.append(
                    ConvBlock(
                        (2 ** (i - 1) * 8),
                        (2 ** (i)) * 8,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        activation=self.activation,
                        is_BN=True,
                    )
                )

        self.splat_conv = nn.Sequential(*splat_layers)

        # -----------------------------------------------------------------------
        global_conv_layers = [
            ConvBlock(64, 64, stride=2, activation=self.activation, is_BN=True),
            ConvBlock(64, 64, stride=2, activation=self.activation, is_BN=True),
        ]
        self.global_conv = nn.Sequential(*global_conv_layers)

        global_fc_layers = [
            fc(1024 * 2, 256, activation=self.activation, is_BN=True),
            fc(256, 128, activation=self.activation, is_BN=True),
            fc(128, 64),
        ]
        self.global_fc = nn.Sequential(*global_fc_layers)

        # -----------------------------------------------------------------------
        local_layers = [
            ConvBlock(64, 64, activation=self.activation, is_BN=True),
            ConvBlock(64, 64, use_bias=False, activation=None, is_BN=False),
        ]
        self.local_conv = nn.Sequential(*local_layers)

        # -----------------------------------------------------------------------
        self.linear = nn.Conv2d(64, 96, kernel_size=1)

        self.guide_func = Guide()
        self.slice_func = Slice()
        self.transform_func = Transform()

    def forward(self, full_res_input: "Tensor"):
        # Transform 4 to 3 channel

        r_full_res_input = full_res_input[:, 0].unsqueeze(1)
        g_full_res_input = full_res_input[:, 1].unsqueeze(1)
        b_full_res_input = full_res_input[:, 2].unsqueeze(1)
        full_res_input = torch.cat(
            [r_full_res_input, g_full_res_input, b_full_res_input], dim=1
        )

        low_res_input = self.downsample(full_res_input)
        bs, _, _, _ = low_res_input.size()

        splat_fea = self.splat_conv(low_res_input)

        local_fea = self.local_conv(splat_fea)

        global_fea = self.global_conv(splat_fea)
        global_fea = self.global_fc(global_fea.view(bs, -1))

        fused = self.activation(global_fea.view(-1, 64, 1, 1) + local_fea)
        fused = self.linear(fused)

        bilateral_grid = fused.view(-1, 12, 8, 16, 16 * 2)

        guidemap = self.guide_func(full_res_input)
        coeff = self.slice_func(bilateral_grid, guidemap)
        output = self.transform_func(coeff, full_res_input)
        output = F.tanh(output)

        return output


@ex.automain
def main(_run):
    from torchsummary import summary

    args = tupperware(_run.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = HDRNet().to(device)
    summary(net, input_size=(3, 1024, 2048))
