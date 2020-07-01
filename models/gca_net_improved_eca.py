"""
Code derived from:

https://github.com/cddlyf/GCANet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import initialise
from sacred import Experiment

ex = Experiment("GACNet")

ex = initialise(ex)

from models.DGF_utils.weights_init import weights_init_identity_pixelshuffle
from models.DGF_utils.adaptive_norm import AdaptiveInstanceNorm
from models.FFA_utils import ECALayer, PALayer


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel size should be odd"
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(
            inc, 1, self.kernel_size, self.kernel_size
        ).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)


class SmoothDilatedResidualFFABlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super().__init__()
        self.pre_conv1 = ShareSepConv(dilation * 2 - 1)
        self.conv1 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm1 = AdaptiveInstanceNorm(channel_num)
        self.pre_conv2 = ShareSepConv(dilation - 1)
        self.conv2 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm2 = AdaptiveInstanceNorm(channel_num)

        self.ecalayer = ECALayer(channel_num)
        self.palayer = PALayer(channel_num)

    def forward(self, x):
        y = F.leaky_relu(self.norm1(self.conv1(self.pre_conv1(x))), 0.2)
        y = self.norm2(self.conv2(self.pre_conv2(y + x)))
        y = self.palayer(self.ecalayer(y))
        return F.leaky_relu(x + y, 0.2)


class ResidualFFABlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm1 = AdaptiveInstanceNorm(channel_num)
        self.conv2 = nn.Conv2d(
            channel_num,
            channel_num,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            groups=group,
            bias=False,
        )
        self.norm2 = AdaptiveInstanceNorm(channel_num)

        self.ecalayer = ECALayer(channel_num)
        self.palayer = PALayer(channel_num)

    def forward(self, x):
        y = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        y = self.norm2(self.conv2(y + x))
        y = self.palayer(self.ecalayer(y))
        return F.leaky_relu(x + y, 0.2)


class GCAECANet_improved(nn.Module):
    def __init__(self, in_c=4, out_c=3):
        super(GCAECANet_improved, self).__init__()

        interm_channels = 48
        residual_adds = 3

        smooth_dialated_block = SmoothDilatedResidualFFABlock
        residual_block = ResidualFFABlock

        self.conv1 = nn.Conv2d(in_c, interm_channels, 3, 1, 1, bias=False)
        self.norm1 = AdaptiveInstanceNorm(interm_channels)

        self.res1 = smooth_dialated_block(interm_channels, dilation=2 ** 1)

        self.res2_a = smooth_dialated_block(interm_channels, dilation=2 ** 2)
        self.res2_b = smooth_dialated_block(interm_channels, dilation=2 ** 2)

        self.res3_a = smooth_dialated_block(interm_channels, dilation=2 ** 3)
        self.res3_b = smooth_dialated_block(interm_channels, dilation=2 ** 3)

        self.res4_a = smooth_dialated_block(interm_channels, dilation=2 ** 4)
        self.res4_b = smooth_dialated_block(interm_channels, dilation=2 ** 4)

        self.res5_a = smooth_dialated_block(interm_channels, dilation=2 ** 5)
        self.res5_b = smooth_dialated_block(interm_channels, dilation=2 ** 5)

        self.res_final = residual_block(interm_channels, dilation=1)

        self.gate = nn.Conv2d(
            interm_channels * residual_adds, residual_adds, 3, 1, 1, bias=True
        )

        # self.ecalayer = ECALayer(interm_channels)
        # self.palayer = PALayer(interm_channels)

        self.deconv2 = nn.Conv2d(interm_channels, interm_channels, 3, 1, 1)
        self.norm5 = AdaptiveInstanceNorm(interm_channels)
        self.deconv1 = nn.Conv2d(interm_channels, out_c, 1)

        self.apply(weights_init_identity_pixelshuffle)

    def forward(self, x):
        y1 = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)

        y = self.res1(y1)
        y = self.res2_a(y)
        y = self.res2_b(y)
        y = self.res3_a(y)
        y2 = self.res3_b(y)

        y = self.res4_a(y2)
        y = self.res4_b(y)
        y = self.res5_a(y)
        y = self.res5_b(y)
        y3 = self.res_final(y)

        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        gated_y = (
            y1 * gates[:, [0], :, :]
            + y2 * gates[:, [1], :, :]
            + y3 * gates[:, [2], :, :]
        )

        # gated_y = self.palayer(self.ecalayer(gated_y))
        y = F.leaky_relu(self.norm5(self.deconv2(gated_y)), 0.2)
        y = F.leaky_relu(self.deconv1(y), 0.2)

        return y


@ex.automain
def main(_run):
    from utils.tupperware import tupperware
    from torchsummary import summary

    args = tupperware(_run.config)

    model = GCAECANet_improved(in_c=12, out_c=12).to(args.device)

    summary(model, (12, 256, 512))
