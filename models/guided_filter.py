import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.ops import unpixel_shuffle

from models.lr_net import SmoothDilatedResidualAtrousGuidedBlock, LRNet
from models.model_utils import AdaptiveInstanceNorm


from sacred import Experiment
from config import initialise

ex = Experiment("DAGF")
ex = initialise(ex)


class ConvGuidedFilter(nn.Module):
    """
    Adapted from https://github.com/wuhuikai/DeepGuidedFilter
    """
    def __init__(self, radius=1, norm=nn.BatchNorm2d, conv_a_kernel_size: int = 1):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(
            3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(
                6,
                32,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                3,
                kernel_size=conv_a_kernel_size,
                padding=conv_a_kernel_size // 2,
                bias=False,
            ),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b


class DeepAtrousGuidedFilter(nn.Module):
    def __init__(self, args, radius=1):
        super().__init__()

        self.args = args
        norm = AdaptiveInstanceNorm

        c = args.guided_map_channels
        self.guided_map = SmoothDilatedResidualAtrousGuidedBlock(
            in_channel=3, channel_num=c, args=args
        )

        self.lr = LRNet(
            in_c=3 * args.pixelshuffle_ratio ** 2,
            out_c=3 * args.pixelshuffle_ratio ** 2,
            args=args,
        )

        self.gf = ConvGuidedFilter(radius, norm=norm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)

        # Unpixelshuffle
        x_lr_unpixelshuffled = unpixel_shuffle(x_lr, self.args.pixelshuffle_ratio)

        # Pixelshuffle
        y_lr = F.pixel_shuffle(
            self.lr(x_lr_unpixelshuffled), self.args.pixelshuffle_ratio
        )

        return F.tanh(
            self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr))
        )


@ex.automain
def main(_run):
    from utils.tupperware import tupperware
    from torchsummary import summary

    args = tupperware(_run.config)
    model = DeepAtrousGuidedFilter(args).to(args.device)
    summary(model, (3, 1024, 2048))
