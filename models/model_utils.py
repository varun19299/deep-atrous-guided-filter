import torch
import torch.nn as nn


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveInstanceNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.ins_norm = nn.InstanceNorm2d(n, momentum=0.999, eps=0.001, affine=True)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.ins_norm(x)


class PALayer(nn.Module):
    def __init__(self, channel: int):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel: int):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
