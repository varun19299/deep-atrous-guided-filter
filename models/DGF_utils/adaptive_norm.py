import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveInstanceNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.ins_norm = nn.InstanceNorm2d(n, momentum=0.999, eps=0.001, affine=True)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.ins_norm(x)


norm_dict = {
    "adaptive-instance": AdaptiveInstanceNorm,
    "adaptive-batch": AdaptiveNorm,
    "instance": nn.InstanceNorm2d,
    "batch": nn.BatchNorm2d,
    "group": partial(F.group_norm, num_groups=8),
}
