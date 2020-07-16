"""
Metrics file
"""
import torch
from typing import TYPE_CHECKING
from sacred import Experiment

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("metrics")


def PSNR(source: "Tensor", target: "Tensor"):
    """
    Peak Signal to noise ratio

    Ref: https://www.mathworks.com/help/vision/ref/psnr.html

    Images between [-1,1]
    """
    source = source.mul(0.5).add(0.5).clamp(0, 1)
    target = target.mul(0.5).add(0.5).clamp(0, 1)
    noise = ((source - target) ** 2).mean(dim=3).mean(dim=2).mean(dim=1)
    signal_max = 1.0
    return (10 * torch.log10(signal_max ** 2 / noise)).mean()


def PSNR_quant(source: "Tensor", target: "Tensor"):
    """
    Peak Signal to noise ratio

    Ref: https://www.mathworks.com/help/vision/ref/psnr.html

    Images between [-1,1]
    """
    source = source.mul(0.5).add(0.5).clamp(0, 1)
    target = target.mul(0.5).add(0.5).clamp(0, 1)

    source = (source * 255.0).int()
    target = (target * 255.0).int()

    noise = ((source - target) ** 2).float().mean(dim=3).mean(dim=2).mean(dim=1)
    signal_max = 255.0
    return (10 * torch.log10(signal_max ** 2 / noise)).mean().item()
