"""
Metrics file
"""
import torch
import numpy as np
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

    noise = ((source - target) ** 2).double().mean(dim=3).mean(dim=2).mean(dim=1)
    signal_max = 255.0
    return (10 * torch.log10(signal_max ** 2 / noise)).mean().float().item()


def PSNR_numpy(source, target):
    """
    :param source: H x W x C
    :param target: H x W x C
    :return:
    """
    source_numpy_int8 = (source * 255.0).astype(np.uint8) / 255.0
    target_numpy_int8 = (target * 255.0).astype(np.uint8) / 255.0
    return 10 * np.log10(1 / ((source_numpy_int8 - target_numpy_int8) ** 2).mean())
