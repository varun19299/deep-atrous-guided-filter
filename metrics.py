"""
Metrics file
"""
import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def PSNR(output: "Tensor[N,C,H,W]", target: "Tensor[N,C,H,W]") -> "Tensor[0]":
    """
    Peak Signal to noise ratio

    Ref: https://www.mathworks.com/help/vision/ref/psnr.html

    Images between [-1,1]
    """
    output = output.mul(0.5).add(0.5).clamp(0, 1)
    target = target.mul(0.5).add(0.5).clamp(0, 1)
    noise = ((output - target) ** 2).mean(dim=3).mean(dim=2).mean(dim=1)
    signal_max = 1.0
    return (10 * torch.log10(signal_max ** 2 / noise)).mean()


def PSNR_numpy(output: "Array[H,W,C]", target: "Array[H,W,C]") -> float:
    """
    :param output: H x W x C
    :param target: H x W x C
    :return:
    """
    output_numpy_int8 = (output * 255.0).astype(np.uint8) / 255.0
    target_numpy_int8 = (target * 255.0).astype(np.uint8) / 255.0
    return 10 * np.log10(1 / ((output_numpy_int8 - target_numpy_int8) ** 2).mean())
