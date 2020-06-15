import torch
from typing import TYPE_CHECKING
from sacred import Experiment
import numpy as np
import torch.nn as nn

from config import initialise
from utils.tupperware import tupperware

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("FFT-Layer")
ex = initialise(ex)


def fft_conv2d(input, kernel):
    """
    Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain!
    :param input: shape (B, Cin, H, W)
    :param kernel: shape (Cout, Cin, H, W)
    :param bias: shape of (B, Cout, H, W)
    :return:
    """
    input = torch.rfft(input, 2, onesided=False)
    kernel = torch.rfft(kernel, 2, onesided=False)

    # Compute the multiplication
    # (a+bj)*(c+dj) = (ac-bd)+(ad+bc)j
    real = input[..., 0] * kernel[..., 0] - input[..., 1] * kernel[..., 1]
    im = input[..., 0] * kernel[..., 1] + input[..., 1] * kernel[..., 0]

    # Stack both channels and sum-reduce the input channels dimension
    out = torch.stack([real, im], -1)

    out = torch.irfft(out, 2, onesided=False)
    return out


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def get_wiener_matrix(psf, Gamma: int = 20000, centre_roll: bool = True):
    """
    Get PSF matrix
    :param psf:
    :param gamma_exp:
    :return:
    """

    # Gamma = 10 ** (-0.1 * gamma_exp)

    if centre_roll:
        for dim in range(2):
            psf = roll_n(psf, axis=dim, n=psf.size(dim) // 2)

    psf = psf.unsqueeze(0)

    H = torch.rfft(psf, 2, onesided=False)
    Habsq = H[:, :, :, 0].pow(2) + H[:, :, :, 1].pow(2)

    W_0 = (torch.div(H[:, :, :, 0], (Habsq + Gamma))).unsqueeze(-1)
    W_1 = (-torch.div(H[:, :, :, 1], (Habsq + Gamma))).unsqueeze(-1)
    W = torch.cat((W_0, W_1), -1)

    weiner_mat = torch.irfft(W, 2, onesided=False)

    return weiner_mat[0]


class FFTLayer(nn.Module):
    def __init__(self, args: "tupperware"):
        super().__init__()
        self.args = args

        # No grad if you're not training this layer
        requires_grad = not (args.fft_epochs == args.num_epochs)

        wiener = (
            torch.tensor(np.load(args.wiener_mat)).float().unsqueeze(0).unsqueeze(0)
        )

        self.fft_layer = nn.Parameter(wiener, requires_grad=requires_grad)

        self.normalizer = nn.Parameter(
            torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1), requires_grad=requires_grad
        )

    def forward(self, img):
        # Convert to 0...1
        img = 0.5 * img + 0.5

        # Do FFT convolve
        img = fft_conv2d(img, self.fft_layer) * self.normalizer

        return img


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFTLayer(args).to(device)
    img = torch.rand(1, 3, 1024, 2048).to(device)
    model(img)
