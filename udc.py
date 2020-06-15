"""
Idea:

Estimate PSF by dividing HQ by LQ in fourier domain.

Then use it on same image, new image.
"""

import torch
from pathlib import Path
from sacred import Experiment
from models.fftlayer import roll_n, fft_conv2d
import cv2
from matplotlib import pyplot as plt
import numpy as np

ex = Experiment("udc")


@ex.config
def config():
    udc_path = Path("/Users/Ankivarun/Downloads/udc/Poled")

    file_name = "1.png"
    HQ_path = udc_path / "HQ" / file_name
    HQ = cv2.imread(str(HQ_path))[:, :, ::-1] / 255.0
    HQ = torch.tensor(HQ).float()

    LQ_path = udc_path / "LQ" / file_name
    LQ = cv2.imread(str(LQ_path))[:, :, ::-1] / 255.0
    LQ = torch.tensor(LQ).float()


def get_wiener_matrix(HQ, LQ, Gamma: int = 20000, centre_roll: bool = True):
    """
    Get PSF matrix
    :param psf:
    :param gamma_exp:
    :return:
    """
    if centre_roll:
        for dim in range(2):
            HQ = roll_n(HQ, axis=dim, n=HQ.size(dim) // 2)

        for dim in range(2):
            LQ = roll_n(LQ, axis=dim, n=LQ.size(dim) // 2)

    HQ = HQ.unsqueeze(0)
    LQ = LQ.unsqueeze(0)

    HQ_fft = torch.rfft(HQ, 2, onesided=False)
    LQ_fft = torch.rfft(LQ, 2, onesided=False)

    HQ_abs = HQ_fft[:, :, :, 0].pow(2) + HQ_fft[:, :, :, 1].pow(2)
    LQ_abs = LQ_fft[:, :, :, 0].pow(2) + LQ_fft[:, :, :, 1].pow(2)

    W_0 = (
        torch.div(
            HQ_fft[:, :, :, 0] * LQ_fft[:, :, :, 0]
            + HQ_fft[:, :, :, 1] * LQ_fft[:, :, :, 1],
            (HQ_abs + Gamma * LQ_abs),
        )
    ).unsqueeze(-1)

    W_1 = (
        torch.div(
            HQ_fft[:, :, :, 1] * LQ_fft[:, :, :, 0]
            - HQ_fft[:, :, :, 0] * LQ_fft[:, :, :, 1],
            (HQ_abs + Gamma * LQ_abs),
        )
    ).unsqueeze(-1)

    W = torch.cat((W_0, W_1), -1)

    weiner_mat = torch.irfft(W, 2, onesided=False)

    return weiner_mat[0]


@ex.automain
def main(HQ, LQ, udc_path):
    # Display LQ
    plt.imshow(LQ)
    plt.show()

    # Display HQ
    plt.imshow(HQ)
    plt.show()

    wiener = []
    for i in range(3):
        wiener_mat = get_wiener_matrix(
            HQ[:, :, i], LQ[:, :, i], Gamma=1000, centre_roll=False
        )
        # for dim in range(2):
        #     wiener_mat = roll_n(wiener_mat, axis=dim, n=wiener_mat.size(dim) // 2)

        wiener.append(wiener_mat)

    # Display wiener
    plt.imshow(wiener[0])
    plt.show()
    np.save("poled_wiener_R.npy", wiener[0])
    np.save("data/poled_wiener.npy", torch.stack(wiener))

    LQ = LQ.permute(2, 0, 1)
    HQ_sim = []

    for i in range(3):
        HQ_sim.append(fft_conv2d(LQ[i], wiener[0]))

    HQ_sim = torch.stack(HQ_sim)

    # Display HQ sim
    breakpoint()
    HQ_sim = (HQ_sim - HQ_sim.min()) / (HQ_sim.max() - HQ_sim.min())
    HQ_sim = HQ_sim.permute(1, 2, 0)

    # cv2.imwrite(str(udc_path / "1.png"), HQ_sim.numpy()[:, :, ::-1] * 255.0)
    plt.imshow(HQ_sim)
    plt.show()
