"""
Metrics file
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3

from typing import TYPE_CHECKING
import logging
from scipy.stats import entropy
from scipy import linalg
from sacred import Experiment
import numpy as np

from models.inception import InceptionV3

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("metrics")


def PSNR(source: "Tensor", target: "Tensor"):
    """
    Peak Signal to noise ratio

    Ref: https://www.mathworks.com/help/vision/ref/psnr.html

    Images between [-1,1]
    """
    source = source.mul(0.5).add(0.5).clamp(0,1)
    target = target.mul(0.5).add(0.5).clamp(0,1)
    noise = ((source - target) ** 2).mean(dim=3).mean(dim=2).mean(dim=1)
    signal_max = 1.0
    return (10 * torch.log10(signal_max / noise)).mean().item()


class Inception_Metrics(nn.Module):
    def __init__(self, source_device: torch.device = torch.device("cpu")):
        super(Inception_Metrics, self).__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False).eval()

        self.inception = self.inception.to(source_device)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

        self.FID_inception = InceptionV3([block_idx]).to(source_device)
        self.up = nn.Upsample(
            size=(299, 299), mode="bilinear", align_corners=False
        ).float()

        self.source_device = source_device

    def get_inception_score(self, pred: "Tensor"):
        py = np.mean(pred, axis=0)
        is_scores = []
        for i in range(pred.shape[0]):
            pyx = pred[i, :]
            is_scores.append(entropy(pyx, py))
        return torch.exp(torch.Tensor(is_scores).mean())

    def get_FID(self, output_pool_ll: "Tensor", target_pool_ll: "Tensor"):
        mu_1 = np.mean(output_pool_ll, axis=0)
        sigma_1 = np.cov(output_pool_ll, rowvar=False)

        mu_2 = np.mean(target_pool_ll, axis=0)
        sigma_2 = np.cov(target_pool_ll, rowvar=False)

        return self.calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

    def forward(self, outputs, targets):
        outputs = outputs.to(self.source_device)
        targets = targets.to(self.source_device)

        x = self.inception(2 * self.up(outputs) - 1)
        pred = F.softmax(x, dim=1).detach().cpu().numpy()

        # IS
        # inception_score = self.get_inception_score(pred)

        # array of pool 3 vectors
        output_pool_ll = (
            self.FID_inception(outputs)[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(outputs.shape[0], -1)
        )
        target_pool_ll = (
            self.FID_inception(targets)[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(targets.shape[0], -1)
        )

        return pred, output_pool_ll, target_pool_ll

        FID = self.get_FID(output_pool_ll, target_pool_ll)

        return inception_score.item(), FID

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test_16_jan mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test_16_jan covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@ex.automain
def main():
    source = torch.rand(3, 3, 256, 256)
    target = torch.rand(3, 3, 256, 256)
    #
    # psnr = PSNR(source, target)
    # logging.info(f"PSNR is {psnr}")

    inception_metrics = Inception_Metrics()

    IS, fid = inception_metrics(source, target)
    logging.info(f"Inception Score is {IS} FID is {fid}")
