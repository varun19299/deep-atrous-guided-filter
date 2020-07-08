from typing import TYPE_CHECKING
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

# Contextual
from utils.contextual_loss import contextual_bilateral_loss
from utils.ops import sample_patches

# SSIM
from pytorch_msssim import MS_SSIM
from pytorch_msssim.ssim import gaussian_filter, _fspecial_gauss_1d

if TYPE_CHECKING:
    from utils.typing_alias import *


def label_like(label: int, x: "Tensor") -> "Tensor":
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def soft_zeros_like(x: "Tensor") -> "Tensor":
    zeros = label_like(0, x)
    return torch.rand_like(zeros)


def soft_ones_like(x: "Tensor") -> "Tensor":
    ones = label_like(1, x)
    return ones * 0.7 + torch.rand_like(ones) * 0.5


def zeros_like(x: "Tensor") -> "Tensor":
    zeros = label_like(0, x)
    return zeros


def ones_like(x: "Tensor") -> "Tensor":
    ones = label_like(1, x)
    return ones


class GLoss(nn.Module):
    def __init__(
        self, args ):
        super(GLoss, self).__init__()

        self.args = args
        self.downsample = nn.Upsample(
            size=(256, 512), mode="bilinear", align_corners=True
        )

        if args.lambda_ms_ssim:
            self.ms_ssim_module = MS_SSIM(
                data_range=1.0, size_average=True, channel=3
            )
            self.win = _fspecial_gauss_1d(11, 1.5)

        if args.lambda_perception:
            self.LossNetwork = Vgg16FeatureExtractor()
            self.LossNetwork.eval()

    def _perception_metric(self, X, Y):
        # feat_X = self.LossNetwork(X.to(self.perception_device))
        # feat_Y = self.LossNetwork(Y.to(self.perception_device))
        #

        feat_X = self.LossNetwork(self.downsample(X))
        feat_Y = self.LossNetwork(self.downsample(Y))

        loss = F.mse_loss(feat_X.relu2_2, feat_Y.relu2_2)
        loss = loss + F.mse_loss(feat_X.relu4_3, feat_Y.relu4_3)

        return loss

    def _CoBi_RGB(self, X, Y, patch_size: int = 24):
        """
        See https://arxiv.org/pdf/1905.05169.pdf
        for details of CoBi

        :param patch_size: Optimal around 16-32
        """
        n, c, h, w = X.shape

        patch_size = self.args.cobi_rgb_patch_size
        stride = self.args.cobi_rgb_stride

        # Extract patches
        X_patch = sample_patches(X, patch_size=patch_size, stride=stride)
        Y_patch = sample_patches(Y, patch_size=patch_size, stride=stride)
        _, _, h_patch, w_patch, n_patches = X_patch.shape

        X_vec = X_patch.reshape(n, -1, h_patch, w_patch)
        Y_vec = Y_patch.reshape(n, -1, h_patch, w_patch)

        return contextual_bilateral_loss(X_vec, Y_vec)

    def forward(
        self,
        fake_logit: "Tensor" = [],
        real_logit: "Tensor" = [],
        output: "Tensor" = [],
        target: "Tensor" = [],
    ):
        self.total_loss = torch.tensor(0.0).type_as(output)
        self.adversarial_loss = torch.tensor(0.0).type_as(output)
        self.perception_loss = torch.tensor(0.0).type_as(output)
        self.image_loss = torch.tensor(0.0).type_as(output)
        self.ms_ssim_loss = torch.tensor(0.0).type_as(output)
        self.cobi_rgb_loss = torch.tensor(0.0).type_as(output)

        # L1
        if self.args.lambda_image:
            # l1_diff = torch.abs(output - target)
            # if self.args.lambda_ms_ssim:
            #     # Blur l1
            #     _, C, _, _ = l1_diff.shape
            #     win = self.win.repeat(C, 1, 1, 1)
            #     l1_diff = gaussian_filter(l1_diff, win)

            self.image_loss += F.l1_loss(output, target) * self.args.lambda_image

        # VGG 19
        if self.args.lambda_perception:
            self.perception_loss += (
                self._perception_metric(output, target).to(device)
                * self.args.lambda_perception
            )

        # https://github.com/VainF/pytorch-msssim
        if self.args.lambda_ms_ssim:
            self.ms_ssim_loss += (
                1
                - self.ms_ssim_module(
                    output.mul(0.5).add(0.5), target.mul(0.5).add(0.5)
                )
            ) * self.args.lambda_ms_ssim

        if self.args.lambda_CoBi_RGB:
            self.cobi_rgb_loss += (
                self._CoBi_RGB(output, target)
            ) * self.args.lambda_CoBi_RGB

        if self.args.lambda_adversarial:
            if self.args.gan_type == "NSGAN":
                self.adversarial_loss += (
                    F.binary_cross_entropy_with_logits(
                        fake_logit, ones_like(fake_logit)
                    )
                    * self.args.lambda_adversarial
                )
            elif self.args.gan_type == "RAGAN":
                self.adversarial_loss += (
                    F.binary_cross_entropy_with_logits(
                        fake_logit - torch.mean(real_logit), ones_like(fake_logit)
                    )
                    * self.args.lambda_adversarial
                )

                self.adversarial_loss += (
                    F.binary_cross_entropy_with_logits(
                        real_logit - torch.mean(fake_logit), zeros_like(fake_logit)
                    )
                    * self.args.lambda_adversarial
                )

                self.adversarial_loss /= 2.0

        self.total_loss += (
            self.adversarial_loss
            + self.image_loss
            + self.perception_loss
            + self.ms_ssim_loss
            + self.cobi_rgb_loss
        )

        return self.total_loss


class DLoss(nn.Module):
    def __init__(self, args):
        super(DLoss, self).__init__()
        self.args = args

    def forward(self, real_logit, fake_logit):
        self.total_loss = 0.0

        if self.args.gan_type == "NSGAN":
            self.real_loss = F.binary_cross_entropy_with_logits(
                real_logit, soft_ones_like(real_logit)
            )
            self.fake_loss = F.binary_cross_entropy_with_logits(
                fake_logit, soft_zeros_like(fake_logit)
            )

        elif self.args.gan_type == "RAGAN":
            self.real_loss = F.binary_cross_entropy_with_logits(
                real_logit - torch.mean(fake_logit), ones_like(real_logit)
            )
            self.fake_loss = F.binary_cross_entropy_with_logits(
                fake_logit - torch.mean(real_logit), zeros_like(fake_logit)
            )

        self.total_loss = (self.real_loss + self.fake_loss) / 2.0

        return self.total_loss


class Vgg16FeatureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16FeatureExtractor, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        # h_relu1_2 = h

        h = self.slice2(h)
        h_relu2_2 = h

        h = self.slice3(h)
        h_relu3_3 = h

        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple("VggOutputs", ["relu2_2", "relu3_3", "relu4_3"])

        out = vgg_outputs(h_relu2_2, h_relu3_3, h_relu4_3)

        return out
