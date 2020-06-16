from typing import TYPE_CHECKING
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F

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


class GLoss(nn.modules.Module):
    def __init__(
        self, args, device=torch.device("cpu"), perception_device=torch.device("cpu")
    ):
        super(GLoss, self).__init__()

        self.args = args
        self.device = device

        self.perception_device = perception_device
        self.downsample = nn.Upsample(
            size=(256, 512), mode="bilinear", align_corners=True
        )

        if self.args.lambda_perception:
            self.LossNetwork = Vgg16FeatureExtractor().to(perception_device)
            self.LossNetwork.eval()

    def _perception_metric(self, X, Y):
        # feat_X = self.LossNetwork(X.to(self.perception_device))
        # feat_Y = self.LossNetwork(Y.to(self.perception_device))
        #

        feat_X = self.LossNetwork(self.downsample(X).to(self.perception_device))
        feat_Y = self.LossNetwork(self.downsample(Y).to(self.perception_device))

        loss = F.mse_loss(feat_X.relu2_2, feat_Y.relu2_2)
        loss = loss + F.mse_loss(feat_X.relu4_3, feat_Y.relu4_3)

        return loss

    def forward(
        self,
        fake_logit: "Tensor" = [],
        real_logit: "Tensor" = [],
        output: "Tensor" = [],
        target: "Tensor" = [],
        pretrain: bool = False,
    ):
        self.total_loss = torch.tensor(0.0).to(self.device)

        self.adversarial_loss = torch.tensor(0.0).to(self.device)
        self.perception_loss = torch.tensor(0.0).to(self.device)
        self.image_loss = torch.tensor(0.0).to(self.device)

        device = output.device

        # During pretrain only MSE

        if self.args.lambda_image:
            self.image_loss += F.l1_loss(output, target).mean() * self.args.lambda_image

        if self.args.lambda_perception:
            self.perception_loss += (
                self._perception_metric(output, target).to(device)
                * self.args.lambda_perception
            )

        if len(fake_logit):

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
            self.adversarial_loss + self.image_loss + self.perception_loss
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
