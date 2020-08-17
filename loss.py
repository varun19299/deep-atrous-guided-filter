from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

# Contextual
from utils.contextual_loss import contextual_bilateral_loss
from utils.ops import sample_patches

if TYPE_CHECKING:
    from utils.typing_alias import *


class GLoss(nn.Module):
    def __init__(self, args):
        super(GLoss, self).__init__()
        self.args = args

    def _CoBi_RGB(self, X, Y):
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
        self, output: "Tensor[N,C,H,W]", target: "Tensor[N,C,H,W]"
    ) -> "Tensor[torch.float32]":
        self.total_loss = torch.tensor(0.0).type_as(output)
        self.image_loss = torch.tensor(0.0).type_as(output)
        self.cobi_rgb_loss = torch.tensor(0.0).type_as(output)

        # L1
        if self.args.lambda_image:
            self.image_loss += F.l1_loss(output, target) * self.args.lambda_image

        if self.args.lambda_CoBi_RGB:
            self.cobi_rgb_loss += (
                self._CoBi_RGB(output, target)
            ) * self.args.lambda_CoBi_RGB

        self.total_loss += +self.image_loss + self.cobi_rgb_loss

        return self.total_loss
