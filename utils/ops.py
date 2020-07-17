import torch
from torch.nn import functional as F


def unpixel_shuffle(feature, r: int = 1):
    b, c, h, w = feature.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    feature_view = feature.contiguous().view(b, c, out_h, r, out_w, r)
    feature_prime = (
        feature_view.permute(0, 1, 3, 5, 2, 4)
        .contiguous()
        .view(b, out_channel, out_h, out_w)
    )
    return feature_prime


def sample_patches(
    inputs: torch.Tensor, patch_size: int = 3, stride: int = 2
) -> torch.Tensor:
    """

    :param inputs: the input feature maps, shape: (n, c, h, w).
    :param patch_size: the spatial size of sampled patches
    :param stride: the stride of sampling.
    :return: extracted patches, shape: (n, c, patch_size, patch_size, n_patches).
    """
    """
    Patch sampler for feature maps.
    Parameters
    ---
    inputs : torch.Tensor
        
    patch_size : int, optional
       
    stride : int, optional
        
    Returns
    ---
    patches : torch.Tensor
        
    """

    n, c, h, w = inputs.shape
    patches = (
        inputs.unfold(2, patch_size, stride)
        .unfold(3, patch_size, stride)
        .reshape(n, c, -1, patch_size, patch_size)
        .permute(0, 1, 3, 4, 2)
    )
    return patches


def chop_patches(
    img: torch.Tensor, patch_size_h: int = 256, patch_size_w: int = 512
) -> torch.Tensor:
    """

    :param inputs: the input feature maps, shape: (n, c, h, w).
    :param patch_size: the spatial size of sampled patches
    :param stride: the stride of sampling.
    :return: extracted patches, shape: (n, c, patch_size, patch_size, n_patches).
    """
    """
    Patch sampler for feature maps.
    Parameters
    ---
    inputs : torch.Tensor

    patch_size : int, optional

    stride : int, optional

    Returns
    ---
    patches : torch.Tensor

    """
    n, c, h, w = img.shape
    patches = (
        img.unfold(2, patch_size_h, patch_size_h)
        .unfold(3, patch_size_w, patch_size_w)
        .contiguous()
        .permute(2, 3, 0, 1, 4, 5)
        .flatten(start_dim=0, end_dim=2)
        # .reshape(-1, c, patch_size_h, patch_size_w)
    )
    return patches


def unchop_patches(
    patches: torch.Tensor, img_h: int = 1024, img_w: int = 2048, n: int = 1
) -> torch.Tensor:
    """
    Assumes non-overlapping patches

    See: https://discuss.pytorch.org/t/reshaping-windows-into-image/19805
    """
    _, c, patch_size_h, patch_size_w = patches.shape
    num_h = img_h // patch_size_h
    num_w = img_w // patch_size_w

    img = patches.reshape(n, num_h * num_w, patch_size_h * patch_size_w * c).permute(
        0, 2, 1
    )
    img = F.fold(
        img,
        (img_h, img_w),
        (patch_size_h, patch_size_w),
        1,
        0,
        (patch_size_h, patch_size_w),
    )
    return img.reshape(n, c, img_h, img_w)
