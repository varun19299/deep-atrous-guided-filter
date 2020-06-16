import torch


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
    Patch sampler for feature maps.
    Parameters
    ---
    inputs : torch.Tensor
        the input feature maps, shape: (c, h, w).
    patch_size : int, optional
        the spatial size of sampled patches
    stride : int, optional
        the stride of sampling.
    Returns
    ---
    patches : torch.Tensor
        extracted patches, shape: (c, patch_size, patch_size, n_patches).
    """

    n, c, h, w = inputs.shape
    patches = (
        inputs.unfold(2, patch_size, stride)
        .unfold(3, patch_size, stride)
        .reshape(n, c, -1, patch_size, patch_size)
        .permute(0, 1, 3, 4, 2)
    )
    return patches
