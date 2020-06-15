"""
Get model
"""
import logging
from models.fftlayer import FFTLayer

from models.unet import Unet
from models.unet_pixel_shuffle import Unet as Unet_PixelShuffle
from models.hdrnet import HDRNet

from models.Discriminator import Discriminator


def model(args, source_device=None, target_device=None):
    if args.model == "unet-fft":
        return (
            Unet(args),
            FFTLayer(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "hdrnet-fft":
        return (
            HDRNet(),
            FFTLayer(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "unet-pixelshuffle-fft":
        return (
            Unet_PixelShuffle(args),
            FFTLayer(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    else:
        logging.info(f"Model {args.model} not implemented.")
        breakpoint()
