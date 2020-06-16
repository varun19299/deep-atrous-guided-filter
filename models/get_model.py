"""
Get model
"""
import logging

from models.unet import Unet
from models.hdrnet import HDRNet
from models.Discriminator import Discriminator
from models.guided_filtering_net import (
    DeepGuidedFilterGuidedMapConvGF,
    DeepGuidedFilterGuidedMapConvGFResUnet,
)


def model(args, source_device=None, target_device=None):
    if args.model == "unet":
        return (
            Unet(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "hdrnet":
        return (
            HDRNet(),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter":
        return (
            DeepGuidedFilterGuidedMapConvGF(),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-deeper":
        return (
            DeepGuidedFilterGuidedMapConvGF(layer=9),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-resunet":
        return (
            DeepGuidedFilterGuidedMapConvGFResUnet(args),
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
