"""
Get model
"""
import logging

from models.Discriminator.discriminator import Discriminator
from models.guided_filtering_net import (
    DeepGuidedFilterGuidedMapConvGF,
    DeepGuidedFilterGuidedMapConvGFGDRN,
    DeepGuidedFilterGuidedMapConvGFPixelShuffle,
    DeepGuidedFilterGuidedMapConvGFPixelShuffleGCA,
    DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAImproved,
)


def model(args, source_device=None, target_device=None):
    if args.model == "guided-filter":
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
            DeepGuidedFilterGuidedMapConvGF(layer=args.CAN_layers),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-gdrn":
        return (
            DeepGuidedFilterGuidedMapConvGFGDRN(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-pixelshuffle":
        return (
            DeepGuidedFilterGuidedMapConvGFPixelShuffle(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-pixelshuffle-gca":
        return (
            DeepGuidedFilterGuidedMapConvGFPixelShuffleGCA(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-pixelshuffle-gca-improved":
        return (
            DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAImproved(args),
            Discriminator(
                args,
                source_device=source_device,
                target_device=target_device,
                use_pool=not args.use_patch_gan,
            ),
        )

    elif args.model == "guided-filter-pixelshuffle-gca-improved-FFA":
        return (
            DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAImproved(args, use_FFA=True),
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
