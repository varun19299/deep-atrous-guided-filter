"""
Val Script
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging
import cv2

# Torch Libs
import torch
from torch.utils.tensorboard import SummaryWriter
from PerceptualSimilarity.models import PerceptualLoss

# Modules
from dataloader import get_dataloaders
from utils.tupperware import tupperware
from models import get_model
from metrics import PSNR_numpy
from config import initialise

# from skimage.metrics import structural_similarity as ssim
from utils.myssim import compare_ssim as ssim

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.train_helper import load_models, AvgLoss_with_dict

# Self ensemble
from utils.self_ensemble import ensemble_ops

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.finetune = False
    args.batch_size = 1

    device = args.device

    # Get data
    data = get_dataloaders(args)

    # Model
    G = get_model.model(args).to(device)

    # LPIPS Criterion
    use_gpu  = device != "cpu"
    lpips_criterion = PerceptualLoss(
        model="net-lin", net="alex", use_gpu=use_gpu, gpu_ids=[device]
    ).to(device)

    # Load Models
    G, _, global_step, start_epoch, loss = load_models(
        G, g_optimizer=None, args=args, tag=args.inference_mode
    )

    # Metric loggers
    val_metrics_dict = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_01": 0.0, "LPIPS_11": 0.0}
    avg_val_metrics = AvgLoss_with_dict(loss_dict=val_metrics_dict, args=args)

    logging.info(f"Loaded experiment {args.exp_name} trained for {start_epoch} epochs.")

    # Train, val and test paths
    val_path = args.output_dir / f"val_{args.inference_mode}_epoch_{start_epoch}"
    test_path = args.output_dir / f"test_{args.inference_mode}_epoch_{start_epoch}"

    if args.self_ensemble:
        val_path = val_path.parent / f"{val_path.name}_self_ensemble"
        test_path = test_path.parent / f"{test_path.name}_self_ensemble"

    val_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        G.eval()

        # Run val for an epoch
        avg_val_metrics.reset()
        pbar = tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)

        for i, batch in enumerate(data.val_loader):
            metrics_dict = defaultdict(float)

            source, target, filename = batch
            source, target = (source.to(device), target.to(device))

            output = G(source)

            if args.self_ensemble:
                output_ensembled = [output]

                for k in ensemble_ops.keys():
                    # Forward transform
                    source_t = ensemble_ops[k][0](source)
                    output_t = G(source_t)
                    # Inverse transform
                    output_t = ensemble_ops[k][1](output_t)

                    output_ensembled.append(output_t)

                output_ensembled = torch.cat(output_ensembled, dim=0)

                output = torch.mean(output_ensembled, dim=0, keepdim=True)

            # PSNR
            output_255 = (output.mul(0.5).add(0.5) * 255.0).int()
            output_quant = (output_255.float() / 255.0).sub(0.5).mul(2)

            target_255 = (target.mul(0.5).add(0.5) * 255.0).int()
            target_quant = (target_255.float() / 255.0).sub(0.5).mul(2)

            # LPIPS
            metrics_dict["LPIPS_01"] += lpips_criterion(
                output_quant.mul(0.5).add(0.5), target_quant.mul(0.5).add(0.5)
            ).item()

            metrics_dict["LPIPS_11"] += lpips_criterion(
                output_quant, target_quant
            ).item()

            for e in range(args.batch_size):
                # Compute SSIM
                target_numpy = (
                    target[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )

                output_numpy = (
                    output[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )

                metrics_dict["PSNR"] += PSNR_numpy(target_numpy, output_numpy)
                metrics_dict["SSIM"] += ssim(
                    target_numpy,
                    output_numpy,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                    multichannel=True,
                )

                # Dump to output folder
                path_output = val_path / filename[e]

                cv2.imwrite(
                    str(path_output), (output_numpy[:, :, ::-1] * 255.0).astype(np.int)
                )

            metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
            metrics_dict["PSNR"] = metrics_dict["PSNR"] / args.batch_size

            avg_val_metrics += metrics_dict

            pbar.update(args.batch_size)
            pbar.set_description(
                f"Val Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_val_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_val_metrics.loss_dict['SSIM']:.3f} | LPIPS 01: {avg_val_metrics.loss_dict['LPIPS_01']:.3f} | LPIPS 11: {avg_val_metrics.loss_dict['LPIPS_11']:.3f}"
            )

        with open(val_path / "metrics.txt", "w") as f:
            L = [
                f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                "Val Metrics \n\n",
            ]
            L = L + [f"{k}:{v}\n" for k, v in avg_val_metrics.loss_dict.items()]
            f.writelines(L)

        if data.test_loader:
            pbar = tqdm(
                range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
            )

            for i, batch in enumerate(data.test_loader):

                source, filename = batch
                source = source.to(device)

                output = G(source)

                if args.self_ensemble:
                    output_ensembled = [output]

                    for k in ensemble_ops.keys():
                        # Forward transform
                        source_t = ensemble_ops[k][0](source)
                        output_t = G(source_t)
                        # Inverse transform
                        output_t = ensemble_ops[k][1](output_t)

                        output_ensembled.append(output_t)

                    output_ensembled = torch.cat(output_ensembled, dim=0)
                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                for e in range(args.batch_size):
                    output_numpy = (
                        output[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    # Dump to output folder
                    path_output = test_path / filename[e]

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                pbar.update(args.batch_size)
                pbar.set_description(f"Test Epoch : {start_epoch} Step: {global_step}")
