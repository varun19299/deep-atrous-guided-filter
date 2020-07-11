"""
Val Script for Phase/Amp mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from loss import GLoss
import logging
import cv2

# Torch Libs
import torch
from torch.utils.tensorboard import SummaryWriter
from PerceptualSimilarity.models import PerceptualLoss

# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from utils.tupperware import tupperware
from models import get_model
from metrics import PSNR
from config import initialise
from skimage.metrics import structural_similarity as ssim


# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.train_helper import set_device, load_models, AvgLoss_with_dict

# Self ensemble
from utils.self_ensemble import ensemble_ops, plot_single

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# Save mat
from scipy.io.matlab.mio import savemat

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.lambda_perception = 0.0
    args.finetune = False
    args.batch_size = 1
    args.do_augment = False

    # Set device, init dirs
    device, source_device = set_device(args)
    dir_init(args)
    # Get data
    data = get_dataloaders(args)

    # Model
    G, _ = get_model.model(args)
    G = G.to(device)

    # LPIPS Criterion
    lpips_criterion = PerceptualLoss(
        model="net-lin", net="alex", use_gpu=True, gpu_ids=[device]
    ).to(device)

    # Load Models
    (G, _), _, global_step, start_epoch, loss = load_models(
        G,
        D=None,
        g_optimizer=None,
        d_optimizer=None,
        args=args,
        tag=args.inference_mode,
    )

    # Compatibility with checkpoints without global_step
    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size

    _metrics_dict = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_01": 0.0, "LPIPS_11": 0.0}

    avg_train_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict.copy(), args=args)

    avg_val_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict, args=args)

    logging.info(f"Loaded experiment {args.exp_name} trained for {start_epoch} epochs.")

    # Train, val and test paths
    if args.self_ensemble:
        train_path = (
            args.output_dir
            / f"train_{args.inference_mode}_epoch_{start_epoch}_self_ensemble"
        )
    else:
        train_path = args.output_dir / f"val_{args.inference_mode}_epoch_{start_epoch}"
    train_path.mkdir(exist_ok=True, parents=True)

    if args.self_ensemble:
        val_path = (
            args.output_dir
            / f"val_{args.inference_mode}_epoch_{start_epoch}_self_ensemble"
        )
    else:
        val_path = args.output_dir / f"val_{args.inference_mode}_epoch_{start_epoch}"
    val_path.mkdir(exist_ok=True, parents=True)

    if args.self_ensemble:
        test_path = (
            args.output_dir
            / f"test_{args.inference_mode}_epoch_{start_epoch}_self_ensemble"
        )
    else:
        test_path = args.output_dir / f"test_{args.inference_mode}_epoch_{start_epoch}"
    test_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        G.eval()

        if data.train_loader and args.save_train:
            # Run val for an epoch
            avg_val_metrics.reset()
            pbar = tqdm(
                range(len(data.train_loader) * args.batch_size), dynamic_ncols=True
            )

            for i, batch in enumerate(data.train_loader):
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

                    output_channel_concat = torch.cat(output_ensembled, dim=1).squeeze(
                        0
                    )
                    output_ensembled = torch.cat(output_ensembled, dim=0)

                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                if args.save_ensemble_channels:
                    name = filename[0].replace(".png", ".npy")
                    path_output = train_path / f"channel_concat_{name}"
                    np.save(path_output, output_channel_concat.cpu().numpy())

                # PSNR
                output_quant = (output.mul(0.5).add(0.5) * 255.0).int().float() / 255.0
                output_quant = output_quant.sub(0.5).mul(2)
                target_quant = (target.mul(0.5).add(0.5) * 255.0).int().float() / 255.0
                target_quant = target_quant.sub(0.5).mul(2)
                metrics_dict["PSNR"] += PSNR(output_quant, target_quant)

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
                        target_quant[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    output_numpy = (
                        output_quant[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    metrics_dict["SSIM"] += ssim(
                        target_numpy, output_numpy, multichannel=True, data_range=1.0
                    )

                    # Dump to output folder
                    # Phase and amplitude are nested
                    name = filename[e]
                    path_output = val_path / name

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
                avg_train_metrics += metrics_dict

                pbar.update(args.batch_size)
                pbar.set_description(
                    f"Train Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_train_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_train_metrics.loss_dict['SSIM']:.3f} | LPIPS 01: {avg_train_metrics.loss_dict['LPIPS_01']:.3f} | LPIPS 11: {avg_train_metrics.loss_dict['LPIPS_11']:.3f}"
                )

            with open(train_path / "metrics.txt", "w") as f:
                L = [
                    f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                    "Train Metrics \n\n",
                ]
                L = L + [f"{k}:{v}\n" for k, v in avg_val_metrics.loss_dict.items()]
                f.writelines(L)

        if data.val_loader:
            # Run val for an epoch
            avg_val_metrics.reset()
            pbar = tqdm(
                range(len(data.val_loader) * args.batch_size), dynamic_ncols=True
            )

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

                    output_channel_concat = torch.cat(output_ensembled, dim=1).squeeze(
                        0
                    )
                    output_ensembled = torch.cat(output_ensembled, dim=0)

                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                if args.save_ensemble_channels:
                    name = filename[0].replace(".png", ".npy")
                    path_output = val_path / f"channel_concat_{name}"
                    np.save(path_output, output_channel_concat.cpu().numpy())

                # PSNR
                output_quant = (output.mul(0.5).add(0.5) * 255.0).int().float() / 255.0
                output_quant = output_quant.sub(0.5).mul(2)
                target_quant = (target.mul(0.5).add(0.5) * 255.0).int().float() / 255.0
                target_quant = target_quant.sub(0.5).mul(2)
                metrics_dict["PSNR"] += PSNR(output_quant, target_quant)

                # LPIPS
                # TODO: check if results agree with udc_paper
                metrics_dict["LPIPS_01"] += lpips_criterion(
                    output_quant.mul(0.5).add(0.5), target_quant.mul(0.5).add(0.5)
                ).item()

                metrics_dict["LPIPS_11"] += lpips_criterion(
                    output_quant, target_quant
                ).item()

                for e in range(args.batch_size):
                    # Compute SSIM
                    target_numpy = (
                        target_quant[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    output_numpy = (
                        output_quant[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    metrics_dict["SSIM"] += ssim(
                        target_numpy, output_numpy, multichannel=True, data_range=1.0
                    )

                    # Dump to output folder
                    # Phase and amplitude are nested
                    name = filename[e]
                    path_output = val_path / name

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
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

            if args.save_mat:
                output_mat = np.zeros(
                    (len(data.test_loader), args.image_height, args.image_width, 3),
                    dtype=np.uint8,
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

                    output_channel_concat = torch.cat(output_ensembled, dim=1).squeeze(
                        0
                    )
                    output_ensembled = torch.cat(output_ensembled, dim=0)

                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                if args.save_ensemble_channels:
                    name = filename[0].replace(".png", ".npy")
                    path_output = test_path / f"channel_concat_{name}"
                    np.save(path_output, output_channel_concat.cpu().numpy())

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
                    name = filename[e]
                    path_output = test_path / name

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                    if args.save_mat:
                        # Save to mat file
                        output_numpy_int8 = (output_numpy * 255.0).astype(np.uint8)
                        output_index = int(name.replace(".png", "")) - 1
                        output_mat[output_index] = output_numpy_int8

                pbar.update(args.batch_size)
                pbar.set_description(f"Test Epoch : {start_epoch} Step: {global_step}")

            if args.save_mat:
                # mat file
                savemat(test_path / "results.mat", {"results": output_mat})

                # submission indormation
                runtime = 0.0  # seconds / megapixel
                cpu_or_gpu = 0  # 0: GPU, 1: CPU
                method = 1  # 0: traditional methods, 1: deep learning method
                other = "(optional) any additional description or information"

                # prepare and save readme file
                with open(test_path / "readme.txt", "w") as readme_file:
                    readme_file.write(f"Runtime (seconds / megapixel): {runtime}\n")
                    readme_file.write(f"CPU[1] / GPU[0]: {cpu_or_gpu}\n")
                    readme_file.write(f"Method: {method}\n")
                    readme_file.write(f"Other description: {other}\n")
