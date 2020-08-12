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
import kornia

# Torch Libs
import torch
from torch.utils.tensorboard import SummaryWriter
from PerceptualSimilarity.models import PerceptualLoss

# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
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
from utils.train_helper import set_device, load_models, AvgLoss_with_dict

# Self ensemble
from utils.self_ensemble import ensemble_ops

# Sample patches
from utils.ops import chop_patches, unchop_patches, roll_n

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# Save mat
from scipy.io.matlab.mio import savemat, loadmat

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


def forward_img(G, source, args):
    if args.use_chop_val:
        output_ll = []

        for stride_height in range(0, args.crop_height, args.stride_height):
            for stride_width in range(0, args.crop_width, args.stride_width):
                # Roll
                source_rolled = source.clone()
                source_rolled = roll_n(source_rolled, axis=2, n=stride_height)
                source_rolled = roll_n(source_rolled, axis=3, n=stride_width)

                # Patch based val
                n, c, h, w = source_rolled.shape
                source_chopped = chop_patches(
                    source_rolled, args.crop_height, args.crop_width
                )
                output_chopped = G(source_chopped)

                output = unchop_patches(
                    output_chopped, args.image_height, args.image_width, n=n
                )

                # Roll back
                output = roll_n(output, axis=2, n=h - stride_height)
                output = roll_n(output, axis=3, n=w - stride_width)

                output_ll.append(output)

        output = torch.cat(output_ll, dim=0).mean(dim=0, keepdim=True)
    else:
        output = G(source)

    if args.use_median_filter:
        output = kornia.filters.median_blur(
            output, kernel_size=args.median_filter_kernel
        )

    return output


def evaluate_model(G, data, lpips_criterion, device, args):
    # Load Models
    (G, _), _, global_step, start_epoch, loss = load_models(
        G,
        D=None,
        g_optimizer=None,
        d_optimizer=None,
        args=args,
        tag=args.inference_mode,
    )

    # Metric loggers
    train_metrics_dict = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_01": 0.0, "LPIPS_11": 0.0}
    avg_train_metrics = AvgLoss_with_dict(loss_dict=train_metrics_dict, args=args)

    val_metrics_dict = {
        "PSNR": 0.0,
        "SSIM": 0.0,
        "LPIPS_01": 0.0,
        "LPIPS_11": 0.0,
        "Time": 0.0,
    }
    avg_val_metrics = AvgLoss_with_dict(loss_dict=val_metrics_dict, args=args)

    logging.info(f"Loaded experiment {args.exp_name} trained for {start_epoch} epochs.")

    # Train, val and test paths
    train_path = args.output_dir / f"train_{args.inference_mode}_epoch_{start_epoch}"
    val_path = args.output_dir / f"val_{args.inference_mode}_epoch_{start_epoch}"
    test_path = args.output_dir / f"test_{args.inference_mode}_epoch_{start_epoch}"

    if args.self_ensemble:
        train_path = train_path.parent / f"{train_path.name}_self_ensemble"
        val_path = val_path.parent / f"{val_path.name}_self_ensemble"
        test_path = test_path.parent / f"{test_path.name}_self_ensemble"

    if args.use_median_filter:
        train_path = train_path.parent / f"{train_path.name}_median_filter"
        val_path = val_path.parent / f"{val_path.name}_median_filter"
        test_path = test_path.parent / f"{test_path.name}_median_filter"

    train_path.mkdir(exist_ok=True, parents=True)
    val_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)

    # CUDA events timing
    if args.device == "cuda:0":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

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

                output = forward_img(G, source, args)

                if args.self_ensemble:
                    output_ensembled = [output]

                    for k in ensemble_ops.keys():
                        # Forward transform
                        source_t = ensemble_ops[k][0](source)
                        output_t = forward_img(G, source_t, args)
                        # Inverse transform
                        output_t = ensemble_ops[k][1](output_t)
                        output_ensembled.append(output_t)

                    output_channel_concat = torch.cat(output_ensembled, dim=1).squeeze(
                        0
                    )
                    output_ensembled = torch.cat(output_ensembled, dim=0)

                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                if args.save_ensemble_channels:
                    name = (
                        filename[0]
                        .replace(".png", ".npy")
                        .replace("channel_concat_", "")
                    )
                    path_output = train_path / f"channel_concat_{name}"
                    np.save(path_output, output_channel_concat.cpu().numpy())

                # PSNR
                output_255 = (output.mul(0.5).add(0.5) * 255.0).int()
                output_quant = (output_255.float() / 255.0).sub(0.5).mul(2)

                target_255 = (target.mul(0.5).add(0.5) * 255.0).int()
                target_quant = (target_255.float() / 255.0).sub(0.5).mul(2)

                # metrics_dict["PSNR"] += PSNR_quant(output, target)

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
                        target[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    output_numpy = (
                        output[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
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
                    name = (
                        filename[e]
                        .replace(".npy", ".png")
                        .replace("channel_concat_", "")
                    )
                    path_output = train_path / name

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
                metrics_dict["PSNR"] = metrics_dict["PSNR"] / args.batch_size

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

            if args.save_mat:
                output_mat = np.zeros(
                    (len(data.test_loader), args.image_height, args.image_width, 3),
                    dtype=np.uint8,
                )

            for i, batch in enumerate(data.val_loader):
                metrics_dict = defaultdict(float)

                source, target, filename = batch
                source, target = (source.to(device), target.to(device))

                if args.device == "cuda:0":
                    start.record()

                output = forward_img(G, source, args)

                if args.self_ensemble:
                    output_ensembled = [output]

                    for k in ensemble_ops.keys():
                        # Forward transform
                        source_t = ensemble_ops[k][0](source)

                        output_t = forward_img(G, source_t, args)

                        # Inverse transform
                        output_t = ensemble_ops[k][1](output_t)

                        output_ensembled.append(output_t)

                    output_channel_concat = torch.cat(output_ensembled, dim=1).squeeze(
                        0
                    )
                    output_ensembled = torch.cat(output_ensembled, dim=0)

                    output = torch.mean(output_ensembled, dim=0, keepdim=True)

                # Inference time
                if args.device == "cuda:0":
                    end.record()
                    torch.cuda.synchronize()
                    metrics_dict["Time"] = start.elapsed_time(end)
                else:
                    metrics_dict["Time"] = 0.0

                if args.save_ensemble_channels:
                    name = filename[0].replace(".png", ".npy")
                    path_output = val_path / f"channel_concat_{name}"
                    np.save(path_output, output_channel_concat.cpu().numpy())

                # PSNR
                output_255 = (output.mul(0.5).add(0.5) * 255.0).int()
                output_quant = (output_255.float() / 255.0).sub(0.5).mul(2)

                target_255 = (target.mul(0.5).add(0.5) * 255.0).int()
                target_quant = (target_255.float() / 255.0).sub(0.5).mul(2)

                # metrics_dict["PSNR"] += PSNR_quant(output, target)

                # LPIPS
                metrics_dict["LPIPS_01"] += lpips_criterion(
                    output_quant.mul(0.5).add(0.5), target_quant.mul(0.5).add(0.5)
                ).item()

                metrics_dict["LPIPS_11"] += lpips_criterion(
                    output_quant, target_quant
                ).item()

                # if filename[0] in ["8.png", "15.png", "16.png"]:
                #     breakpoint()

                for e in range(args.batch_size):
                    # Compute SSIM
                    target_numpy = (
                        target[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    output_numpy = (
                        output[e]
                        .mul(0.5)
                        .add(0.5)
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
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
                    name = (
                        filename[e]
                        .replace(".npy", ".png")
                        .replace("channel_concat_", "")
                    )
                    path_output = val_path / name

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )

                    if args.save_mat:
                        # Save to mat file
                        output_numpy_int8 = (output_numpy * 255.0).astype(np.uint8)
                        output_index = int(name.replace(".png", "")) - 1
                        output_mat[output_index] = output_numpy_int8

                metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
                metrics_dict["PSNR"] = metrics_dict["PSNR"] / args.batch_size

                avg_val_metrics += metrics_dict

                pbar.update(args.batch_size)
                pbar.set_description(
                    f"Val Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_val_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_val_metrics.loss_dict['SSIM']:.3f} | LPIPS 01: {avg_val_metrics.loss_dict['LPIPS_01']:.3f} | LPIPS 11: {avg_val_metrics.loss_dict['LPIPS_11']:.3f}"
                )

            if args.save_mat:
                # mat file
                savemat(val_path / "results.mat", {"results": output_mat})

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

                output = forward_img(G, source, args)

                if args.self_ensemble:
                    output_ensembled = [output]

                    for k in ensemble_ops.keys():
                        # Forward transform
                        source_t = ensemble_ops[k][0](source)

                        output_t = forward_img(G, source_t, args)

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
                    name = (
                        filename[e]
                        .replace(".npy", ".png")
                        .replace("channel_concat_", "")
                    )
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

    if not args.model_ensemble:
        evaluate_model(
            G=G, data=data, lpips_criterion=lpips_criterion, device=device, args=args
        )
    else:
        # save mat = True
        args.save_mat = True

        epoch_list = list(
            range(args.num_epochs - args.save_num_snapshots, args.num_epochs)
        )
        epoch_list = [892, 893, 894, 895, 956, 957, 958, 959]

        # ensemble mats
        ensembled_val_mat = []
        ensembled_test_mat = []

        # ensemble paths
        ensemble_val_path = args.output_dir / f"val_model_ensemble_{epoch_list}"
        ensemble_test_path = args.output_dir / f"test_model_ensemble_{epoch_list}"

        if args.self_ensemble:
            ensemble_val_path = (
                ensemble_val_path.parent / f"{ensemble_val_path.name}_self_ensemble"
            )
            ensemble_test_path = (
                ensemble_test_path.parent / f"{ensemble_test_path.name}_self_ensemble"
            )

        ensemble_val_path.mkdir(exist_ok=True, parents=True)
        ensemble_test_path.mkdir(exist_ok=True, parents=True)

        for epoch in epoch_list:
            args.inference_mode = epoch

            # Average all val and test .mat files
            # Train, val and test paths
            val_mat_path = args.output_dir / f"val_{args.inference_mode}_epoch_{epoch}"
            test_mat_path = (
                args.output_dir / f"test_{args.inference_mode}_epoch_{epoch}"
            )

            if args.self_ensemble:
                val_mat_path = (
                    val_mat_path.parent / f"{val_mat_path.name}_self_ensemble"
                )
                test_mat_path = (
                    test_mat_path.parent / f"{test_mat_path.name}_self_ensemble"
                )

            val_mat_path = val_mat_path / "results.mat"
            test_mat_path = test_mat_path / "results.mat"

            if (not val_mat_path.exists()) or (not test_mat_path.exists()):
                # Evaluate model
                evaluate_model(
                    G=G,
                    data=data,
                    lpips_criterion=lpips_criterion,
                    device=device,
                    args=args,
                )
            else:
                print(f"Model at epoch {epoch + 1} evaluated. Skipping.")

            # Open mat files and append
            val_mat = loadmat(val_mat_path)["results"]
            test_mat = loadmat(test_mat_path)["results"]

            ensembled_val_mat.append(val_mat)
            ensembled_test_mat.append(test_mat)

        # Now average them
        ensembled_val_mat = np.array(ensembled_val_mat).mean(axis=0).astype(np.uint8)
        ensembled_test_mat = np.array(ensembled_test_mat).mean(axis=0).astype(np.uint8)

        # Save mat files
        savemat(ensemble_val_path / "results.mat", {"results": ensembled_val_mat})
        savemat(ensemble_test_path / "results.mat", {"results": ensembled_test_mat})
