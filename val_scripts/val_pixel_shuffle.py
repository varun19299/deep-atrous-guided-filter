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
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
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
from utils.ops import unpixel_shuffle
from utils.train_helper import set_device, load_models, AvgLoss_with_dict

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.config
def config():
    gain = 1.0
    tag = "384"


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.lambda_lpips = 1.0
    args.lambda_perception = 0.0
    args.batch_size = 1
    args.finetune = False

    # Set device, init dirs
    device, source_device = set_device(args)
    dir_init(args)

    # Get data
    data = get_dataloaders(args)

    # Model
    G, FFT, _ = get_model.model(args, source_device=source_device, target_device=device)
    G = G.to(device)
    FFT = FFT.to(device)

    # Load Models
    (G, FFT, _), _, global_step, start_epoch, loss = load_models(
        G,
        FFT,
        D=None,
        g_optimizer=None,
        fft_optimizer=None,
        d_optimizer=None,
        args=args,
        tag=args.inference_mode,
    )

    # Initialise losses
    g_loss = GLoss(
        args, device, lpips_device=source_device, perception_device=source_device
    )

    # Compatibility with checkpoints without global_step
    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size
    start_epoch = global_step // len(data.train_loader.dataset)

    _metrics_dict = {
        "PSNR": 0.0,
        "LPIPS_01": 0.0,
        "LPIPS_11": 0.0,
        "SSIM": 0.0,
        "Time": 0.0,
    }
    avg_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict, args=args)

    logging.info(
        f"Loaded experiment {args.exp_name}, dataset {args.dataset_name}, trained for {start_epoch} epochs."
    )

    # Run val for an epoch
    avg_metrics.reset()
    pbar = tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)

    if args.device == "cuda:0":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = end = 0

    # Val and test paths
    val_path = (
        args.output_dir / f"val_{args.inference_mode}_tag_{args.tag}_gain_{args.gain}"
    )
    val_path.mkdir(exist_ok=True, parents=True)

    test_path = (
        args.output_dir / f"test_{args.inference_mode}_tag_{args.tag}_gain_{args.gain}"
    )
    test_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        G.eval()
        FFT.eval()

        for i, batch in enumerate(data.val_loader):
            metrics_dict = defaultdict(float)

            source, target, filename = batch
            source, target = (source.to(device), target.to(device))

            if args.device == "cuda:0" and i:
                start.record()

            fft_output = FFT(source)
            fft_output_16_channel = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
            output_12_channel = G(fft_output_16_channel * args.gain)[-1]
            output = F.pixel_shuffle(output_12_channel, args.pixelshuffle_ratio)

            if args.device == "cuda:0" and i:
                end.record()
                torch.cuda.synchronize()
                metrics_dict["Time"] = start.elapsed_time(end)
            else:
                metrics_dict["Time"] = 0.0

            # PSNR
            metrics_dict["PSNR"] += PSNR(output, target)
            metrics_dict["LPIPS_01"] += (
                g_loss.lpips_model.forward(
                    output.mul(0.5).add(0.5).to(g_loss.lpips_device),
                    target.mul(0.5).add(0.5).to(g_loss.lpips_device),
                )
                .mean()
                .item()
            )

            metrics_dict["LPIPS_11"] += (
                g_loss.lpips_model.forward(
                    output.to(g_loss.lpips_device), target.to(g_loss.lpips_device)
                )
                .mean()
                .item()
            )

            if filename[0] in ["n02980441_33851.png", "n07747607_16335.png"]:
                breakpoint()

            for e in range(args.batch_size):
                # Compute SSIM
                target_numpy = (
                    target[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )

                fft_output_vis = fft_output[e]
                fft_output_vis = fft_output_vis.cpu().detach()
                fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                fft_output_vis_rgb[0] = fft_output_vis[0]
                fft_output_vis_rgb[1] = (fft_output_vis[1] + fft_output_vis[2]) * 0.5
                fft_output_vis_rgb[2] = fft_output_vis[3]

                # for i in range(3):
                #     fft_output_vis_rgb[i] = (
                #         fft_output_vis_rgb[i] - fft_output_vis_rgb[i].min()
                #     ) / (fft_output_vis_rgb[i].max() - fft_output_vis_rgb[i].min())

                fft_output_vis_rgb = (fft_output_vis_rgb - fft_output_vis_rgb.min()) / (
                    fft_output_vis_rgb.max() - fft_output_vis_rgb.min()
                )

                fft_output_vis_rgb = fft_output_vis_rgb.permute(1, 2, 0).numpy()

                output_numpy = (
                    output[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                metrics_dict["SSIM"] += ssim(
                    target_numpy, output_numpy, multichannel=True, data_range=1.0
                )

                # Dump to output folder
                # Phase and amplitude are nested
                name = filename[e].replace(".JPEG", ".png")
                parent = name.split("_")[0]
                path = val_path / parent
                path.mkdir(exist_ok=True, parents=True)
                path_output = path / ("output_" + name)
                path_fft = path / ("fft_" + name)

                cv2.imwrite(
                    str(path_output), (output_numpy[:, :, ::-1] * 255.0).astype(np.int)
                )
                cv2.imwrite(
                    str(path_fft),
                    (fft_output_vis_rgb[:, :, ::-1] * 255.0).astype(np.int),
                )

            metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
            avg_metrics += metrics_dict

            pbar.update(args.batch_size)
            pbar.set_description(
                f"Val Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f} | LPIPS_01: {avg_metrics.loss_dict['LPIPS_01']:.3f}"
            )

        with open(val_path / "metrics.txt", "w") as f:
            L = [
                f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                f"Inference mode {args.inference_mode}\n",
                "Metrics \n\n",
            ]
            L = L + [f"{k}:{v}\n" for k, v in avg_metrics.loss_dict.items()]
            f.writelines(L)

        if data.test_loader:
            pbar = tqdm(
                range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
            )
            for i, batch in enumerate(data.test_loader):

                source, filename = batch
                source = source.to(device)

                fft_output = FFT(source)
                fft_output_16_channel = unpixel_shuffle(fft_output, 2)
                output_12_channel = G(fft_output_16_channel * args.gain)[-1]
                output = F.pixel_shuffle(output_12_channel, 2)

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
                    fft_output_vis = fft_output[e]
                    fft_output_vis = fft_output_vis.cpu().detach()
                    fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                    fft_output_vis_rgb[0] = fft_output_vis[0]
                    fft_output_vis_rgb[1] = (
                        fft_output_vis[1] + fft_output_vis[2]
                    ) * 0.5
                    fft_output_vis_rgb[2] = fft_output_vis[3]

                    # for i in range(3):
                    #     fft_output_vis_rgb[i] = (
                    #         fft_output_vis_rgb[i] - fft_output_vis_rgb[i].min()
                    #     ) / (fft_output_vis_rgb[i].max() - fft_output_vis_rgb[i].min())

                    fft_output_vis_rgb = (
                        fft_output_vis_rgb - fft_output_vis_rgb.min()
                    ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                    fft_output_vis_rgb = fft_output_vis_rgb.permute(1, 2, 0).numpy()

                    # Dump to output folder
                    # Phase and amplitude are nested
                    name = filename[e].replace(".JPEG", ".png")
                    parent, name = name.split("/")
                    path = test_path / parent
                    path.mkdir(exist_ok=True, parents=True)
                    path_output = path / ("output_" + name)
                    path_fft = path / ("fft_" + name)

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )
                    cv2.imwrite(
                        str(path_fft),
                        (fft_output_vis_rgb[:, :, ::-1] * 255.0).astype(np.int),
                    )

                pbar.update(args.batch_size)
                pbar.set_description(f"Test Epoch : {start_epoch} Step: {global_step}")
