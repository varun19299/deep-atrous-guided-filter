"""
Train Script for Phase Mask and Amplitude Mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
from math import ceil

# Torch Libs
import torch
import torch.nn.functional as F

# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from models import get_model
from loss import GLoss, DLoss
from config import initialise
from metrics import PSNR

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.train_helper import (
    set_device,
    get_optimisers,
    load_models,
    save_weights,
    ExpLoss_with_dict,
    AvgLoss_with_dict,
    pprint_args,
)
from utils.ops import unpixel_shuffle
from utils.tupperware import tupperware

# Experiment, add any observers by command line
ex = Experiment("Pixelshuffle-Train")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    # Set device, init dirs
    device, source_device = set_device(args)
    dir_init(args)

    # Get data
    data = get_dataloaders(args)

    # Model
    # d_source_device = torch.device("cuda:2")
    G, FFT, D = get_model.model(
        args
    )  # , source_device=source_device, target_device=device)
    G = G.to(device)
    FFT = FFT.to(device)
    D = D.to(source_device)

    # Optimisers
    (g_optimizer, fft_optimizer, d_optimizer), (
        g_lr_scheduler,
        fft_lr_scheduler,
        d_lr_scheduler,
    ) = get_optimisers(G, FFT, D, args)

    # Load Models
    (G, FFT, D), (
        g_optimizer,
        fft_optimizer,
        d_optimizer,
    ), global_step, start_epoch, loss = load_models(
        G, FFT, D, g_optimizer, fft_optimizer, d_optimizer, args
    )

    if args.dataparallel:
        G = torch.nn.DataParallel(G, device_ids=args.device_list)
        FFT = torch.nn.DataParallel(FFT, device_ids=args.device_list)
        D = torch.nn.DataParallel(D, device_ids=args.device_list)

    writer = SummaryWriter(log_dir=str(args.run_dir / args.exp_name))
    writer.add_text("Args", pprint_args(args))

    # Initialise losses
    g_loss = GLoss(
        args, device, lpips_device=source_device, perception_device=source_device
    )
    d_loss = DLoss(args)

    # Compatibility with checkpoints without global_step
    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size

    start_epoch = global_step // len(data.train_loader.dataset)

    # Exponential averaging of loss
    loss_dict = {
        "g_loss": 0.0,
        "d_loss": 0.0,
        "adversarial_loss": 0.0,
        "perception_loss": 0.0,
        "contextual_loss": 0.0,
        "image_loss": 0.0,
        "lpips_loss": 0.0,
        "ssim_loss": 0.0,
        "edge_loss": 0.0,
        "forward_loss": 0.0,
        "interm_loss": 0.0,
        "train_PSNR": 0.0,
        "fft_grad_norm": 0.0,
    }

    metric_dict = {"PSNR": 0.0, "LPIPS": 0.0, "g_loss": 0.0}
    avg_metrics = AvgLoss_with_dict(loss_dict=metric_dict, args=args)
    exp_loss = ExpLoss_with_dict(loss_dict=loss_dict, args=args)

    try:

        for epoch in range(start_epoch, args.num_epochs):
            # Train mode
            G.train()
            FFT.train()
            D.train()

            pbar = tqdm(
                range(len(data.train_loader) * args.batch_size), dynamic_ncols=True
            )
            for i, batch in enumerate(data.train_loader):
                # allows for interrupted training
                if (
                    (global_step + 1) % (len(data.train_loader) * args.batch_size) == 0
                ) and (epoch == start_epoch):
                    break

                loss_dict = defaultdict(float)

                source, target, filename = batch
                source, target = (source.to(device), target.to(device))

                # Get multistage source, targets
                target_ll = [target]

                for j in range(1, args.scales):
                    target_ll.append(
                        F.interpolate(
                            target,
                            size=(
                                ceil(target.shape[2] / pow(2, j)),
                                ceil(target.shape[3] / pow(2, j)),
                            ),
                            mode="bilinear",
                            align_corners=True,
                        )
                    )

                target_ll = target_ll[::-1]

                target_12_channel_ll = []
                for j in range(args.scales):
                    n, c, h, w = target_ll[j].shape
                    target_12_channel_ll.append(
                        unpixel_shuffle(target_ll[j], 2).reshape(
                            n * args.pixelshuffle_ratio ** 2,
                            c,
                            h // args.pixelshuffle_ratio,
                            w // args.pixelshuffle_ratio,
                        )
                    )

                if epoch >= args.pretrain_epochs:
                    # ------------------------------- #
                    # Update Disc
                    # ------------------------------- #
                    D.zero_grad()
                    G.zero_grad()
                    FFT.zero_grad()

                    fft_output = FFT(source)

                    # Unpixelshuffle
                    fft_output_16_channel = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )

                    output_12_channel_ll = G(fft_output_16_channel)

                    for j in range(args.scales):
                        n, c, h, w = output_12_channel_ll[j].shape
                        output_12_channel_ll[j] = output_12_channel_ll[j].reshape(
                            n * args.pixelshuffle_ratio ** 2,
                            c // args.pixelshuffle_ratio ** 2,
                            h,
                            w,
                        )

                    output_12_channel_ll = [
                        output.detach() for output in output_12_channel_ll
                    ]

                    real_logit_ll = D(target_12_channel_ll)
                    fake_logit_ll = D(output_12_channel_ll)

                    d_loss(real_logit_ll=real_logit_ll, fake_logit_ll=fake_logit_ll)

                    d_loss.total_loss.backward()

                    loss_dict["d_loss"] += d_loss.total_loss.item()
                    d_optimizer.step()
                else:
                    loss_dict["d_loss"] += 0.0

                # ------------------------------- #
                # Update Gen
                # ------------------------------- #
                G.zero_grad()
                FFT.zero_grad()

                fft_output = FFT(source)
                # Unpixelshuffle
                fft_output_16_channel = unpixel_shuffle(
                    fft_output, args.pixelshuffle_ratio
                )
                output_12_channel_ll = G(fft_output_16_channel)

                for j in range(args.scales):
                    n, c, h, w = output_12_channel_ll[j].shape
                    output_12_channel_ll[j] = output_12_channel_ll[j].reshape(
                        n * args.pixelshuffle_ratio ** 2,
                        c // args.pixelshuffle_ratio ** 2,
                        h,
                        w,
                    )

                if epoch >= args.pretrain_epochs:
                    fake_logit_ll = D(output_12_channel_ll)
                    real_logit_ll = D(target_12_channel_ll)

                    g_loss(
                        fake_logit_ll=fake_logit_ll,
                        real_logit_ll=real_logit_ll,
                        output_ll=output_12_channel_ll,
                        target_ll=target_12_channel_ll,
                        fft_output=fft_output,
                    )
                else:
                    g_loss(
                        output_ll=output_12_channel_ll,
                        target_ll=target_12_channel_ll,
                        fft_output=fft_output,
                        pretrain=True,
                    )

                g_loss.total_loss.backward()

                output_ll = []
                for j in range(args.scales):
                    n, c, h, w = output_12_channel_ll[j].shape
                    output_12_channel_ll[j] = output_12_channel_ll[j].reshape(
                        n // args.pixelshuffle_ratio ** 2,
                        c * args.pixelshuffle_ratio ** 2,
                        h,
                        w,
                    )
                    output_ll.append(
                        F.pixel_shuffle(
                            output_12_channel_ll[j], args.pixelshuffle_ratio
                        )
                    )

                # Train PSNR
                loss_dict["train_PSNR"] += PSNR(output_ll[-1], target_ll[-1])

                # Gradient norm of FFT Layer
                if args.fft_epochs != args.num_epochs:
                    if args.fft_init == "wiener_crop":
                        if args.dataparallel:
                            loss_dict[
                                "fft_grad_norm"
                            ] += FFT.module.wiener_crop.grad.norm()
                        else:
                            loss_dict["fft_grad_norm"] += FFT.wiener_crop.grad.norm()
                    elif args.fft_init in [
                        "wiener_crop",
                        "wiener_smaller",
                        "transpose",
                        "random",
                    ]:
                        if args.dataparallel:
                            loss_dict[
                                "fft_grad_norm"
                            ] += FFT.module.fft_layer.grad.norm()
                        else:
                            loss_dict["fft_grad_norm"] += FFT.fft_layer.grad.norm()
                    else:
                        loss_dict["fft_grad_norm"] += 0.0
                else:
                    loss_dict["fft_grad_norm"] += 0.0

                g_optimizer.step()
                # Update lr schedulers
                g_lr_scheduler.step(epoch + i / len(data.train_loader))

                if epoch >= args.fft_epochs:
                    fft_optimizer.step()
                    fft_lr_scheduler.step(
                        epoch - args.fft_epochs + i / len(data.train_loader)
                    )

                if epoch >= args.pretrain_epochs:
                    d_lr_scheduler.step(
                        epoch - args.pretrain_epochs + i / len(data.train_loader)
                    )

                # Accumulate all losses
                loss_dict["g_loss"] += g_loss.total_loss.item()
                loss_dict["adversarial_loss"] += g_loss.adversarial_loss.item()
                loss_dict["perception_loss"] += g_loss.perception_loss.item()
                loss_dict["contextual_loss"] += g_loss.contextual_loss.item()
                loss_dict["image_loss"] += g_loss.image_loss.item()
                loss_dict["lpips_loss"] += g_loss.lpips_loss.item()
                loss_dict["ssim_loss"] += g_loss.ssim_loss.item()
                loss_dict["edge_loss"] += g_loss.edge_loss.item()
                loss_dict["forward_loss"] += g_loss.forward_loss.item()
                loss_dict["interm_loss"] += g_loss.interm_loss.item()

                exp_loss += loss_dict
                pbar.update(args.batch_size)
                global_step += args.batch_size

                pretrain_str = ""
                if epoch < args.pretrain_epochs:
                    pretrain_str = "Pretrain"
                pbar.set_description(
                    f"Epoch {pretrain_str}: {epoch + 1} | Gen loss: {exp_loss.loss_dict['g_loss']:.3f} "
                )

                if i % 10 == 0:
                    gen_lr = g_optimizer.param_groups[0]["lr"]
                    writer.add_scalar("lr/gen", gen_lr, global_step)

                    if epoch >= args.fft_epochs:
                        fft_lr = fft_optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr/fft", fft_lr, global_step)

                    if epoch >= args.pretrain_epochs:
                        disc_lr = d_optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr/disc", disc_lr, global_step)

                    for metric in exp_loss.loss_dict:
                        writer.add_scalar(
                            f"Train_Metrics/{metric}",
                            exp_loss.loss_dict[metric],
                            global_step,
                        )

                # Display images
                if i % (args.log_interval) == 0:
                    n = np.min([3, args.batch_size])
                    for e in range(n):
                        source_vis = torch.zeros_like(source[e])[:3]
                        source_vis[0] = source[e][0]
                        source_vis[1] = 0.5 * (source[e][1] + source[e][2])
                        source_vis[2] = source[e][3]
                        source_vis = source_vis.mul(0.5).add(0.5)

                        writer.add_image(
                            f"Source/Train_{e+1}",
                            source_vis.cpu().detach(),
                            global_step,
                        )

                        if len(fft_output):
                            fft_output_vis = fft_output[e]
                            fft_output_vis = fft_output_vis.cpu().detach()
                            fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                            fft_output_vis_rgb[0] = fft_output_vis[0]
                            fft_output_vis_rgb[1] = (
                                fft_output_vis[1] + fft_output_vis[2]
                            ) * 0.5
                            fft_output_vis_rgb[2] = fft_output_vis[3]

                            fft_output_vis_rgb = (
                                fft_output_vis_rgb - fft_output_vis_rgb.min()
                            ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                            writer.add_image(
                                f"FFT/Train_{e+1}", fft_output_vis_rgb, global_step
                            )

                        writer.add_text(
                            f"Filename/Train_{e+1}", filename[e], global_step
                        )

                        for scale in range(len(target_ll)):
                            target_vis = target_ll[scale][e].mul(0.5).add(0.5)
                            output_vis = output_ll[scale][e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"Target/Train_{e+1}_scale_{scale}",
                                target_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Output/Train_{e+1}_scale_{scale}",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                # Save checkpoint
                if i % (args.log_interval * 20) == 0:

                    logging.info(
                        f"Saving weights at epoch {epoch + 1} global step {global_step}"
                    )

                    # Save weights
                    save_weights(
                        epoch=epoch,
                        global_step=global_step,
                        G=G,
                        D=D,
                        FFT=FFT,
                        g_optimizer=g_optimizer,
                        d_optimizer=d_optimizer,
                        fft_optimizer=fft_optimizer,
                        loss=loss,
                        tag="latest",
                        args=args,
                    )

            pbar = tqdm(
                range(len(data.val_loader) * args.batch_size), dynamic_ncols=True
            )

            with torch.no_grad():
                G.eval()
                FFT.eval()
                D.eval()
                filename_static = []
                avg_metrics.reset()

                for i, batch in enumerate(data.val_loader):
                    metrics_dict = defaultdict(float)

                    source, target, filename = batch
                    source, target = (source.to(device), target.to(device))

                    # Get multistage source, targets
                    target_ll = [target]

                    for j in range(1, args.scales):
                        target_ll.append(
                            F.interpolate(
                                target,
                                size=(
                                    ceil(target.shape[2] / pow(2, j)),
                                    ceil(target.shape[3] / pow(2, j)),
                                ),
                                mode="bilinear",
                                align_corners=True,
                            )
                        )

                    target_ll = target_ll[::-1]

                    target_12_channel_ll = []
                    for j in range(args.scales):
                        n, c, h, w = target_ll[j].shape
                        target_12_channel_ll.append(
                            unpixel_shuffle(target_ll[j], 2).reshape(
                                n * args.pixelshuffle_ratio ** 2,
                                c,
                                h // args.pixelshuffle_ratio,
                                w // args.pixelshuffle_ratio,
                            )
                        )

                    fft_output = FFT(source)
                    # Unpixelshuffle
                    fft_output_16_channel = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_12_channel_ll = G(fft_output_16_channel)

                    for j in range(args.scales):
                        n, c, h, w = output_12_channel_ll[j].shape
                        output_12_channel_ll[j] = output_12_channel_ll[j].reshape(
                            n * args.pixelshuffle_ratio ** 2,
                            c // args.pixelshuffle_ratio ** 2,
                            h,
                            w,
                        )

                    if epoch >= args.pretrain_epochs:
                        fake_logit_ll = D(output_12_channel_ll)
                        real_logit_ll = D(target_12_channel_ll)

                        g_loss(
                            fake_logit_ll=fake_logit_ll,
                            real_logit_ll=real_logit_ll,
                            output_ll=output_12_channel_ll,
                            target_ll=target_12_channel_ll,
                            fft_output=fft_output,
                        )
                    else:
                        g_loss(
                            output_ll=output_12_channel_ll,
                            target_ll=target_12_channel_ll,
                            fft_output=fft_output,
                            pretrain=True,
                        )

                    metrics_dict["g_loss"] += g_loss.total_loss.item()

                    output_ll = []
                    for j in range(args.scales):
                        n, c, h, w = output_12_channel_ll[j].shape
                        output_12_channel_ll[j] = output_12_channel_ll[j].reshape(
                            n // args.pixelshuffle_ratio ** 2,
                            c * args.pixelshuffle_ratio ** 2,
                            h,
                            w,
                        )
                        output_ll.append(
                            F.pixel_shuffle(
                                output_12_channel_ll[j], args.pixelshuffle_ratio
                            )
                        )
                    output = output_ll[-1]  # Take only last scale

                    # Save image
                    if args.static_val_image in filename:
                        filename_static = filename
                        source_static = source
                        fft_output_static = fft_output
                        target_static = target
                        output_static = output

                    # PSNR
                    metrics_dict["PSNR"] += PSNR(output, target)

                    if args.lambda_lpips:
                        metrics_dict["LPIPS"] += (
                            g_loss.lpips_model.forward(
                                output.to(g_loss.lpips_device),
                                target.to(g_loss.lpips_device),
                            )
                            .mean()
                            .item()
                        )
                    else:
                        metrics_dict["LPIPS"] += 0.0

                    avg_metrics += metrics_dict
                    pbar.update(args.batch_size)
                    pbar.set_description(
                        f"Val Epoch : {epoch + 1} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f}"
                    )

                for metric in avg_metrics.loss_dict:
                    writer.add_scalar(
                        f"Val_Metrics/{metric}",
                        avg_metrics.loss_dict[metric],
                        global_step,
                    )

                n = np.min([3, args.batch_size])
                for e in range(n):
                    source_vis = torch.zeros_like(source[e])[:3]
                    source_vis[0] = source[e][0]
                    source_vis[1] = 0.5 * (source[e][1] + source[e][2])
                    source_vis[2] = source[e][3]
                    source_vis = source_vis.mul(0.5).add(0.5)

                    target_vis = target[e].mul(0.5).add(0.5)
                    output_vis = output[e].mul(0.5).add(0.5)

                    if len(fft_output):
                        fft_output_vis = fft_output[e]
                        fft_output_vis = fft_output_vis.cpu().detach()
                        fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                        fft_output_vis_rgb[0] = fft_output_vis[0]
                        fft_output_vis_rgb[1] = (
                            fft_output_vis[1] + fft_output_vis[2]
                        ) * 0.5
                        fft_output_vis_rgb[2] = fft_output_vis[3]

                        fft_output_vis_rgb = (
                            fft_output_vis_rgb - fft_output_vis_rgb.min()
                        ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                        writer.add_image(
                            f"FFT/Val_{e+1}", fft_output_vis_rgb, global_step
                        )

                    writer.add_image(
                        f"Source/Val_{e+1}", source_vis.cpu().detach(), global_step
                    )
                    writer.add_image(
                        f"Target/Val_{e+1}", target_vis.cpu().detach(), global_step
                    )
                    writer.add_image(
                        f"Output/Val_{e+1}", output_vis.cpu().detach(), global_step
                    )

                    writer.add_text(f"Filename/Val_{e + 1}", filename[e], global_step)

                for e, filename in enumerate(filename_static):
                    if filename == args.static_val_image:
                        source_vis = torch.zeros_like(source_static[e])[:3]
                        source_vis[0] = source_static[e][0]
                        source_vis[1] = 0.5 * (
                            source_static[e][1] + source_static[e][2]
                        )
                        source_vis[2] = source_static[e][3]
                        source_vis = source_vis.mul(0.5).add(0.5)

                        target_vis = target_static[e].mul(0.5).add(0.5)
                        output_vis = output_static[e].mul(0.5).add(0.5)

                        if len(fft_output):
                            fft_output_vis = fft_output_static[e]
                            fft_output_vis = fft_output_vis.cpu().detach()
                            fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                            fft_output_vis_rgb[0] = fft_output_vis[0]
                            fft_output_vis_rgb[1] = (
                                fft_output_vis[1] + fft_output_vis[2]
                            ) * 0.5
                            fft_output_vis_rgb[2] = fft_output_vis[3]

                            fft_output_vis_rgb = (
                                fft_output_vis_rgb - fft_output_vis_rgb.min()
                            ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                            writer.add_image(
                                f"FFT/Val_Static", fft_output_vis_rgb, global_step
                            )

                        writer.add_image(
                            f"Source/Val_Static", source_vis.cpu().detach(), global_step
                        )
                        writer.add_image(
                            f"Target/Val_Static", target_vis.cpu().detach(), global_step
                        )
                        writer.add_image(
                            f"Output/Val_Static", output_vis.cpu().detach(), global_step
                        )

                        writer.add_text(
                            f"Filename/Val_Static", filename[e], global_step
                        )

                        break

                logging.info(
                    f"Saving weights at END OF epoch {epoch + 1} global step {global_step}"
                )

                # Save weights
                if avg_metrics.loss_dict["g_loss"] < loss:
                    is_min = True
                    loss = avg_metrics.loss_dict["g_loss"]
                else:
                    is_min = False

                # Save weights
                save_weights(
                    epoch=epoch,
                    global_step=global_step,
                    G=G,
                    FFT=FFT,
                    D=D,
                    g_optimizer=g_optimizer,
                    d_optimizer=d_optimizer,
                    fft_optimizer=fft_optimizer,
                    loss=loss,
                    is_min=is_min,
                    args=args,
                    tag="best",
                )

                # Test
                if not data.test_loader:
                    continue

                pbar = tqdm(
                    range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
                )

                filename_static = []

                for i, batch in enumerate(data.test_loader):
                    source, filename = batch
                    source = source.to(device)

                    fft_output = FFT(source)
                    # Unpixelshuffle
                    fft_output_16_channel = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_12_channel_ll = G(fft_output_16_channel)

                    output_ll = []
                    for j in range(args.scales):
                        output_ll.append(
                            F.pixel_shuffle(
                                output_12_channel_ll[j], args.pixelshuffle_ratio
                            )
                        )
                    output = output_ll[-1]  # Take only last scale

                    # Save image
                    if args.static_test_image in filename:
                        filename_static = filename
                        source_static = source
                        fft_output_static = fft_output
                        output_static = output

                    pbar.update(args.batch_size)
                    pbar.set_description(
                        f"Test Epoch : {epoch + 1} Step: {global_step}"
                    )

                n = np.min([3, args.batch_size])
                for e in range(n):
                    source_vis = torch.zeros_like(source[e])[:3]
                    source_vis[0] = source[e][0]
                    source_vis[1] = 0.5 * (source[e][1] + source[e][2])
                    source_vis[2] = source[e][3]
                    source_vis = source_vis.mul(0.5).add(0.5)

                    output_vis = output[e].mul(0.5).add(0.5)

                    if len(fft_output):
                        fft_output_vis = fft_output[e]
                        fft_output_vis = fft_output_vis.cpu().detach()
                        fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                        fft_output_vis_rgb[0] = fft_output_vis[0]
                        fft_output_vis_rgb[1] = (
                            fft_output_vis[1] + fft_output_vis[2]
                        ) * 0.5
                        fft_output_vis_rgb[2] = fft_output_vis[3]

                        fft_output_vis_rgb = (
                            fft_output_vis_rgb - fft_output_vis_rgb.min()
                        ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                        writer.add_image(
                            f"FFT/Test_{e+1}", fft_output_vis_rgb, global_step
                        )

                    writer.add_image(
                        f"Source/Test_{e+1}", source_vis.cpu().detach(), global_step
                    )

                    writer.add_image(
                        f"Output/Test_{e+1}", output_vis.cpu().detach(), global_step
                    )

                    writer.add_text(f"Filename/Test_{e + 1}", filename[e], global_step)

                for e, filename in enumerate(filename_static):
                    if filename == args.static_test_image:
                        source_vis = torch.zeros_like(source_static[e])[:3]
                        source_vis[0] = source_static[e][0]
                        source_vis[1] = 0.5 * (
                            source_static[e][1] + source_static[e][2]
                        )
                        source_vis[2] = source_static[e][3]
                        source_vis = source_vis.mul(0.5).add(0.5)
                        output_vis = output_static[e].mul(0.5).add(0.5)

                        if len(fft_output):
                            fft_output_vis = fft_output_static[e]
                            fft_output_vis = fft_output_vis.cpu().detach()
                            fft_output_vis_rgb = torch.zeros_like(fft_output_vis)[:3]
                            fft_output_vis_rgb[0] = fft_output_vis[0]
                            fft_output_vis_rgb[1] = (
                                fft_output_vis[1] + fft_output_vis[2]
                            ) * 0.5
                            fft_output_vis_rgb[2] = fft_output_vis[3]

                            fft_output_vis_rgb = (
                                fft_output_vis_rgb - fft_output_vis_rgb.min()
                            ) / (fft_output_vis_rgb.max() - fft_output_vis_rgb.min())

                            writer.add_image(
                                f"FFT/Test_Static", fft_output_vis_rgb, global_step
                            )

                        writer.add_image(
                            f"Source/Test_Static",
                            source_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_image(
                            f"Output/Test_Static",
                            output_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_text(
                            f"Filename/Test_Static", filename[e], global_step
                        )

                        break

    except KeyboardInterrupt:
        logging.info("-" * 89)
        logging.info("Exiting from training early. Saving models")

        save_weights(
            epoch=epoch,
            global_step=global_step,
            G=G,
            D=D,
            FFT=FFT,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            fft_optimizer=fft_optimizer,
            loss=loss,
            is_min=True,
            args=args,
        )
