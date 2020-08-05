"""
Train Script
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
import os
import sys

# Torch Libs
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

# Ignore warnings
import warnings

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
    get_optimisers,
    reduce_loss_dict,
    load_models,
    save_weights,
    ExpLoss_with_dict,
    AvgLoss_with_dict,
    pprint_args,
)
from utils.tupperware import tupperware

# Experiment, add any observers by command line
ex = Experiment("Train")
ex = initialise(ex)

# local rank 0: for logging, saving ckpts
if "LOCAL_RANK" in os.environ:
    is_local_rank_0 = int(os.environ["LOCAL_RANK"]) == 0
else:
    is_local_rank_0 = True
if not is_local_rank_0:
    sys.stdout = open(os.devnull, "w")

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    # Dir init
    dir_init(args, is_local_rank_0=is_local_rank_0)

    # Ignore warnings
    if not is_local_rank_0:
        warnings.filterwarnings("ignore")

    # Mutli GPUS Setup
    if args.distdataparallel:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        rank = args.device
        world_size = 1

    # Get data
    data = get_dataloaders(args, is_local_rank_0=is_local_rank_0)

    # Model
    G = get_model.model(args).to(rank)

    # Optimisers
    g_optimizer, g_lr_scheduler = get_optimisers(G, args)

    # Load Models
    G, g_optimizer, global_step, start_epoch, loss = load_models(
        G, g_optimizer, args, is_local_rank_0=is_local_rank_0
    )

    if args.distdataparallel:
        # Wrap with Distributed Data Parallel
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[rank], output_device=rank
        )

    # Log no of GPUs
    if is_local_rank_0:
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        logging.info("Using {} GPUs".format(world_size))

        writer = SummaryWriter(log_dir=str(args.run_dir))
        writer.add_text("Args", pprint_args(args))

        # Pbars
        train_pbar = tqdm(
            range(len(data.train_loader) * args.batch_size), dynamic_ncols=True
        )

        val_pbar = (
            tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)
            if data.val_loader
            else None
        )

        test_pbar = (
            tqdm(range(len(data.test_loader) * args.batch_size), dynamic_ncols=True)
            if data.test_loader
            else None
        )

    # Initialise losses
    g_loss = GLoss(args).to(rank)

    # Compatibility with checkpoints without global_step
    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size

    start_epoch = global_step // len(data.train_loader.dataset)

    # Exponential averaging of loss
    loss_dict = {
        "total_loss": 0.0,
        "image_loss": 0.0,
        "cobi_rgb_loss": 0.0,
        "train_PSNR": 0.0,
    }

    metric_dict = {"PSNR": 0.0, "total_loss": 0.0}
    avg_metrics = AvgLoss_with_dict(loss_dict=metric_dict, args=args)
    exp_loss = ExpLoss_with_dict(loss_dict=loss_dict, args=args)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            # Train mode
            G.train()


            if is_local_rank_0:
                train_pbar.reset()

            if args.distdataparallel:
                data.train_loader.sampler.set_epoch(epoch)

            for i, batch in enumerate(data.train_loader):
                # allows for interrupted training
                if (
                    (global_step + 1) % (len(data.train_loader) * args.batch_size) == 0
                ) and (epoch == start_epoch):
                    break

                loss_dict = defaultdict(float)

                source, target, filename = batch
                source, target = (source.to(rank), target.to(rank))

                # ------------------------------- #
                # Update Gen
                # ------------------------------- #
                G.zero_grad()
                output = G(source)

                g_loss(output=output, target=target)

                g_loss.total_loss.backward()
                g_optimizer.step()

                # Update lr schedulers
                g_lr_scheduler.step(epoch + i / len(data.train_loader))

                # if is_local_rank_0:
                # Train PSNR
                loss_dict["train_PSNR"] += PSNR(output, target)

                # Accumulate all losses
                loss_dict["total_loss"] += g_loss.total_loss
                loss_dict["image_loss"] += g_loss.image_loss
                loss_dict["cobi_rgb_loss"] += g_loss.cobi_rgb_loss

                exp_loss += reduce_loss_dict(loss_dict, world_size=world_size)

                global_step += args.batch_size * world_size

                if is_local_rank_0:
                    train_pbar.update(args.batch_size)
                    train_pbar.set_description(
                        f"Epoch: {epoch + 1} | Gen loss: {exp_loss.loss_dict['total_loss']:.3f} "
                    )

                # Write lr rates and metrics
                if is_local_rank_0 and i % (args.log_interval) == 0:
                    gen_lr = g_optimizer.param_groups[0]["lr"]
                    writer.add_scalar("lr/gen", gen_lr, global_step)

                    for metric in exp_loss.loss_dict:
                        writer.add_scalar(
                            f"Train_Metrics/{metric}",
                            exp_loss.loss_dict[metric],
                            global_step,
                        )

                    # Display images at end of epoch
                    n = np.min([3, args.batch_size])
                    for e in range(n):
                        source_vis = source[e].mul(0.5).add(0.5)
                        target_vis = target[e].mul(0.5).add(0.5)
                        output_vis = output[e].mul(0.5).add(0.5)

                        writer.add_image(
                            f"Source/Train_{e + 1}",
                            source_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_image(
                            f"Target/Train_{e + 1}",
                            target_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_image(
                            f"Output/Train_{e + 1}",
                            output_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_text(
                            f"Filename/Train_{e + 1}", filename[e], global_step
                        )

            if is_local_rank_0:
                # Save ckpt at end of epoch
                logging.info(
                    f"Saving weights at epoch {epoch + 1} global step {global_step}"
                )

                # Save weights
                save_weights(
                    epoch=epoch,
                    global_step=global_step,
                    G=G,
                    g_optimizer=g_optimizer,
                    loss=loss,
                    tag="latest",
                    args=args,
                )

                train_pbar.refresh()

            # Run val and test only occasionally
            if epoch % args.val_test_epoch_interval != 0:
                continue

            # Val and test
            with torch.no_grad():
                G.eval()

                if data.val_loader:
                    avg_metrics.reset()
                    if is_local_rank_0:
                        val_pbar.reset()

                    filename_static = []

                    for i, batch in enumerate(data.val_loader):
                        metrics_dict = defaultdict(float)

                        source, target, filename = batch
                        source, target = (source.to(rank), target.to(rank))

                        output = G(source)
                        g_loss(output=output, target=target)

                        # Total loss
                        metrics_dict["total_loss"] += g_loss.total_loss
                        # PSNR
                        metrics_dict["PSNR"] += PSNR(output, target)

                        avg_metrics += reduce_loss_dict(
                            metrics_dict, world_size=world_size
                        )

                        # Save image
                        if args.static_val_image in filename:
                            filename_static = filename
                            source_static = source
                            target_static = target
                            output_static = output

                        if is_local_rank_0:
                            val_pbar.update(args.batch_size)
                            val_pbar.set_description(
                                f"Val Epoch : {epoch + 1} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f}"
                            )
                    if is_local_rank_0:
                        for metric in avg_metrics.loss_dict:
                            writer.add_scalar(
                                f"Val_Metrics/{metric}",
                                avg_metrics.loss_dict[metric],
                                global_step,
                            )

                        n = np.min([3, args.batch_size])
                        for e in range(n):
                            source_vis = source[e].mul(0.5).add(0.5)
                            target_vis = target[e].mul(0.5).add(0.5)
                            output_vis = output[e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"Source/Val_{e+1}",
                                source_vis.cpu().detach(),
                                global_step,
                            )
                            writer.add_image(
                                f"Target/Val_{e+1}",
                                target_vis.cpu().detach(),
                                global_step,
                            )
                            writer.add_image(
                                f"Output/Val_{e+1}",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_text(
                                f"Filename/Val_{e + 1}", filename[e], global_step
                            )

                        for e, name in enumerate(filename_static):
                            if name == args.static_val_image:
                                source_vis = source_static[e].mul(0.5).add(0.5)
                                target_vis = target_static[e].mul(0.5).add(0.5)
                                output_vis = output_static[e].mul(0.5).add(0.5)

                                writer.add_image(
                                    f"Source/Val_Static",
                                    source_vis.cpu().detach(),
                                    global_step,
                                )
                                writer.add_image(
                                    f"Target/Val_Static",
                                    target_vis.cpu().detach(),
                                    global_step,
                                )
                                writer.add_image(
                                    f"Output/Val_Static",
                                    output_vis.cpu().detach(),
                                    global_step,
                                )

                                writer.add_text(
                                    f"Filename/Val_Static",
                                    filename_static[e],
                                    global_step,
                                )

                                break

                        logging.info(
                            f"Saving weights at END OF epoch {epoch + 1} global step {global_step}"
                        )

                        # Save weights
                        if avg_metrics.loss_dict["total_loss"] < loss:
                            is_min = True
                            loss = avg_metrics.loss_dict["total_loss"]
                        else:
                            is_min = False

                        # Save weights
                        save_weights(
                            epoch=epoch,
                            global_step=global_step,
                            G=G,
                            g_optimizer=g_optimizer,
                            loss=loss,
                            is_min=is_min,
                            args=args,
                            tag="best",
                        )

                        val_pbar.refresh()

                # Test
                if data.test_loader:
                    filename_static = []

                    if is_local_rank_0:
                        test_pbar.reset()

                    for i, batch in enumerate(data.test_loader):
                        source, filename = batch
                        source = source.to(rank)

                        output = G(source)

                        # Save image
                        if args.static_test_image in filename:
                            filename_static = filename
                            source_static = source
                            output_static = output

                        if is_local_rank_0:
                            test_pbar.update(args.batch_size)
                            test_pbar.set_description(
                                f"Test Epoch : {epoch + 1} Step: {global_step}"
                            )

                    if is_local_rank_0:
                        n = np.min([3, args.batch_size])
                        for e in range(n):
                            source_vis = source[e].mul(0.5).add(0.5)
                            output_vis = output[e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"Source/Test_{e+1}",
                                source_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Output/Test_{e+1}",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_text(
                                f"Filename/Test_{e + 1}", filename[e], global_step
                            )

                        for e, name in enumerate(filename_static):
                            if name == args.static_test_image:
                                source_vis = source_static[e]
                                output_vis = output_static[e]

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
                                    f"Filename/Test_Static",
                                    filename_static[e],
                                    global_step,
                                )

                                break

                        test_pbar.refresh()

    except KeyboardInterrupt:
        if is_local_rank_0:
            logging.info("-" * 89)
            logging.info("Exiting from training early. Saving models")

            for pbar in [train_pbar, val_pbar, test_pbar]:
                if pbar:
                    pbar.refresh()

            save_weights(
                epoch=epoch,
                global_step=global_step,
                G=G,
                g_optimizer=g_optimizer,
                loss=loss,
                is_min=True,
                args=args,
            )
