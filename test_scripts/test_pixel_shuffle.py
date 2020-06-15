"""
Val Script for Phase/Amp mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
import numpy as np
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
from config import initialise


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
    args.batch_size = 1

    # Set device, init dirs
    device, source_device = set_device(args)
    dir_init(args)

    # Get data
    data = get_dataloaders(args)

    # Model
    G, FFT, _ = get_model.model(args)
    G = G.to(device)
    FFT = FFT.to(device)

    diff_gain = 0.0380 / torch.tensor([0.0380, 0.0375, 0.0367, 0.0378])
    diff_gain = diff_gain.reshape(1, 4, 1, 1).to(device)

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

    logging.info(
        f"Loaded experiment {args.exp_name}, dataset {args.dataset_name}, trained for {start_epoch} epochs."
    )

    # test path
    test_path = (
        args.output_dir / f"test_{args.inference_mode}_tag_{args.tag}_gain_{args.gain}"
    )
    test_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        G.eval()
        FFT.eval()

        if data.test_loader:
            pbar = tqdm(
                range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
            )
            for i, batch in enumerate(data.test_loader):
                source, filename = batch
                source = source.to(device)

                parent, name = filename[0].split("/")
                if (
                    args.test_skip_existing
                    and (test_path / f"{parent}/output_{name}").exists()
                ):
                    logging.info(f"Skipping {name}")
                    pbar.update(args.batch_size)
                    continue

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
