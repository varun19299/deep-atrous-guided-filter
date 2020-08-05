# Libraries
from utils.model_serialization import load_state_dict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.adamw import AdamW

# Torch Libs
import torch
import logging
import torch.distributed as dist

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def reduce_loss_dict(
    loss_dict: "Dict[str,Tensor]", world_size: int
) -> "Dict[str,Tensor]":
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    if world_size < 2:
        return {k: v.item() for k, v in loss_dict.items()}

    with torch.no_grad():
        loss_names = []
        all_losses = []

        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)

        dist.reduce(all_losses, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}

    return reduced_losses


def pprint_args(args: "tupperware"):
    """
    Pretty print args
    """
    string = str(args)
    string_ll = string.replace("Tupperware(", "").rstrip(")").split(", ")
    string_ll = sorted(string_ll, key=lambda x: x.split("=")[0].lower())

    string_ll = [
        f"*{line.split('=')[0]}* = {line.split('=')[-1]}" for line in string_ll
    ]
    string = "\n".join(string_ll)

    return string


def get_optimisers(G: "nn.Module", args: "tupperware") -> "Union[optim, lr_scheduler]":

    g_optimizer = AdamW(
        G.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2)
    )

    g_lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=g_optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=2e-10
    )

    return g_optimizer, g_lr_scheduler


def load_models(
    G: "nn.Module" = None,
    g_optimizer: "optim" = None,
    args: "tupperware" = None,
    tag: str = "latest",
    is_local_rank_0: bool = True,
) -> "Union[nn.Module, optim, int, int, int]":

    latest_path = args.ckpt_dir / args.save_filename_latest_G
    best_path = args.ckpt_dir / args.save_filename_G

    if tag == "latest":
        path = latest_path
        if not path.exists():
            path = best_path
            tag = "best"

    elif tag == "best":
        path = best_path
        if not path.exists():
            path = latest_path
            tag = "latest"

    # Defaults
    start_epoch = 0
    global_step = 0
    loss = 1e6

    if args.resume:
        if path.is_file():
            checkpoint = torch.load(path, map_location=torch.device("cpu"))

            if is_local_rank_0:
                logging.info(f"Loading checkpoint from {path} with tag {tag}.")
            load_state_dict(G, checkpoint["state_dict"])
            # G.load_state_dict(checkpoint["state_dict"])

            if not args.finetune:
                if g_optimizer and "optimizer" in checkpoint:
                    g_optimizer.load_state_dict(checkpoint["optimizer"])

                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] - 1

                if "global_step" in checkpoint:
                    global_step = checkpoint["global_step"]

                if "loss" in checkpoint:
                    loss = checkpoint["loss"]

                    if is_local_rank_0:
                        logging.info(f"Model has loss of {loss}")

        else:
            if is_local_rank_0:
                logging.info(f"No checkpoint found  at {path} with tag {tag}.")

    return G, g_optimizer, global_step, start_epoch, loss


def save_weights(
    global_step: int,
    epoch: int,
    G: "nn.Module" = None,
    g_optimizer: "optim" = None,
    loss: "float" = None,
    is_min: bool = True,
    args: "tupperware" = None,
    tag: str = "latest",
    is_local_rank_0: bool = True,
):
    if is_min or tag == "latest":
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} saving weights")

        # Gen
        G_state = {
            "global_step": global_step,
            "epoch": epoch + 1,
            "state_dict": G.state_dict(),
            "optimizer": g_optimizer.state_dict(),
            "loss": loss,
        }
        save_filename_G = (
            args.save_filename_latest_G if tag == "latest" else args.save_filename_G
        )

        path_G = str(args.ckpt_dir / save_filename_G)

        torch.save(G_state, path_G)

        # Specific saving
        if tag == "latest":
            for i in range(1, args.save_num_snapshots + 1):
                if (epoch + i) % args.save_copy_every_epochs == 0:
                    save_filename_G = f"Epoch_{epoch}_{save_filename_G}"

                    path_G = str(args.ckpt_dir / args.exp_name / save_filename_G)

                    torch.save(G_state, path_G)

    else:
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} NOT saving weights")


class SmoothenValue(object):
    "Create a smooth moving average for a value (loss, etc) using `beta`."

    def __init__(self, beta: float = 0.9):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float):
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


# Dictionary based loss collectors
# See train.py for usage
class AvgLoss_with_dict(object):

    """
    Utility class for storing running averages of losses
    """

    def __init__(self, loss_dict: "Dict", args: "tupperware" = None, count: int = 0):
        self.args = args
        self.count = count
        self.loss_dict = loss_dict

    def reset(self):
        self.count = 0
        for k in self.loss_dict:
            self.loss_dict[k] = 0.0

    def __add__(self, loss_dict: "Dict"):
        self.count += 1

        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"

        for k in self.loss_dict:
            self.loss_dict[k] += (loss_dict[k] - self.loss_dict[k]) / self.count

        return self


class ExpLoss_with_dict(object):
    def __init__(self, loss_dict: "Dict", args: "tupperware" = None):
        """
        :param dict: Expects default dict
        """
        self.args = args
        self.loss_dict = loss_dict
        self.set_collector()

    def set_collector(self):
        self.collector_dict = {}
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"] = SmoothenValue()

    def __add__(self, loss_dict: "Dict"):
        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"].add_value(loss_dict[k])
            self.loss_dict[k] = self.collector_dict[k + "_collector"].smooth

        return self
