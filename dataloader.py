"""
Dataloaders
"""

# Libs
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
from sacred import Experiment

# Torch modules
from torch.utils.data import DataLoader, Dataset
import torch
import torch.distributed as dist
import cv2
from config import initialise
import random

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("data")
ex = initialise(ex)


@dataclass
class Data:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class OLEDDataset(Dataset):
    """
    Assume folders have images one level deep
    """

    def __init__(
        self,
        args,
        mode: str = "train",
        max_len: int = None,
        is_local_rank_0: bool = True,
    ):
        super(OLEDDataset, self).__init__()

        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.args = args

        if self.mode == "train":
            self.source_dir = args.train_source_dir
            self.target_dir = args.train_target_dir

        elif self.mode == "val":
            self.source_dir = args.val_source_dir
            self.target_dir = args.val_target_dir

        elif self.mode == "test":
            self.source_dir = args.test_source_dir
            self.target_dir = None

        self.max_len = max_len
        self.source_paths, self.target_paths = self._load_dataset()

        if is_local_rank_0:
            logging.info(
                f"{mode.capitalize()} Set | Source Dir: {self.source_dir} | Target Dir: {self.target_dir}"
            )
        self.is_local_rank_0 = is_local_rank_0

    def _load_dataset(self, glob_str="*.png"):

        if self.source_dir:
            if args.use_source_npy:
                source_paths = list(self.source_dir.glob("*.npy"))[: self.max_len]
            else:
                source_paths = list(self.source_dir.glob(glob_str))[: self.max_len]
        else:
            source_paths = []

        if self.target_dir:
            target_paths = [self.target_dir / file.name for file in source_paths]
        else:
            target_paths = []

        return source_paths, target_paths

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_path = self.source_paths[index]

        if args.use_source_npy:
            source = np.load(source_path)
            source = torch.tensor(source).float()
            if self.mode == "train":
                target_path = self.target_paths[index]
                target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0
                target = cv2.resize(
                    target, (self.args.image_width, self.args.image_height)
                )

            elif self.mode == "val":
                target_path = self.target_paths[index]
                target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

            if self.mode in ["train", "val"]:
                target = torch.tensor(target).float().permute(2, 0, 1)
                target = (target - 0.5) * 2

                return source, target, source_path.name

            else:
                return source, source_path.name

        if self.mode == "train":
            target_path = self.target_paths[index]
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0
            target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

            source = cv2.resize(source, (self.args.image_width, self.args.image_height))
            target = cv2.resize(target, (self.args.image_width, self.args.image_height))

            # Data augmentation
            if self.args.do_augment:
                # Vertical flip
                if random.random() < 0.25:
                    source = source[::-1]
                    target = target[::-1]

                # Horz flip
                if random.random() < 0.25:
                    source = source[:, ::-1]
                    target = target[:, ::-1]

                # 180 rotate
                if random.random() < 0.25:
                    source = cv2.rotate(source, cv2.ROTATE_180)
                    target = cv2.rotate(target, cv2.ROTATE_180)

                source = cv2.resize(
                    source, (self.args.image_width, self.args.image_height)
                )
                target = cv2.resize(
                    target, (self.args.image_width, self.args.image_height)
                )

        elif self.mode == "val":
            target_path = self.target_paths[index]
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0
            target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

        elif self.mode == "test":
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0

        source = torch.tensor(source).float().permute(2, 0, 1)
        source = (source - 0.5) * 2

        if self.mode in ["train", "val"]:
            target = torch.tensor(target).float().permute(2, 0, 1)
            target = (target - 0.5) * 2

            return (source, target, source_path.name)

        else:
            return (source, source_path.name)


def get_dataloaders(args, is_local_rank_0: bool = True):
    """
    Get dataloaders for train and val

    Returns:
    :data
    """
    train_dataset = OLEDDataset(args, mode="train", is_local_rank_0=is_local_rank_0)
    val_dataset = OLEDDataset(args, mode="val", is_local_rank_0=is_local_rank_0)
    test_dataset = OLEDDataset(args, mode="test", is_local_rank_0=is_local_rank_0)

    if is_local_rank_0:
        logging.info(
            f"Dataset Train: {len(train_dataset)} Val: {len(val_dataset)}  Test: {len(test_dataset)}"
        )

    train_loader = None
    val_loader = None
    test_loader = None

    if args.tpu_distributed:
        import torch_xla.core.xla_model as xm

    if len(train_dataset):
        if args.distdataparallel:
            if args.tpu_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True,
                )

            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=dist.get_world_size(), shuffle=True
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_threads,
                pin_memory=False,
                drop_last=True,
                sampler=train_sampler,
            )
        else:

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_threads,
                pin_memory=False,
                drop_last=True,
                sampler=None,
            )

    if len(val_dataset):
        if args.distdataparallel:
            if args.tpu_distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True,
                )

            else:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset, num_replicas=dist.get_world_size(), shuffle=True
                )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
                sampler=val_sampler,
            )
        else:

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
                sampler=None,
            )

    if len(test_dataset):
        if args.distdataparallel:
            if args.tpu_distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True,
                )
            else:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset, num_replicas=dist.get_world_size(), shuffle=True
                )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
                sampler=test_sampler,
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
                sampler=None,
            )

    return Data(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )


@ex.automain
def main(_run):
    from tqdm import tqdm
    from utils.tupperware import tupperware

    args = tupperware(_run.config)
    data = get_dataloaders(args)

    for batch in tqdm(data.train_loader.dataset):
        pass
    # for batch in data.train_loader:
    #     pass
