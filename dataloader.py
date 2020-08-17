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

    def _load_dataset(self, glob_str="*.png") -> "Union[List,List]":
        source_paths = list(self.source_dir.glob(glob_str))[: self.max_len]

        if self.target_dir:
            target_paths = [self.target_dir / file.name for file in source_paths]
        else:
            target_paths = []

        return source_paths, target_paths

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_path = self.source_paths[index]

        if self.mode == "train":
            target_path = self.target_paths[index]
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0
            target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

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

        elif self.mode == "val":
            target_path = self.target_paths[index]
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0
            target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

        elif self.mode == "test":
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0

        source = torch.tensor(source.copy()).float().permute(2, 0, 1)
        source = (source - 0.5) * 2

        if self.mode in ["train", "val"]:
            target = torch.tensor(target.copy()).float().permute(2, 0, 1)
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

    if len(train_dataset):
        if args.distdataparallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )

            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
        )

    if len(val_dataset):
        if args.distdataparallel:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )

            shuffle = False
        else:
            val_sampler = None
            shuffle = True

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            sampler=val_sampler,
        )

    if len(test_dataset):
        if args.distdataparallel:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )

            shuffle = False
        else:
            test_sampler = None
            shuffle = True

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            sampler=test_sampler,
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
