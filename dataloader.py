"""
Dataloaders
"""

# Libs
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
from sacred import Experiment
import skimage
import copy

# Torch modules
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
from config import initialise
from pathlib import Path

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

    def __init__(self, args, mode: str = "train", max_len: int = None):
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

        logging.info(
            f"{mode.capitalize()} Set | Source Dir: {self.source_dir} | Target Dir: {self.target_dir}"
        )

    def _load_dataset(self):
        if self.mode == "train":
            glob_str = "*.png"
        elif self.mode in ["val", "test"]:
            glob_str = "*.npy"

        if self.source_dir:
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

        if self.mode == "train":
            target_path = self.target_paths[index]
            source = cv2.imread(str(source_path))[:, :, ::-1] / 255.0
            target = cv2.imread(str(target_path))[:, :, ::-1] / 255.0

        elif self.mode == "val":
            target_path = self.target_paths[index]
            source = np.load(source_path) / 255.0
            target = np.load(target_path) / 255.0

        elif self.mode == "test":
            source = np.load(source_path) / 255.0

        source = torch.tensor(source).float().permute(2, 0, 1)
        # source = (source - 0.5) * 2

        if self.mode in ["train", "val"]:
            target = torch.tensor(target).float().permute(2, 0, 1)
            # target = (target - 0.5) * 2

            return (source, target, source_path.name)

        else:
            return (source, source_path.name)


def get_dataloaders(args):
    """
    Get dataloaders for train and val

    Returns:
    :data
    """
    train_dataset = OLEDDataset(args, mode="train")
    val_dataset = OLEDDataset(args, mode="val")
    test_dataset = OLEDDataset(args, mode="test")

    logging.info(
        f"Dataset Train: {len(train_dataset)} Val: {len(val_dataset)}  Test: {len(test_dataset)}"
    )

    train_loader = None
    val_loader = None
    test_loader = None

    if len(train_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
        )

    if len(val_dataset):
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

    if len(test_dataset):
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
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
