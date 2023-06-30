from argparse import Namespace
from torch.utils.data import random_split
from hyperiap.datasets.point_dataset import PointDataset
from hyperiap.datasets.base_module import BaseDataModule
from hyperiap.datasets.transforms import UnitVectorNorm, Normalize

from typing import Optional

import numpy as np
import xarray as xr
import torch
import json

BATCH_SIZE = 64
# 0->8
N_CLASS = 11
N_BAND = 202
# N_BAND = 267
N_DIM = 9

PROCESSED_TEST_DATA = "data/fran_testsample.zarr"
PROCESSED_TRAIN_DATA = "data/fran_trainsample.zarr"
PROCESSED_VALID_DATA = "data/fran_valsample.zarr"
CLASS_NAMES = "data/name_mapping.json"
WLDIM, ZDIM, BATCHDIM = "wl", "z", "index"
CHUNKS = {ZDIM: -1, WLDIM: -1, BATCHDIM: 32}


class PointDataModule(BaseDataModule):
    """lightning data module for xarray point data"""

    def __init__(self, args: Namespace = None) -> None:
        super().__init__(args)
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_classes = N_CLASS
        self.num_bands = N_BAND
        self.num_dim = N_DIM

        self.data_train = None
        self.data_test = None
        self.data_val = None

        # load data
        try:
            self.batch_gen_train = xr.open_dataset(PROCESSED_TRAIN_DATA, chunks=CHUNKS)
        except FileNotFoundError:
            print(f"Train data file {PROCESSED_TRAIN_DATA} not found")
        try:
            self.batch_gen_valid = xr.open_dataset(PROCESSED_VALID_DATA, chunks=CHUNKS)
        except FileNotFoundError:
            print(f"Valid data file {PROCESSED_VALID_DATA} not found")
        try:
            self.batch_gen_test = xr.open_dataset(PROCESSED_TEST_DATA, chunks=CHUNKS)
        except FileNotFoundError:
            print(f"Test data file {PROCESSED_TEST_DATA} not found")

        # get class names
        try:
            class_dict = json.load(open(CLASS_NAMES))
            self.class_names = list(class_dict.values())
        except FileNotFoundError:
            print(f"class names file {CLASS_NAMES} not found")

        # store wl for later use
        self.wl = self.batch_gen_train.sel(wl=slice(0, 2.1)).wl.values

    def prepare_data(self, *args, **kwargs) -> None:
        """download data here"""

    def setup(self, stage: Optional[str] = None):
        """
        Read downloaded data
        Setup Datasets
        Split the dataset into train/val/test."""

        train_dataset_size = self.batch_gen_train.dims[BATCHDIM]
        valid_dataset_size = self.batch_gen_valid.dims[BATCHDIM]
        test_dataset_size = self.batch_gen_test.dims[BATCHDIM]

        self.data_train = PointDataset(
            self.batch_gen_train, BATCHDIM, train_dataset_size, transform=Normalize()
        )
        self.data_val = PointDataset(
            self.batch_gen_valid, BATCHDIM, valid_dataset_size, transform=Normalize()
        )
        self.data_test = PointDataset(
            self.batch_gen_test, BATCHDIM, test_dataset_size, transform=Normalize()
        )

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}.",
        )
        return parser

    def __str__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "xarray example dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Num bands: {self.num_bands}\n"
            f"Num dim: {self.num_dim}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))

        test_size = 0 if self.data_test is None else len(self.data_test)

        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {test_size}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min().item(), x.mean().item(), x.std().item(), x.max().item())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min().item(), y.max().item())}\n"
        )
        return basic + data
