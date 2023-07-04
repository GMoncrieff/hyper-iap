from argparse import Namespace
from torch.utils.data import random_split
from hyperiap.datasets.xarray_dataset import XarrayDataset
from hyperiap.datasets.base_module import BaseDataModule
from hyperiap.datasets.transforms import UnitVectorNorm, Normalize

from typing import Optional

import numpy as np
from os.path import dirname, abspath

import xarray as xr

SPLIT = 0.2
N_CLASS = 53
N_BAND = 202
# N_BAND = 267
N_DIM = 9
# tempcnn
# BATCH_SIZE = 2
# vit
BATCH_SIZE = 2

PROCESSED_PROJECT = "science-sharing"
# PROCESSED_TRAIN_PATH = "gcs://fran-share/clean_batched_torch.zarr"
PROCESSED_TRAIN_PATH = "data/clean_batched_torch.zarr"
PROCESSED_TESTDATA_PATH = "data/test_torch_batched.zarr"

XDIM, YDIM, WLDIM, BATCHDIM = "x_batch", "y_batch", "wl", "input_batch"
TEST = 0

class XarrayDataModule(BaseDataModule):
    """lightning data module for xarray data"""

    def __init__(self, args: Namespace = None) -> None:
        super().__init__(args)
        self.test = self.args.get("test", TEST)
        self.split = self.args.get("split", SPLIT)
        self.num_classes = N_CLASS
        self.num_bands = N_BAND
        self.num_dim = N_DIM
        self.class_names = [""] * N_CLASS
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)

        self.data_train = None
        self.data_test = None
        self.data_val = None

        # load data
        if self.test==0:
            self.chunks = {XDIM: -1, YDIM: -1, WLDIM: -1, BATCHDIM: 10000}
            try:
                self.batch_gen_train = xr.open_dataset(PROCESSED_TRAIN_PATH, chunks=self.chunks)
            except FileNotFoundError:
                print(f"Train data file {PROCESSED_TRAIN_PATH} not found")
        else:
            self.chunks = {XDIM: -1, YDIM: -1, WLDIM: -1, BATCHDIM: 100}
            try:
                self.batch_gen_train = xr.open_dataset(PROCESSED_TESTDATA_PATH, chunks=self.chunks)
            except FileNotFoundError:
                print(f"Testing data file {PROCESSED_TESTDATA_PATH} not found")
            

        # try:
        #    testdata = xr.open_zarr(self.full_test_file)
        # except FileNotFoundError:
        #    print(f'Test data file {self.full_test_file} not found')

        # store wl for later use
        self.wl = self.batch_gen_train.sel(wl=slice(0, 2.0)).wl.values

    def prepare_data(self, *args, **kwargs) -> None:
        """download data here"""

    def setup(self, stage: Optional[str] = None):
        """
        Read data from cloud storage
        Setup Datasets
        Split the dataset into train/val/test."""

        dataset_size = (self.batch_gen_train.dims[BATCHDIM] // self.chunks[BATCHDIM]) - 1

        traindata = XarrayDataset(
            self.batch_gen_train,
            BATCHDIM,
            dataset_size,
            self.chunks[BATCHDIM],
            transform=Normalize(),
        )
        # self.data_test = XarrayDataset(self.batch_gen_test)

        split = int(np.floor(self.split * dataset_size))

        self.data_train, self.data_val = random_split(
            traindata, [dataset_size - split, split]
        )

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--split",
            type=float,
            default=SPLIT,
            help=f"timeseries test/val split. Default is {SPLIT}",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}.",
        )
        parser.add_argument(
            "--test",
            type=int,
            default=TEST,
            help=f"0 = use full dataset, 1 = use small testing dataset. Default is {TEST}.",
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
