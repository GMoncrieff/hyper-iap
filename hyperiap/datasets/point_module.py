from argparse import Namespace
from torch.utils.data import random_split
from hyperiap.datasets.xarray_dataset import MapDataset
from hyperiap.datasets.base_module import BaseDataModule
from typing import Optional

import numpy as np
from os.path import dirname, abspath

import xarray as xr
import xbatcher

BATCH_SIZE = 128

SPLIT = 0.2
N_CLASS = 10
N_BAND = 267
N_DIM = 9

PROCESSED_TRAIN_DATA = "gcs://fran-share/fran_pixsample.zarr"
PROCESSED_TEST_DATA = "gcs://fran-share/fran_pixsample.zarr"
WLDIM, ZDIM, BATCHDIM = "wl", "z", "index"


class XarrayDataModule(BaseDataModule):
    """lightning data module for xarray point data"""

    def __init__(self, args: Namespace = None) -> None:
        super().__init__(args)
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.split = self.args.get("split", SPLIT)
        self.num_classes = N_CLASS
        self.num_bands = N_BAND
        self.num_dim = N_DIM

        self.data_train = None
        self.data_test = None
        self.data_val = None

    def prepare_data(self, *args, **kwargs) -> None:
        """download data here"""

    def setup(self, stage: Optional[str] = None):
        """
        Read downloaded data
        Setup Datasets
        Split the dataset into train/val/test."""

        # load data
        try:
            traindata = xr.open_dataset(
                PROCESSED_TRAIN_DATA,
                chunks="auto",
                backend_kwargs={
                    "storage_options": {"project": "science-sharing", "token": "anon"}
                },
                engine="zarr",
            )
        except FileNotFoundError:
            print(f"Train data file {self.full_train_file} not found")

        # try:
        #    testdata = xr.open_zarr(self.full_test_file)
        # except FileNotFoundError:
        #    print(f'Test data file {self.full_test_file} not found')

        # traindata = traindata.stack(batch=(XDIM, YDIM))
        # testdata = testdata.stack(batch=(XDIM, YDIM))

        self.batch_gen_train = xbatcher.BatchGenerator(
            traindata,
            input_dims={BATCHDIM: 1, WLDIM: N_BAND, ZDIM: N_DIM},
            batch_dims={BATCHDIM: BATCH_SIZE},
            concat_input_dims=True,
            preload_batch=False,
        )
        # self.batch_gen_test = xbatcher.BatchGenerator(
        # testdata,
        # input_dims = {WLDIM: N_BAND,ZDIM:N_DIM,'batch':BATCH_SIZE},
        # concat_input_dims=False,
        # preload_batch=True)

        traindata = MapDataset(self.batch_gen_train)
        # self.data_test=traindata = MapDataset(self.batch_gen_test)

        dataset_size = len(traindata)
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
