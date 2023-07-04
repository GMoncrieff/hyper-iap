from argparse import Namespace
import os
from typing import Optional, List

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
# NUM_AVAIL_CPUS = 1
NUM_AVAIL_GPUS = torch.cuda.device_count()
TRANSFORM = None

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU
# so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = (
    NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS
)


class BaseDataModule(pl.LightningDataModule):
    """Base for all of our LightningDataModules.
    Learn more at about LDMs at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)
        self.transform = self.args.get("transform", TRANSFORM)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.batch_size: int
        self.num_classes: int
        self.num_bands: int
        self.num_dim: int
        self.class_names: List[str]
        self.data_train: Dataset
        self.data_val: Dataset
        self.data_test: Dataset

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        parser.add_argument(
            "--transform",
            type=str,
            default=TRANSFORM,
            help=f"Transforms to apply to data. Default is {TRANSFORM}.",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "num_classes": self.num_classes,
            "num_bands": self.num_bands,
            "num_dim": self.num_dim,
            "wl": self.wl,
            "class_names": self.class_names,
        }

    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.
        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            batch_size=self.batch_size,
        )


#    def predict_dataloader(self):
#        return DataLoader(
#            self.data_test,
#            shuffle=False,
#            num_workers=self.num_workers,
#            pin_memory=self.on_gpu,
#            batch_size=self.batch_size,
#        )
