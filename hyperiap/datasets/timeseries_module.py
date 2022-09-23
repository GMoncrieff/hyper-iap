from argparse import Namespace
import torch
from torch.utils.data import random_split
from hyperiap.datasets.base_dataset import BaseDataset
from hyperiap.datasets.base_module import BaseDataModule
from typing import Optional
import numpy as np
import os
from os.path import dirname, abspath
import requests

SPLIT = 0.2
PROCESSED_DATA_DIRNAME = "/data"
PROCESSED_TRAIN_DATA_FILENAME = "ts_data_train.tsv"
PROCESSED_TEST_DATA_FILENAME = "ts_data_test.tsv"
RAW_TRAIN_DATA_URL = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
RAW_TEST_DATA_URL = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"

class TimeSeriesDataModule(BaseDataModule):
    """lightning data module for timeseries data"""
    def __init__(
        self,
        args: Namespace = None,
    ):
        super().__init__(args)

        self.split = self.args.get("split", SPLIT)
        self.num_classes = 2
        self.num_bands = 500
        self.num_dim = 27
        
        self.data_train = None
        self.data_test = None
        self.data_val = None
        
    def prepare_data(self, *args, **kwargs) -> None:
        """download data here"""
        #create paths and filenames
        parent = dirname(abspath('__file__'))
        self.full_path = parent+PROCESSED_DATA_DIRNAME
        print(self.full_path)
        self.full_test_file = self.full_path+'/'+PROCESSED_TEST_DATA_FILENAME
        self.full_train_file = self.full_path+'/'+PROCESSED_TRAIN_DATA_FILENAME
        
        #download data
        if not os.path.isfile(self.full_train_file):
            print(f"downloading train data to {self.full_train_file}...")
            _download_csv(RAW_TRAIN_DATA_URL, self.full_path, self.full_train_file)
            print(f"successfully downloaded train data to {self.full_train_file}")
        else:
            print(f"found train data at {self.full_train_file}")
            
        if not os.path.isfile(self.full_test_file):
            print(f"downloading test data to {self.full_test_file}...")
            _download_csv(RAW_TEST_DATA_URL, self.full_path, self.full_test_file)
            print(f"successfully downloaded test data to {self.full_test_file}")
        else:
            print(f"found test data at {self.full_test_file}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Read downloaded data
        Setup Datasets
        Split the dataset into train/val/test."""
         
        test_x, test_y= _read_ts_data(self.full_test_file)
        train_x, train_y = _read_ts_data(self.full_train_file)
        
        self.data_test = BaseDataset(test_x, test_y, transform=self.transform)
        data = BaseDataset(train_x, train_y, transform=self.transform)
        
        dataset_size = len(data)
        split = int(np.floor(self.split * dataset_size))

        self.data_train, self.data_val = random_split(
            data, [dataset_size - split, split]
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
        return parser
    
    def __str__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "time series example dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Num bands: {self.num_bands}\n"
            f"Num dim: {self.num_dim}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min().item(), x.mean().item(), x.std().item(), x.max().item())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min().item(), y.max().item())}\n"
        )
        return basic + data
    
def _read_ts_data(file: str):
    """Reads the data from the file and returns a tuple of tensors"""
    data = np.loadtxt(file, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    #x = x.reshape((x.shape[0], x.shape[1], 1))
    #rep 27 times to simulate added dim
    x = np.repeat(x[:, :, np.newaxis], 27, axis=2)
    #x = x.transpose(0,2,1)
    y = y.astype(int)
    y[y == -1] = 0
    return torch.tensor(x).float(), torch.tensor(y)
    
def _download_csv(url: str, dir:str, file:str) -> None:
    """download csv file from url and place in dir"""
    os.makedirs(dir,exist_ok=True)
    if not os.path.isfile(file):
        r = requests.get(url)
        with open(file, 'wb') as f:
            f.write(r.content)
            

    