from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr
from einops import rearrange
import numpy as np

PREDICTOR_VAR = "reflectance"
# LABEL_VAR = "recode"
LABEL_VAR = "label"


class S2Dataset(Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        x_batch: str,
        length: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):

        self.dataset = dataset
        self.x_batch = x_batch
        self.length = length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        batch = self.dataset.isel({self.x_batch: index})
        x_batch = np.expand_dims(batch[PREDICTOR_VAR].load().data, axis=2)
        y_batch = np.expand_dims(batch[LABEL_VAR].load().data, axis=0)
        x_batch = torch.from_numpy(x_batch)
        x_batch = rearrange(x_batch, "z wl b -> b z wl")
        y_batch = torch.from_numpy(y_batch).type(torch.int64)

        if self.transform:
            x_batch = self.transform(x_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return x_batch, y_batch
