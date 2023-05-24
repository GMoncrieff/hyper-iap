from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr
from einops import rearrange

PREDICTOR_VAR = "reflectance"
LABEL_VAR = "label"


class XarrayDataset(Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        x_batch: str,
        length: int,
        batch_size: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.x_batch = x_batch
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        batch = self.dataset.isel(
            {
                self.x_batch: slice(
                    index * self.batch_size, (index * self.batch_size) + self.batch_size
                )
            }
        )
        batch = batch.sel(wl=slice(0, 2))
        x_batch = batch[PREDICTOR_VAR].load()
        y_batch = batch[LABEL_VAR].load()
        x_batch = torch.from_numpy(x_batch.data)
        x_batch = rearrange(x_batch, "wl x y b -> b (x y) wl")
        y_batch = torch.from_numpy(y_batch.data).type(torch.int64)

        if self.transform:
            x_batch = self.transform(x_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return x_batch, y_batch
