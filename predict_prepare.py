import sys

from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import rioxarray as riox
import json
import pickle

from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    # check args
    if len(sys.argv) != 1:
        print("Usage: python export_raster.py")
        sys.exit(1)

    # setup dask
    print(" 1: setup client \n")
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)

    # open dataset
    print("2: open datset \n")
    dsc = xr.open_dataset("data/clean_batched_torch.zarr", chunks="auto", engine="zarr")

    print("3: attach coords \n")
    ind = pickle.load(open("data/input_dims.pickle", "rb"))
    dsc = dsc.assign_coords({"input_batch": ind})
    dsc = dsc.drop_indexes(["x", "y"])
    dsc = dsc.unstack("input_batch")
    dsc = dsc.sel(wl=slice(0, 2))
    # dsx = dsx.chunk({"wl": -1})
    dsc = dsc.drop("label")
    # dsc = dsc.sel(x=slice(320000, 323000), y=slice(6260000, 6257000))
    # dsx = dsx.sel(x=slice(320000, 326000), y=slice(6260000, 6255000))
    # dsx = dsx.sel(x=slice(320600, 325000), y=slice(6246000, 6240500))
    dsc = dsc.sel(x=slice(313990.11, 362917.11), y=slice(6266452.947, 6236800.947))
    dsc = dsc.stack({"z": ["x_batch", "y_batch"]})

    print("4: cast at int\n")
    dsc = dsc.reflectance.astype(np.int16)

    print("5 drop inds and chuhkn\n")
    dsc = dsc.drop_indexes(["z", "x", "y", "x_batch", "y_batch", "wl"])
    dsc = dsc.drop(["z"])
    dsc = dsc.chunk({"z": -1, "wl": -1, "x": 10, "y": 1000})

    # export to zarr
    print("6: write to zarr\n")
    dsc.to_dataset(name="prediction").to_zarr("data/fran_prediction.zarr", mode="w")
