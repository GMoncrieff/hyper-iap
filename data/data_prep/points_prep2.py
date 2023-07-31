import json
import urllib.request
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as riox
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder

# the start
dsc = xr.open_dataset(
    "data/clean_batched_torch.zarr",
    chunks="auto",
    engine="zarr",
)

ind = pickle.load(open("data/input_batch.pickle", "rb"))
dsc = dsc.assign_coords({"input_batch": ind})
dsc = dsc.drop_indexes(["x", "y"])
dsx = dsc.unstack("input_batch")
# dsc = dsc.drop('label')

# dsx = dsc.sel(wl=slice(0, 2))
# dsx = dsx.chunk({"wl": -1})
dsx = dsx.drop("label")
# dsx = dsx.sel(x=slice(320000, 326000), y=slice(6260000, 6255000))
# dsx = dsx.sel(x=slice(320600, 325000), y=slice(6246000, 6240500))
dsx = dsx.isel(x=slice(0, 16310), y=slice(0, 9885))
# dsx = dsx.chunk({"x": 100, "y": 100})
dsx

# open labels
############
points = gpd.read_file("https://storage.googleapis.com/fran-share/all2.gpkg")
points = points.to_crs("EPSG:32734")


# drop some classes if needed
# points = points[~points["class"].isin([0,4,9])]

# encode code column
le = LabelEncoder()
points["label"] = le.fit_transform(points["class"])
px = np.array(points["geometry"].centroid.x)
py = np.array(points["geometry"].centroid.y)

# create dataarray of points
xi = xr.DataArray(px, dims=["index"])
yi = xr.DataArray(py, dims=["index"])

# select values at points
xdp = dsx.sel(x=xi, y=yi, method="nearest")

# add label and group column
pointsl = points["label"].to_xarray()
pointsg = points["group"].to_xarray()
xdg = xdp.merge(pointsl).merge(pointsg)

# clean up
xdg = xdg.dropna(dim="index", how="any")
# export if needed

# xdg.to_zarr("data/fran_temp.zarr", mode="w")
# xdg = xr.open_dataset("data/fran_temp.zarr", chunks="auto")

# stack
xdg = xdg.stack({"z": ("x_batch", "y_batch")}, create_index=False)

xdg = xdg.persist()

# filter to test/train/val and chunk
xval = (
    xdg.where(xdg.group == 0, drop=True)
    .drop("group")
    .chunk({"index": 32, "wl": -1, "z": -1})
)
xtest = (
    xdg.where(xdg.group == 1, drop=True)
    .drop("group")
    .chunk({"index": 32, "wl": -1, "z": -1})
)
xtr = (
    xdg.where(xdg.group == 2, drop=True)
    .drop("group")
    .chunk({"index": 32, "wl": -1, "z": -1})
)

# write data
xval.to_zarr(
    "gcs://fran-share/fran_val_sample2.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)

# write data
xtest.to_zarr(
    "gcs://fran-share/fran_test_sample2.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)

# write data
xtr.to_zarr(
    "gcs://fran-share/fran_train_sample2.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)
