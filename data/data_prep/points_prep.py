import json
import urllib.request

import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as riox
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder

# setup dask
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4)
client = Client(cluster)

# open data on gcs
ds_gcs = xr.open_dataset(
    "gcs://fran-share/fran_mosaic.zarr",
    chunks="auto",
    backend_kwargs={"storage_options": {"project": "science-sharing", "token": "anon"}},
    engine="zarr",
)
# rename vars and set crs
ds_gcs = ds_gcs.rename({"X": "x", "Y": "y"})
ds_gcs = ds_gcs.rio.set_crs("EPSG:32734")

# open dict with band wls
with urllib.request.urlopen(
    "https://storage.googleapis.com/fran-share/wl_dict.json"
) as url:
    wldict = json.load(url)
dsx = ds_gcs.rename(wldict)

# get new band names
wl_bands = list(dsx.data_vars)

# convert to dataset
dsx = dsx[wl_bands].to_array(dim="wl")
dsx = dsx.to_dataset(name="reflectance")
dsx = dsx.sortby("wl")

# drop bad bands
dsx = dsx.where(
    ((dsx.wl > 0.400) & (dsx.wl < 1.340))
    | ((dsx.wl > 1.455) & (dsx.wl < 1.790))
    | ((dsx.wl > 1.955) & (dsx.wl < 2.400)),
    drop=True,
)

# open labels
############
points = gpd.read_file("https://storage.googleapis.com/fran-share/points.gpkg")

# drop some classes
points = points[~points["class"].isin(["rock", "shade", "Pine"])]

# encode code column
le = LabelEncoder()
points["recode"] = le.fit_transform(points["code"])

# get neighbour pixels
#####################

# get point x and y coords (they are not stored as points)
px = np.array(points["geometry"].centroid.x)
py = np.array(points["geometry"].centroid.y)

# get raster res
rx, ry = dsx.rio.resolution()
xmin, xmax = px - rx, px + rx
ymin, ymax = py - ry, py + ry

# concatenate pixel coords with neighboring pixel coords
xdims = np.stack((px, xmin, xmax, px, xmin, xmax, px, xmin, xmax))
ydims = np.stack((py, ymin, ymax, ymin, ymax, py, ymax, py, ymin))

# create xarray for selecting pix
x_indexer = xr.DataArray(xdims, dims=["z", "index"])
y_indexer = xr.DataArray(ydims, dims=["z", "index"])

# select data at x and y coords
xdp = dsx.sel(x=x_indexer, y=y_indexer, method="nearest").load()

# add label column
points = points["recode"].to_xarray()
xdp = xdp.merge(points)

# write data
xdp.to_zarr(
    "gcs://fran-share/fran_pixsample.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)