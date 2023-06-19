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
ds_gcs.rio.write_crs("EPSG:32734", inplace=True)

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
points = gpd.read_file("https://storage.googleapis.com/fran-share/all.gpkg")
points = points.to_crs("EPSG:32734")


# drop some classes if needed
# points = points[~points["class"].isin([0,4,9])]

# encode code column
le = LabelEncoder()
points["label"] = le.fit_transform(points["class"])


# drop na labels
# points = points.query("label < 11")

# save name mapping
# Create a dictionary of the label-to-class mapping
# label_to_class = {
#    str(label): str(original_class) for label, original_class in enumerate(le.classes_)
# }


# dump a file mapping labels to classes
# with open("data/class_mapping.json", "w") as outfile:
#    json.dump(label_to_class, outfile)

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


# add label and group column
pointsl = points["label"].to_xarray()
pointsg = points["group"].to_xarray()
xdg = xdp.merge(pointsl).merge(pointsg)

# clean up
xdg = xdg.dropna(dim="index", how="any")


# drop chunks encoding
del xdg.x.encoding["chunks"]
del xdg.x.encoding["preferred_chunks"]
del xdg.y.encoding["chunks"]
del xdg.y.encoding["preferred_chunks"]

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
    "gcs://fran-share/fran_valsample.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)

# write data
xtest.to_zarr(
    "gcs://fran-share/fran_testsample.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)

# write data
xtr.to_zarr(
    "gcs://fran-share/fran_trainsample.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)
