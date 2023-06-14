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
points = gpd.read_file("https://storage.googleapis.com/fran-share/points.gpkg")
points = points.to_crs("EPSG:32734")

# drop some classes
points = points[~points["class"].isin(["rock", "shade", "Pine"])]

# encode code column
le = LabelEncoder()
points["label"] = le.fit_transform(points["class"])

# drop na labels
points = points.query("label != 9")

# save name mapping
# Create a dictionary of the label-to-class mapping
label_to_class = {
    label: original_class for label, original_class in enumerate(le.classes_)
}

# Remove the final class from the dictionary
del label_to_class[max(label_to_class)]

with open("data/name_mapping.json", "w") as outfile:
    json.dump(label_to_class, outfile)

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
points = points["label"].to_xarray()
xdp = xdp.merge(points)

# clean up
xdp = xdp.dropna(dim="index", how="any")

# drop chunks encoding
del xdp.x.encoding["chunks"]
del xdp.x.encoding["preferred_chunks"]
del xdp.y.encoding["chunks"]
del xdp.y.encoding["preferred_chunks"]
del xdp.label.encoding["chunks"]
del xdp.label.encoding["preferred_chunks"]
del xdp.index.encoding["chunks"]
del xdp.index.encoding["preferred_chunks"]

# rechunnk
xdp = xdp.chunk({"index": 32, "wl": -1, "z": -1})

# write data
xdp.to_zarr(
    "gcs://fran-share/fran_pixsample.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)
