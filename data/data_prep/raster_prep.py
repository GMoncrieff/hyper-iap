import xarray as xr
import rioxarray as riox
from dask_ml.preprocessing import LabelEncoder

import json
import urllib.request

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4)
client = Client(cluster)

ds_gcs = xr.open_dataset(
    "gcs://fran-share/fran_mosaic.zarr",
    chunks="auto",
    backend_kwargs={"storage_options": {"project": "science-sharing", "token": "anon"}},
    engine="zarr",
)
# rename vars and set crs
ds_gcs = ds_gcs.rename({"X": "x", "Y": "y"})

with urllib.request.urlopen(
    "https://storage.googleapis.com/fran-share/wl_dict.json"
) as url:
    wldict = json.load(url)

dsx = ds_gcs.rename(wldict)
# get new band names
wl_bands = list(dsx.data_vars)
# convert to dim
dsx = dsx[wl_bands].to_array(dim="wl")
dsx = dsx.sortby("wl")

# write crs
dsx.rio.write_crs("EPSG:32734", inplace=True)
# bb = minx, miny, maxx, maxy
bb = [313988, 6236797, 362919, 6266456]
dsx = dsx.rio.clip_box(*bb)

dx = riox.open_rasterio(
    "https://storage.googleapis.com/fran-share/SA_NLC_2018_GEO.tif", chunks="auto"
)

# convert to crs of template
dsxb = dsx.rio.transform_bounds(
    "+init=epsg:4326",
    densify_pts=21,
)

# crop using bounding box
dx = dx.rio.clip_box(*dsxb)
dx = dx.rio.reproject_match(dsx)
dx = dx.stack(z=("x", "y")).squeeze(drop=True)

# encode labels
dx.data = LabelEncoder().fit_transform(dx.data)
dx = dx.unstack("z")

# Combine the two datasets
dx = dx.assign_coords(
    {
        "x": dsx.x,
        "y": dsx.y,
    }
)
dx = dx.to_dataset(name="label")
dsx = dsx.fillna(0).to_dataset(name="reflectance")
ds_join = dsx.drop("spatial_ref").merge(dx.drop("spatial_ref"))
ds_join = ds_join.chunk({"x": 1000, "y": 1000})

# write
ds_join.to_zarr(
    "gcs://fran-share/fran_pytorch.zarr",
    consolidated=True,
    storage_options={"project": "science-sharing", "token": "anon"},
)
