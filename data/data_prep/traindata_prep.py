import xarray as xr

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4)
client = Client(cluster)

# set dim names
XDIM, YDIM, WLDIM = "x", "y", "wl"
# open raw data
dsx = xr.open_dataset("data/fran_torch.zarr", chunks="auto")

# delte encoding so that chunks are determined by dask chunks
del dsx.label.encoding["chunks"]
del dsx.label.encoding["preferred_chunks"]
del dsx.reflectance.encoding["chunks"]
del dsx.reflectance.encoding["preferred_chunks"]
# reorder dims and rechunk
dsx["label"] = dsx["label"].transpose()
dsx = dsx.chunk({"x": 100, "y": 100, "wl": 267})

# dsx.to_zarr('/mnt/disks/extra/clean_torch.zarr', mode='w')
# dsx = xr.open_dataset("/mnt/disks/extra/clean_torch.zarr", chunks='auto')

# extract 3x3 neighbourhood
dsx = dsx.rolling({XDIM: 3, YDIM: 3}, center=True).construct(
    {XDIM: "x_batch", YDIM: "y_batch"}
)
# select only label for center cell
dsx["label"] = dsx["label"].isel(x_batch=1, y_batch=1)
dsx = dsx.chunk({XDIM: 100, YDIM: 100, WLDIM: -1, "x_batch": -1, "y_batch": -1})

# dsx.to_zarr("/mnt/disks/extra/rolling_torch.zarr", mode="w")
# dsx = xr.open_dataset("/mnt/disks/extra/rolling_torch.zarr", chunks='auto')

# stack and fillna (there should not be any)
(
    dsx.isel(y=slice(1, 9886), x=slice(1, 16311))
    .stack({"input_batch": (XDIM, YDIM)}, create_index=False)
    # .dropna(dim='input_batch', how='any'))
    .drop({"x", "y"})
    # where <0 then 0
    # .dropna(dim='input_batch', how='all'))
    .fillna(0)
    .chunk({"input_batch": 10000, "x_batch": -1, "y_batch": -1, "wl": -1})
    .to_zarr("/mnt/disks/extra/clean_batched_torch.zarr", mode="w")
)

# dsx = xr.open_dataset("/mnt/disks/extra/clean_batched_torch.zarr", chunks={'input_batch':10000})
# to load in dataloader:
# len = dsx.label.data.blocks.size

# get block
# dsx.reflectance.data.blocks[0,0,0,i].compute()
# dsx.label.data.blocks[0,0,0,i].compute()
