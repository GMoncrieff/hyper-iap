from xrspatial.convolution import circle_kernel
from xrspatial.focal import apply
import xarray as xr
import json

from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)
    client

    # open dataset
    result = xr.open_dataset("data/prediction_gpu.zarr", chunks="auto")
    result = result.transpose("y", "x", "class")
    resultgeo = result.rio.write_crs("epsg:32734")

    # add class names
    cname = json.load(open("data/name_mapping.json"))
    cname = list(cname.values())
    resultgeo.coords["class"] = cname
    resds = resultgeo["prediction"].to_dataset(
        dim="class"
    )  # .chunk({"x": 100,"y":100})

    kernel = circle_kernel(1, 1, 2)

    for var in resds.data_vars:
        resds[var] = apply(resds[var], kernel)

    # export to tif
    resds = resds * 100
    resds = resds.astype("int8")

    resds.rio.to_raster("data/cnn_class_gpu_int.tif")
