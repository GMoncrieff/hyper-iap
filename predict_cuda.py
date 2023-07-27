import sys

from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import rioxarray as riox
import json
import pickle
from lightning import Trainer, seed_everything
from xrspatial.convolution import circle_kernel
from xrspatial.focal import apply

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from utils.model_helpers import get_vit_model, get_tempcnn_model
from torchuq.evaluate import categorical
from torchuq.transform.calibrate import TemperatureScaling


def apply_model(chunk, dmodel, calibrator=None):
    # get dims
    # chunk = chunk/10000
    device = torch.device("cuda")
    v, w, z = chunk.shape
    # Convert the chunk to a tensor.
    x_batch = rearrange(chunk, "v w z -> v z w")
    # chunk = chunk/10000
    # Convert the chunk to a tensor.
    # x_batch = rearrange(chunk, " b1 w1 z1-> b1 z1 w1")
    tensor = torch.from_numpy(x_batch)
    tensor = tensor.to(device)
    # tensor = tensor.type(torch.double)

    # apply transforms
    # Calculate the mean for each batch and level along the wavelength dimension
    # stds = x_batch.std(dim=-1, keepdim=True)
    # mean = x_batch.mean(dim=-1, keepdim=True)
    # Normalize the tensor along the wavelength dimension
    # tensor = (x_batch - mean) / stds

    # print(tensor.shape)
    dmodel, _ = dmodel.result()
    dmodel = dmodel.to(device)
    # Pass the tensor through the model.

    # dmodel=dmodel.float()
    # tensor=tensor.float()
    with torch.no_grad():
        result = dmodel(tensor)

    result = F.softmax(result, dim=-1)
    # Convert the result back to a numpy array and reshape it to the chunk shape.
    # calibrator = calibrator.float()
    if calibrator is not None:
        result = calibrator(result)

    result = result.to("cpu").detach().numpy()
    # result = rearrange(result,'(x y) d -> x y d',x=x,y=y,d=11)

    return result


def calibrate_model(runid="007kslnc", modelname="cnn"):
    """
    Calibrates a trained model using temperature scaling.

    Args:
    runid (str): A runid for the model. Default is "007kslnc".
    model (str): A model type. Default is "cnn".

    Returns:
    torchuq.transform.calibrate.TemperatureScaling: A temperature scaling calibrator.
    """
    # load model
    if modelname == "vit":
        model, loader = get_vit_model(runid)
    elif modelname == "cnn":
        model, loader = get_tempcnn_model(runid)
    else:
        raise ValueError("Model must be 'vit' or 'cnn'")

    # predict on testdata
    trainer = Trainer()
    loader.prepare_data()
    loader.setup()
    print(" 1.1 \n")
    x = trainer.predict(model, dataloaders=loader.test_dataloader())
    print(" 1.2 \n")

    y = [np.array(y) for x, y in x]
    y = np.concatenate(y, axis=0)
    y = torch.from_numpy(y)
    print(y.shape)
    x = [np.array(x) for x, y in x]
    x = np.concatenate(x, axis=0)
    x = torch.from_numpy(x)
    print(x.shape)
    # calibrate
    calibrator = TemperatureScaling(verbose=True)
    calibrator.train(x, y)

    return calibrator


if __name__ == "__main__":
    # check args
    if len(sys.argv) != 3:
        print("Usage: python predict_torch.py <run_id> <model>")
        sys.exit(1)

    model = sys.argv[2]

    # calibrate model
    print(" 1: calibraete model \n")
    # calibrated = calibrate_model(sys.argv[1], sys.argv[2])

    # setup dask
    print(" 2: setup client \n")
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0")
    client = Client(cluster)
    client

    # open dataset
    # print("3: open datset \n")
    # dsc = xr.open_dataset(
    #    "data/prediction_small.zarr",
    #    chunks="auto")
    dsc = xr.open_dataset("/mnt/hdd1/fran/fran_prediction.zarr", chunks="auto")

    # dsc= xr.open_dataset(
    #    "gcs://fran-local/fran_prediction.zarr",
    #    chunks="auto",
    #    backend_kwargs={"storage_options": {"project": "science-sharing", "token": "anon"}},
    #    engine="zarr",
    # )
    # dsc = dsc.sel(x=slice(319000, 322000), y=slice(6246000, 6242000))
    dsc = dsc.stack(v=("x", "y"))
    dsc = dsc.astype(np.float32)
    dsc = dsc / 10000
    # attach coords
    # send model
    print("6: sumbit model to clinet \n")

    if model == "vit":
        remote_model = client.submit(get_vit_model)
    elif model == "cnn":
        remote_model = client.submit(get_tempcnn_model)
    else:
        raise ValueError("Model must be 'vit' or 'cnn'")

    # map fun
    print("7: map function \n")
    result = xr.apply_ufunc(
        apply_model,
        dsc.prediction,
        input_core_dims=[["wl", "z"]],
        exclude_dims=set(
            (
                "wl",
                "z",
            )
        ),
        output_core_dims=[["class"]],
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"class": 11},
        kwargs={"dmodel": remote_model, "calibrator": None},
    )

    print("8: reshape reulst \n")
    result = result.unstack("v")
    result = result.chunk({"x": 10, "y": 1000, "class": -1}).to_dataset(
        name="prediction"
    )

    # write result
    print("9: writing zarr \n")
    result.to_zarr("data/prediction_gpu.zarr", mode="w")
