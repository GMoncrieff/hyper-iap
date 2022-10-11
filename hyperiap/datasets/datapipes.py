from typing import Any, Dict, Iterator, Optional, Hashable, Tuple, Union

import torch
import xarray as xr

try:
    import xbatcher
except ImportError:
    xbatcher = None

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

xvar = "x"
yvar = "y"


@functional_datapipe("read_from_zarr")
class XarrayReaderIterDataPipe(IterDataPipe[StreamWrapper]):
    """
    Takes multidim array files (e.g. zarr) from local disk or URLs
    (as long as they can be read by xarray)
    and yields :py:class:`xarray.DataArray` objects (functional name:
    ``read_from_zarr``).

    Based on
    https://github.com/pytorch/data/blob/v0.4.0/torchdata/datapipes/iter/load/online.py#L55-L96
    and
    https://github.com/weiji14/zen3geo/blob/main/zen3geo/datapipes/rioxarray.py

    Parameters
    ----------
    source_datapipe : IterDataPipe[str]
        A DataPipe that contains filepaths or URL links to multidim arrays.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`xarray.open_zarr`.

    Yields
    ------
    stream_obj : xarray.DataArray
        An :py:class:`xarray.DataArray` object containing the raster data.
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[StreamWrapper]:
        for filename in self.source_datapipe:
            yield StreamWrapper(xr.open_zarr(store=filename, **self.kwargs))

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("slice_with_xbatcher")
class XbatcherSlicerIterDataPipe(IterDataPipe[Union[xr.DataArray, xr.Dataset]]):
    """
    Takes an :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
    and creates a sliced window view (also known as a chip or tile) of the
    n-dimensional array (functional name: ``slice_with_xbatcher``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[xr.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects.

    input_dims : dict
        A dictionary specifying the size of the inputs in each dimension to
        slice along, e.g. ``{'lon': 64, 'lat': 64}``. These are the dimensions
        the machine learning library will see. All other dimensions will be
        stacked into one dimension called ``batch``.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`xbatcher.BatchGenerator`.

    Yields
    ------
    chip : xarray.DataArray
        An :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset` object
        containing the sliced raster data, with the size/shape defined by the
        ``input_dims`` parameter.

    Raises
    ------
    ModuleNotFoundError
        If ``xbatcher`` is not installed. Follow
        :doc:`install instructions for xbatcher <xbatcher:index>`
        before using this class.

    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Union[xr.DataArray, xr.Dataset]],
        input_dims,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe[
            Union[xr.DataArray, xr.Dataset]
        ] = source_datapipe
        self.input_dims: Dict[Hashable, int] = input_dims
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Union[xr.DataArray, xr.Dataset]]:
        for dataarray in self.source_datapipe:
            if hasattr(dataarray, "name") and dataarray.name is None:
                # Workaround for ValueError: unable to convert unnamed
                # DataArray to a Dataset without providing an explicit name
                dataarray = dataarray.to_dataset(
                    name=xr.backends.api.DATAARRAY_VARIABLE
                )[xr.backends.api.DATAARRAY_VARIABLE]
                # dataarray.name = "z"  # doesn't work for some reason
            for chip in dataarray.batch.generator(
                input_dims=self.input_dims, **self.kwargs
            ):
                yield chip


def xr_collate_fn(samples) -> torch.Tensor:
    """
    Converts individual xarray.Dataset objects to a torch.Tensor,
    and stacks them all into a single torch.Tensor.
    """
    tensors = [
        (
            torch.as_tensor(data=sample.data_vars.get(key=xvar).data.astype("float")),
            torch.as_tensor(data=sample.data_vars.get(key=yvar).data.astype("float")),
        )
        for sample in samples
    ]

    tensors = list(zip(*tensors))
    return {"x": torch.stack(tensors[0]), "y": torch.stack(tensors[1])}
