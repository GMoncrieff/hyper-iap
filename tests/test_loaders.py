from hyperiap.datasets.xarray_module import XarrayDataModule
from hyperiap.datasets.point_module import PointDataModule


def test_xarray_loader():
    data = XarrayDataModule()
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == [1, 100]
    assert list(x.shape) == [1, 100, 9, 267]


def test_point_loader():
    data = PointDataModule()
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == [32, 1]
    assert list(x.shape) == [32, 1, 9, 267]
