from hyperiap.datasets.xarray_module import XarrayDataModule
from hyperiap.datasets.point_module import PointDataModule


def test_xarray_loader():
    data = XarrayDataModule()
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == [2, 100]
    assert list(x.shape) == [2, 100, 9, 202]


def test_point_loader():
    data = PointDataModule()
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == [64, 1]
    assert list(x.shape) == [64, 1, 9, 202]
