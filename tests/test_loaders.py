from hyperiap.datasets.xarray_module import XarrayDataModule
from hyperiap.datasets.point_module import PointDataModule
import types
import pytest


@pytest.mark.parametrize(
    "data_module_class, config, y_shape, x_shape",
    [
        (
            XarrayDataModule,
            {"test": 1, "batch_size": 2, "split": 0.2},
            [2, 100],
            [2, 100, 9, 202],
        ),
        (PointDataModule, {"test": 1, "batch_size": 64}, [64, 1], [64, 1, 9, 202]),
    ],
)
def test_data_loader(data_module_class, config, y_shape, x_shape):
    args = types.SimpleNamespace(**config)
    data = data_module_class(args=args)
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == y_shape
    assert list(x.shape) == x_shape
