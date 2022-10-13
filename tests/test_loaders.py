from hyperiap.datasets.xarray_module import XarrayDataModule


def test_xarray_loader():
    data = XarrayDataModule()
    data.prepare_data()
    data.setup()

    x, y = next(iter(data.train_dataloader()))

    assert list(y.shape) == [1, 128]
    assert list(x.shape) == [1, 322, 9, 128]
