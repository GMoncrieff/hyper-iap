from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.datasets.xarray_module import XarrayDataModule

from hyperiap.models.baseclassifier import BaseClassifier


def test_tempcnn_classifier():
    seed_everything(1234)

    xmod = XarrayDataModule()

    model = BaseClassifier(TEMPCNN(data_config=xmod.config()))

    trainer = Trainer(limit_train_batches=5, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)
    x = trainer.validate(datamodule=xmod)

    assert x[0]["val/acc"] > 0.5


def test_vit_classifier():
    seed_everything(1234)

    xmod = XarrayDataModule()

    model = BaseClassifier(simpleVIT(data_config=xmod.config()))

    trainer = Trainer(limit_train_batches=5, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)
    x = trainer.validate(datamodule=xmod)

    assert x[0]["val/acc"] > 0.5
