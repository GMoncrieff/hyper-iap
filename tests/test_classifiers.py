from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.datasets.xarray_module import XarrayDataModule

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

seed_everything(1234)


def test_tempcnn_classifier():

    xmod = XarrayDataModule()

    model = LitClassifier(TEMPCNN(data_config=xmod.config()))

    trainer = Trainer(limit_train_batches=5, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)
    x = trainer.validate(datamodule=xmod)

    assert x[0]["val_acc"] > 0.01


def test_vit_classifier():

    xmod = XarrayDataModule()

    model = LitClassifier(simpleVIT(data_config=xmod.config()))

    trainer = Trainer(limit_train_batches=5, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)
    x = trainer.validate(datamodule=xmod)

    assert x[0]["val_acc"] > 0.01


def test_mae_decoder():

    xmod = XarrayDataModule()
    encoder = simpleVIT(data_config=xmod.config())
    ss_model = MAE(encoder=encoder)

    model = LitSelfSupervised(ss_model)

    trainer = Trainer(limit_train_batches=5, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)
    x = trainer.validate(datamodule=xmod)

    assert x[0]["val_loss"] > 0.01
