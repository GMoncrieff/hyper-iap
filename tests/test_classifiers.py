from lightning import Trainer, seed_everything
from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.datasets.xarray_module import XarrayDataModule

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised
import types
import pytest

seed_everything(1234)


def model_func_lit(model_constructor):
    return lambda xmod: LitClassifier(model_constructor(data_config=xmod.config()))

def model_func_mae(encoder_constructor):
    return lambda xmod: LitSelfSupervised(MAE(encoder=encoder_constructor(data_config=xmod.config())))

@pytest.mark.parametrize("data_module_class, config, model_func, metric", [
    (XarrayDataModule, {'test':1,'batch_size':2,'split':0.2}, model_func_lit(TEMPCNN), "val_acc"),
    (XarrayDataModule, {'test':1,'batch_size':2,'split':0.2}, model_func_lit(simpleVIT), "val_acc"),
    (XarrayDataModule, {'test':1,'batch_size':2,'split':0.2}, model_func_mae(simpleVIT), "ss_val_loss"),
])
def test_model(data_module_class, config, model_func, metric):
    args = types.SimpleNamespace(**config)
    data_module = data_module_class(args=args)

    model = model_func(data_module)

    trainer = Trainer(
        limit_train_batches=5, limit_val_batches=3, max_epochs=2, accelerator="cpu"
    )
    trainer.fit(model, datamodule=data_module)
    validation_result = trainer.validate(datamodule=data_module)

    assert validation_result[0][metric] >= 0
