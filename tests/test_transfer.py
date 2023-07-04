from lightning import Trainer, seed_everything
from finetuning_scheduler import FinetuningScheduler

from hyperiap.models.vit import simpleVIT, TransferLearningVIT
from hyperiap.models.mae import MAE
from hyperiap.datasets.xarray_module import XarrayDataModule
from hyperiap.models.tempcnn import TEMPCNN, TransferLearningTempCNN

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

seed_everything(1234)
import pytest
import types

def pre_train(model, xmod, epochs):
    trainer = Trainer(
        limit_train_batches=2, limit_val_batches=2, max_epochs=epochs, accelerator="cpu"
    )
    trainer.fit(model, datamodule=xmod)

def fine_tune(ft_schedule_name, model, xmod, epochs):
    trainer = Trainer(
        max_epochs=epochs,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        callbacks=[FinetuningScheduler(ft_schedule=ft_schedule_name)],
    )
    trainer.fit(model, datamodule=xmod)
    return trainer.validate(datamodule=xmod)

@pytest.mark.parametrize("model_func, config, encoder_func, transfer_func, schedule_yaml, epochs", [
    # For the ViT classifier
    (
        lambda cfg: LitSelfSupervised(MAE(encoder=simpleVIT(data_config=cfg))),
        {'test':1,'batch_size':2,'split':0.2},
        simpleVIT,
        TransferLearningVIT,
        """
    0:
        max_transition_epoch: 2
        params:
        - model.linear_head.0.weight
        - model.linear_head.0.bias
        - model.linear_head.1.weight
        - model.linear_head.1.bias
    1:
        lr: 1.0e-05
        max_transition_epoch: 4
        params:
        - model.embedding.pos_embedding
        - model.embedding.to_patch_embedding.1.weight
        - model.embedding.to_patch_embedding.1.bias
        - model.embedding.transformer.layers.0.0.norm.weight
        - model.embedding.transformer.layers.0.0.norm.bias
        - model.embedding.transformer.layers.0.0.to_qkv.weight
        - model.embedding.transformer.layers.0.0.to_out.0.weight
        - model.embedding.transformer.layers.0.0.to_out.0.bias
        - model.embedding.transformer.layers.0.1.net.0.weight
        - model.embedding.transformer.layers.0.1.net.0.bias
        - model.embedding.transformer.layers.0.1.net.1.weight
        - model.embedding.transformer.layers.0.1.net.1.bias
        - model.embedding.transformer.layers.0.1.net.4.weight
        - model.embedding.transformer.layers.0.1.net.4.bias
        - model.embedding.transformer.layers.1.0.norm.weight
        - model.embedding.transformer.layers.1.0.norm.bias
        - model.embedding.transformer.layers.1.0.to_qkv.weight
        - model.embedding.transformer.layers.1.0.to_out.0.weight
        - model.embedding.transformer.layers.1.0.to_out.0.bias
        - model.embedding.transformer.layers.1.1.net.0.weight
        - model.embedding.transformer.layers.1.1.net.0.bias
        - model.embedding.transformer.layers.1.1.net.1.weight
        - model.embedding.transformer.layers.1.1.net.1.bias
        - model.embedding.transformer.layers.1.1.net.4.weight
        - model.embedding.transformer.layers.1.1.net.4.bias
        - model.embedding.transformer.layers.2.0.norm.weight
        - model.embedding.transformer.layers.2.0.norm.bias
        - model.embedding.transformer.layers.2.0.to_qkv.weight
        - model.embedding.transformer.layers.2.0.to_out.0.weight
        - model.embedding.transformer.layers.2.0.to_out.0.bias
        - model.embedding.transformer.layers.2.1.net.0.weight
        - model.embedding.transformer.layers.2.1.net.0.bias
        - model.embedding.transformer.layers.2.1.net.1.weight
        - model.embedding.transformer.layers.2.1.net.1.bias
        - model.embedding.transformer.layers.2.1.net.4.weight
        - model.embedding.transformer.layers.2.1.net.4.bias
        - model.embedding.transformer.layers.3.0.norm.weight
        - model.embedding.transformer.layers.3.0.norm.bias
        - model.embedding.transformer.layers.3.0.to_qkv.weight
        - model.embedding.transformer.layers.3.0.to_out.0.weight
        - model.embedding.transformer.layers.3.0.to_out.0.bias
        - model.embedding.transformer.layers.3.1.net.0.weight
        - model.embedding.transformer.layers.3.1.net.0.bias
        - model.embedding.transformer.layers.3.1.net.1.weight
        - model.embedding.transformer.layers.3.1.net.1.bias
        - model.embedding.transformer.layers.3.1.net.4.weight
        - model.embedding.transformer.layers.3.1.net.4.bias
        - model.embedding.transformer.layers.4.0.norm.weight
        - model.embedding.transformer.layers.4.0.norm.bias
        - model.embedding.transformer.layers.4.0.to_qkv.weight
        - model.embedding.transformer.layers.4.0.to_out.0.weight
        - model.embedding.transformer.layers.4.0.to_out.0.bias
        - model.embedding.transformer.layers.4.1.net.0.weight
        - model.embedding.transformer.layers.4.1.net.0.bias
        - model.embedding.transformer.layers.4.1.net.1.weight
        - model.embedding.transformer.layers.4.1.net.1.bias
        - model.embedding.transformer.layers.4.1.net.4.weight
        - model.embedding.transformer.layers.4.1.net.4.bias

    """,
        4
    ),
    # For the TempCNN classifier
    (
        lambda cfg: LitClassifier(TEMPCNN(data_config=cfg)),
        {'test':1,'batch_size':2,'split':0.2},
        TEMPCNN,
        TransferLearningTempCNN,
                """
    0:
        max_transition_epoch: 5
        params:
        - model.out.bias
        - model.out.weight
    1:
        lr: 0.005
        max_transition_epoch: 8
        params:
        - model.extractor.dense.block.1.bias
        - model.extractor.dense.block.1.weight
        - model.extractor.dense.block.0.bias
        - model.extractor.dense.block.0.weight
        - model.extractor.conv_bn_relu3.block.1.bias
        - model.extractor.conv_bn_relu3.block.1.weight
        - model.extractor.conv_bn_relu3.block.0.bias
        - model.extractor.conv_bn_relu3.block.0.weight
        - model.extractor.conv_bn_relu2.block.1.bias
        - model.extractor.conv_bn_relu2.block.1.weight
        - model.extractor.conv_bn_relu2.block.0.bias
        - model.extractor.conv_bn_relu2.block.0.weight
        - model.extractor.conv_bn_relu1.block.1.bias
        - model.extractor.conv_bn_relu1.block.1.weight
        - model.extractor.conv_bn_relu1.block.0.bias
        - model.extractor.conv_bn_relu1.block.0.weight
    """,
        4
    )
])
def test_transfer(model_func, config, encoder_func, transfer_func, schedule_yaml, epochs):
    args = types.SimpleNamespace(**config)
    xmod = XarrayDataModule(args=args)

    # Write the schedule to a file
    ft_schedule_name = "test_shedule.yaml"
    with open(ft_schedule_name, "w") as f:
        f.write(schedule_yaml)

    # Pre-train
    model = model_func(xmod.config())
    pre_train(model, xmod, epochs)

    # Fine tune
    model_ft = transfer_func(encoder_func(data_config=xmod.config()), data_config=xmod.config())
    model_bc = LitClassifier(model_ft)

    x = fine_tune(ft_schedule_name, model_bc, xmod, epochs)

    assert x[0]["val_acc"] >= 0.0
