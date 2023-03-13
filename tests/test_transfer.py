from pytorch_lightning import Trainer, seed_everything
from finetuning_scheduler import FinetuningScheduler

from hyperiap.models.vit import simpleVIT, TransferLearningVIT
from hyperiap.models.mae import MAE
from hyperiap.datasets.xarray_module import XarrayDataModule

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

seed_everything(1234)


def test_transfer_classifier():

    ft_schedule_yaml = """
    0:
        max_transition_epoch: 1
        params:
        - model.linear_head.0.weight
        - model.linear_head.0.bias
        - model.linear_head.1.weight
        - model.linear_head.1.bias
    1:
        lr: 1.0e-05
        max_transition_epoch: -1
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

    """

    ft_schedule_name = "test_shedule.yaml"
    # Let's write the schedule to a file so we can simulate loading an explicitly defined fine-tuning
    # schedule.
    with open(ft_schedule_name, "w") as f:
        f.write(ft_schedule_yaml)

    # pre-train
    xmod = XarrayDataModule()
    encoder = simpleVIT(data_config=xmod.config())
    ss_model = MAE(encoder=encoder)

    model = LitSelfSupervised(ss_model)

    trainer = Trainer(limit_train_batches=2, limit_val_batches=2, max_epochs=2)
    trainer.fit(model, datamodule=xmod)

    # fine tune
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[FinetuningScheduler(ft_schedule=ft_schedule_name)],
    )
    model_ft = TransferLearningVIT(encoder, data_config=xmod.config())
    # model_ft=simpleVIT(data_config=xmod.config())
    model_bc = LitClassifier(model_ft)

    # trainer = Trainer(limit_train_batches=3, limit_val_batches=1, max_epochs=1)
    trainer.fit(model_bc, datamodule=xmod)

    x = trainer.validate(datamodule=xmod)

    assert x[0]["val_acc"] >= 0.0
