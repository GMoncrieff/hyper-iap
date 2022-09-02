from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.mlp import MLP
from hyperiap.models.baseclassifier import BaseClassifier
from hyperiap.datasets.mnist import MNISTDataModule


def test_lit_classifier():
    seed_everything(1234)

    model = BaseClassifier(MLP())
    mnist = MNISTDataModule("")
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, datamodule=mnist)

    results = trainer.test(datamodule=mnist)
    assert results[0]['test_acc'] > 0.7
