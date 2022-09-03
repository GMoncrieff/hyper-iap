from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.cnn1d import TempCNN
from hyperiap.models.baseclassifier import BaseClassifier
from hyperiap.datasets.timeseries_module import TimeSeriesDataModule


def test_ts_classifier():
    seed_everything(1234)

    ts = TimeSeriesDataModule()
    model = BaseClassifier(TempCNN())
    
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, datamodule=ts)

    results = trainer.test(datamodule=ts)
    assert results[0]['test_acc'] > 0.7
