from pytorch_lightning import Trainer, seed_everything
from hyperiap.models.tempcnn import TEMPCNN
#from hyperiap.models.vit import VIT
from hyperiap.models.baseclassifier import BaseClassifier
from hyperiap.datasets.timeseries_module import TimeSeriesDataModule


def test_ts_classifier():
    seed_everything(1234)

    ts = TimeSeriesDataModule()
    
    model = BaseClassifier(TEMPCNN(data_config=ts.config()))
    
    """ vitmodel = VIT(
    image_size = 3,
    near_band = 3,
    num_patches = 500,
    num_classes = 2,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = 'CAF'
    )
    model = BaseClassifier(vitmodel) """
    
    trainer = Trainer(limit_train_batches=50, limit_val_batches=0, max_epochs=2)
    trainer.fit(model, datamodule=ts)

    results = trainer.test(datamodule=ts)
    assert results[0]['test/acc'] > 0.7
