from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
#from torchvision.datasets.mnist import MNIST
from torchvision import transforms

#from hyperiap.models.mlp import MLP
from hyperiap.models.cnn1d import TempCNN
from hyperiap.models.baseclassifier import BaseClassifier
#from hyperiap.datasets.mnist import MNISTDataModule
from hyperiap.datasets.timeseries import TimeseriesDataset
from hyperiap.datasets.timeseries_module import TimeSeriesDataModule


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    #mnist = MNISTDataModule("")
    ts = TimeSeriesDataModule()

    # ------------
    # model
    # ------------
    #model = BaseClassifier(MLP(hidden_dim=args.hidden_dim), args.learning_rate)
    model = BaseClassifier(TempCNN(args.learning_rate))

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    #trainer.fit(model, datamodule=mnist)
    trainer.fit(model, datamodule=ts)
    
    # ------------
    # testing
    # ------------
    #result = trainer.test(datamodule=mnist)
    result = trainer.test(datamodule=ts)
    print(result)


if __name__ == "__main__":
    cli_main()
