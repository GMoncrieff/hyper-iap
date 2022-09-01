from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from hyperiap.models.mlp import MLP
from hyperiap.models.baseclassifier import BaseClassifier


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
    dataset = MNIST("hyperiap/datasets", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST("hyperiap/datasets", train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = BaseClassifier(MLP(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == "__main__":
    cli_main()
