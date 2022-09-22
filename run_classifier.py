from argparse import ArgumentParser, Namespace
import importlib

import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

#from hyperiap.models.mlp import MLP
from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.baseclassifier import BaseClassifier
from hyperiap.datasets.timeseries_module import TimeSeriesDataModule

DATA_CLASS_MODULE = "hyperiap.datasets"
MODEL_CLASS_MODULE = "hyperiap.models"

#for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'hyperiap.models.TEMPCNN'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def setup_data_and_model_from_args(args: Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    return data, model

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = ArgumentParser(add_help=False, parents=[trainer_parser])
    parser.set_defaults(max_epochs=1)

    #wandb_logger
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    #pytorch profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )
    #select data class
    parser.add_argument(
        "--data_class",
        type=str,
        default="TimeSeriesDataModule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    #select model class
    parser.add_argument(
        "--model_class",
        type=str,
        default="TEMPCNN",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    #load from checkpoint
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    #early stopping
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    basemodel_group = parser.add_argument_group("BaseModel Args")
    BaseClassifier.add_to_argparse(basemodel_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

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
    ts = TimeSeriesDataModule()

    # ------------
    # model
    # ------------
    #model = BaseClassifier(MLP(hidden_dim=args.hidden_dim), args.learning_rate)
    model = BaseClassifier(TempCNN())
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=ts)
    
    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=ts)
    print(result)


if __name__ == "__main__":
    cli_main()
