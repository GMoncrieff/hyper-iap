from argparse import ArgumentParser, Namespace
import importlib
from pathlib import Path

import pytorch_lightning as pl
import torch
import numpy as np

from hyperiap.models.baseclassifier import BaseClassifier

DATA_CLASS_MODULE = "hyperiap.datasets"
MODEL_CLASS_MODULE = "hyperiap.models"

# for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'hyperiap.models.tempcnn.TEMPCNN'."""
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

    # wandb_logger
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    # pytorch profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )
    # select data class
    parser.add_argument(
        "--data_class",
        type=str,
        default="timeseries_module.TimeSeriesDataModule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    # select model class
    parser.add_argument(
        "--model_class",
        type=str,
        default="tempcnn.TEMPCNN",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    # load from checkpoint
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="If passed, loads a model from the provided path.",
    )
    # early stopping
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument.",
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
    """
     Run an experiment.
     Sample command:
     ```
    python run_classifier.py --model_class=tempcnn.TEMPCNN --data_class=timeseries_module.TimeSeriesDataModule
     ```
     For basic help documentation, run the command
     ```
     python run_classifier.py --help
     ```
     The available command line args differ depending on some of the arguments
     including --model_class and --data_class.
     To see which command line args are available and read their documentation
     provide values for those arguments before invoking --help, like so:
     ```
     python run_classifier.py --model_class=tempcnn.TEMPCNN --data_class=timeseries_module.TimeSeriesDataModule --help
    """
    pl.seed_everything(1234)

    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    # -----------
    # setup model
    # -----------
    seq_model_class = BaseClassifier

    if args.load_checkpoint is not None:
        seq_model = seq_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        seq_model = seq_model_class(args=args, model=model)

    # -----------
    # logging
    # -----------
    log_dir = Path("training") / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    if args.wandb:
        logger = pl.loggers.WandbLogger(
            log_model="all", save_dir=str(log_dir), job_type="train"
        )
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
    # -----------
    # callbacks
    # -----------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        filename="epoch={epoch:04d}-val.loss={val/loss:.3f}-val.acc={val/acc:.3f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )
    summary_callback = pl.callbacks.ModelSummary(max_depth=2)
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    callbacks = [summary_callback, checkpoint_callback, learning_rate_callback]

    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    # -----------
    # profiling
    # -----------
    if args.profile:
        sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
        profiler = pl.profiler.PyTorchProfiler(
            export_to_chrome=True, schedule=sched, dirpath=experiment_dir
        )
        profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training
    else:
        profiler = pl.profiler.PassThroughProfiler()

    # -----------
    # training
    # -----------

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.profiler = profiler

    # trainer.tune(seq_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(seq_model, datamodule=data)

    trainer.profiler = (
        pl.profiler.PassThroughProfiler()
    )  # turn profiling off during testing

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        f"Best model saved at: {best_model_path}"
        # Hide lines below until Lab 04
        if args.wandb:
            "Best model also uploaded to W&B"
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(seq_model, datamodule=data)


if __name__ == "__main__":
    cli_main()
