from pathlib import Path
import argparse

import lightning as pl
import torch
import numpy as np

from hyperiap.litmodels.litclassifier import LitClassifier

from utils.run_helpers import (
    setup_data_from_args,
    setup_models_from_args,
    setup_transfer_from_args,
)
from utils.run_helpers import setup_callbacks, setup_parser

DATA_CLASS_MODULE = "hyperiap.datasets"
MODEL_CLASS_MODULE = "hyperiap.models"

# for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pl.seed_everything(1234)


def main():
    """
     Run an experiment.
     Sample command:
     ```
    python run_classifier.py --model_class=vit.simpleVIT --data_class=xarray_module.XarrayDataModule
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
     python run_classifier.py --model_class=vit.simpleVIT --data_class=xarray_module.XarrayDataModule --help

     python run_classifier.py \
            --model_class=vit.simpleVIT \
            --data_class=xarray_module.XarrayDataModule \
            --limit_val_batches=5 --limit_train_batches=5 --max_epochs=10\
            --wandb --log_every_n_steps=2 --monitor=try_

    python run_classifier.py --model_class=vit.simpleVIT \
            --data_class=xarray_module.XarrayDataModule \
            --limit_val_batches=5 --limit_train_batches=10 --max_epochs=5 \
            --load_checkpoint=training/ss_logs/wandb/run-20230307_143609-18sqhvq1/files/ss_classifier.ckpt \
            --wandb --log_every_n_steps=2 \
            --finetune \
            --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml
    """
    # -----------
    # setup inputs and args
    # -----------
    parser = setup_parser(
        model_module=MODEL_CLASS_MODULE,
        ss_module=MODEL_CLASS_MODULE,
        data_module=DATA_CLASS_MODULE,
        point_module=DATA_CLASS_MODULE,
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="If passed, uses transfer learning to fine tune model",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="If passed, loads a model from the provided path. Must be provided when finetuning",
    )
    args = parser.parse_args()

    data, point = setup_data_from_args(
        args, data_module=DATA_CLASS_MODULE, point_module=DATA_CLASS_MODULE
    )

    model, _ = setup_models_from_args(
        args, data, ss_module=MODEL_CLASS_MODULE, model_module=MODEL_CLASS_MODULE
    )

    # -----------
    # setup model
    # -----------
    seq_model_class = LitClassifier
    log_dir = Path("training") / "logs"

    if args.finetune and (args.load_checkpoint is None):
        raise ValueError("Must provide a checkpoint when finetuning")
    if args.finetune and (args.ft_schedule is None):
        raise ValueError("Must provide a schedule when finetuning")

    if args.load_checkpoint is not None:
        seq_model = seq_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
        if args.finetune:
            transfer = setup_transfer_from_args(
                args, seq_model.model, point, model_module=MODEL_CLASS_MODULE
            )
            # transfer = setup_transfer_from_args(args,model,data)
            seq_model = seq_model_class(args=args, model=transfer)

    else:
        seq_model = seq_model_class(args=args, model=model)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args,
        log_dir=log_dir,
        model=seq_model,
        finetune=args.finetune,
        log_metric=f"{args.monitor}val_loss",
    )
    callbacks.append(checkpoint_callback)

    # -----------
    # training
    # -----------
    arg_groups = {}

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    trainer = pl.Trainer(
        **vars(arg_groups["Trainer Args"]), callbacks=callbacks, logger=logger
    )
    trainer.profiler = profiler

    trainer.fit(seq_model, datamodule=data)

    trainer.profiler = (
        pl.pytorch.profilers.PassThroughProfiler()
    )  # turn profiling off during testing

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    if args.wandb:
        print("Best model also uploaded to W&B")

    # trainer.test(seq_model, datamodule=data)


if __name__ == "__main__":
    main()
