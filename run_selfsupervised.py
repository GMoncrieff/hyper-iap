from pathlib import Path

import pytorch_lightning as pl
import torch
import numpy as np

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

from utils.run_helpers import setup_data_from_args, setup_models_from_args
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
     python run_selfsupervised.py --model_class=vit.simpleVIT --ssmodel_class=mae.MAE --data_class=xarray_module.XarrayDataModule
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
     python run_selfsupervised.py --model_class=vit.simpleVIT --ssmodel_class=mae.MAE --data_class=xarray_module.XarrayDataModule --help
     python run_selfsupervised.py --model_class=vit.simpleVIT \
            --ssmodel_class=mae.MAE \
            --save_classifier \
            --data_class=xarray_module.XarrayDataModule \
            --limit_val_batches=25 --limit_train_batches=25 --max_epochs=10 \
            --wandb

    """
    pl.seed_everything(1234)

    parser = setup_parser(
        model_module=MODEL_CLASS_MODULE,
        ss_module=MODEL_CLASS_MODULE,
        data_module=DATA_CLASS_MODULE,
        point_module=DATA_CLASS_MODULE,
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="If passed, loads a model from the provided path. Must be provided when finetuning",
    )
    # save classifier
    parser.add_argument(
        "--save_classifier",
        action="store_true",
        default=True,
        help="If passed, will save the encoder state of best model as a pl classifier.",
    )
    args = parser.parse_args()

    data, _ = setup_data_from_args(
        args, data_module=DATA_CLASS_MODULE, point_module=DATA_CLASS_MODULE
    )

    model, ssmodel = setup_models_from_args(
        args, data, ss_module=MODEL_CLASS_MODULE, model_module=MODEL_CLASS_MODULE
    )
    # -----------
    # setup model
    # -----------
    seq_model_class = LitSelfSupervised
    log_dir = Path("training") / "ss_logs"

    if args.load_checkpoint is not None:
        seq_model = seq_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=ssmodel
        )
    else:
        seq_model = seq_model_class(args=args, model=ssmodel)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args, log_dir=log_dir, model=seq_model
    )
    callbacks.append(checkpoint_callback)

    # -----------
    # training
    # -----------

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.profiler = profiler

    trainer.fit(seq_model, datamodule=data)

    trainer.profiler = (
        pl.profilers.PassThroughProfiler()
    )  # turn profiling off during testing

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    if args.wandb:
        print("Best model also uploaded to W&B")

    if args.save_classifier:
        seq_model = seq_model_class.load_from_checkpoint(
            best_model_path, args=args, model=ssmodel
        )
        # junk to allow us to save model in a form that classifier can load
        litclass = LitClassifier(seq_model.model.encoder)
        trainer = pl.Trainer(
            limit_val_batches=0, enable_checkpointing=False, logger=False
        )
        trainer.validate(litclass, datamodule=data)
        trainer.save_checkpoint(logger.experiment.dir + "/ss_classifier.ckpt")

    # trainer.test(seq_model, datamodule=data)


if __name__ == "__main__":
    main()
