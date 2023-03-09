from argparse import ArgumentParser, Namespace
import importlib
from pathlib import Path

import pytorch_lightning as pl
import torch
import numpy as np
import wandb
import random
from datetime import datetime

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised
from finetuning_scheduler import FinetuningScheduler, fts_supporters

DATA_CLASS_MODULE = "hyperiap.datasets"
MODEL_CLASS_MODULE = "hyperiap.models"

# for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pl.seed_everything(1234)


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'hyperiap.models.vit.SimpleVIT'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_from_args(args: Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    point_class = import_class(f"{DATA_CLASS_MODULE}.{args.point_class}")

    data = data_class(args)
    point = point_class(args)

    return data, point


def setup_ss_from_args(args: Namespace, data):
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")
    ssmodel_class = import_class(f"{MODEL_CLASS_MODULE}.{args.ssmodel_class}")

    model = model_class(data_config=data.config(), args=args)
    ssmodel = ssmodel_class(encoder=model, args=args)

    return model, ssmodel


def setup_transfer_from_args(args: Namespace, model: torch.nn.Module, data):
    transfer_class = import_class(f"{MODEL_CLASS_MODULE}.{args.transfer_class}")
    transfer = transfer_class(model, data_config=data.config())

    return transfer


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
        default=True,
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
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    # select point class
    parser.add_argument(
        "--point_class",
        type=str,
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    # select model class
    parser.add_argument(
        "--model_class",
        type=str,
        default="vit.simpleVIT",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    # select selfsupervised class
    parser.add_argument(
        "--ssmodel_class",
        type=str,
        default="mae.MAE",
        help=f"String identifier for the selfsup model class, relative to {MODEL_CLASS_MODULE}.",
    )
    # select transfer class
    parser.add_argument(
        "--transfer_class",
        type=str,
        default="vit.TransferLearningVIT",
        help=f"String identifier for the transfer model class, relative to {MODEL_CLASS_MODULE}.",
    )
    # load ft shceudle
    # "hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml"
    parser.add_argument(
        "--ft_schedule",
        type=str,
        default=None,
        help="path to schedule for finetuing",
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
    # data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    # point_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.point_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")
    ssmodel_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.ssmodel_class}")

    # Get data, model args
    # data_group = parser.add_argument_group("Data Args")
    # data_class.add_to_argparse(data_group)

    # point_group = parser.add_argument_group("Point Args")
    # point_class.add_to_argparse(point_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    litclassmodel_group = parser.add_argument_group("LitClassModel Args")
    LitClassifier.add_to_argparse(litclassmodel_group)

    ssmodel_group = parser.add_argument_group("Self-Sup model Args")
    ssmodel_class.add_to_argparse(ssmodel_group)

    litselfmodel_group = parser.add_argument_group("LitSelfModel Args")
    LitSelfSupervised.add_to_argparse(litselfmodel_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def setup_callbacks(
    args: Namespace, log_dir, model: torch.nn.Module, finetune=False, append=None
):
    """Set up callbacks for the experiment."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    if args.wandb:
        logger = pl.loggers.WandbLogger(
            log_model=True, save_dir=str(log_dir), job_type="train", project="hyperiap"
        )
        logger.watch(model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        logger.experiment.config["model"] = "hyperiap_classifier"
        experiment_dir = logger.experiment.dir
    # -----------
    # callbacks
    # -----------
    filename_run = "class-model-best" + append
    if finetune:
        checkpoint_callback = fts_supporters.FTSCheckpoint(
            save_top_k=1,
            filename=filename_run,
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=False,
            dirpath=experiment_dir,
            every_n_epochs=args.check_val_every_n_epoch,
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            filename=filename_run,
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=False,
            dirpath=experiment_dir,
            every_n_epochs=args.check_val_every_n_epoch,
        )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [summary_callback, learning_rate_callback]

    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    if finetune:
        ft_callback = FinetuningScheduler(ft_schedule=args.ft_schedule)
        callbacks.append(ft_callback)

    # -----------
    # profiling
    # -----------
    if args.profile:
        sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
        profiler = pl.profilers.PyTorchProfiler(
            export_to_chrome=True, schedule=sched, dirpath=experiment_dir
        )
        profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training
    else:
        profiler = pl.profilers.PassThroughProfiler()

    return callbacks, checkpoint_callback, profiler, logger


def main():
    """
     Run an experiment.
     Sample command:
     ```
     python run_all.py --model_class=vit.simpleVIT --ssmodel_class=mae.MAE
     ```
     For basic help documentation, run the command
     ```
     python run_all.py --help
     ```
     The available command line args differ depending on some of the arguments
     including --model_class and --data_class.
     To see which command line args are available and read their documentation
     provide values for those arguments before invoking --help, like so:
     ```
        python run_all.py --model_class=vit.simpleVIT \
            --limit_val_batches=5 --limit_train_batches=10 --max_epochs=5 \
            --wandb --log_every_n_steps=2 \
            --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml
    """
    # seed random with datetime
    random.seed(datetime.now())
    parser = _setup_parser()
    args = parser.parse_args()

    if args.ft_schedule is None:
        raise ValueError("Must provide a finetuning schedule")

    data, point = setup_data_from_args(args)
    model, ssmodel = setup_ss_from_args(args, data)

    # -----------
    # setup models
    # -----------
    seq_model_class = LitClassifier
    ss_model_class = LitSelfSupervised
    log_dir = Path("training") / "logs"

    wandb.init(project="hyperiap")
    run_id = wandb.run.id

    # -----------
    # ss model
    # -----------

    seq_ss_model = ss_model_class(args=args, model=ssmodel)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args, log_dir=log_dir, model=seq_ss_model, finetune=False, append="_ss"
    )
    callbacks.append(checkpoint_callback)

    # fit
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.profiler = profiler
    trainer.fit(seq_ss_model, datamodule=data)

    # save best model in useable format
    seq_ss_model = ss_model_class.load_from_checkpoint(
        checkpoint_callback.best_model_path, args=args, model=ssmodel
    )
    litclass = seq_model_class(seq_ss_model.model.encoder)
    trainer = pl.Trainer(limit_val_batches=0, enable_checkpointing=False, logger=False)
    trainer.validate(litclass, datamodule=data)
    ss_checkpoint = logger.experiment.dir + "/ss_classifier.ckpt"
    trainer.save_checkpoint(ss_checkpoint)

    # end wandb experiment
    # wandb.finish()

    # -----------
    # noisy training
    # -----------
    wandb.init(id=run_id, resume="must")

    seq_model = seq_model_class.load_from_checkpoint(
        ss_checkpoint, args=args, model=model
    )

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args, log_dir=log_dir, model=seq_model, finetune=False, append="_noisy"
    )
    run_id = logger.version
    callbacks.append(checkpoint_callback)

    # train on noisy lables
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.profiler = profiler
    trainer.fit(seq_model, datamodule=data)

    noisy_checkpoint = checkpoint_callback.best_model_path
    # end wandb experiment
    # wandb.finish()

    # -----------
    # clean training
    # -----------
    wandb.init(id=run_id, resume="must")

    seq_model = seq_model_class.load_from_checkpoint(
        noisy_checkpoint, args=args, model=model
    )
    # transfer = setup_transfer_from_args(args, seq_model.model, data)
    transfer = setup_transfer_from_args(args, seq_model.model, point)
    seq_model = seq_model_class(args=args, model=transfer)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args, log_dir=log_dir, model=seq_model, finetune=False, append="_clean"
    )
    run_id = logger.version
    callbacks.append(checkpoint_callback)

    # train on clean lables
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.profiler = profiler
    # trainer.fit(seq_model, datamodule=data)
    trainer.fit(seq_model, datamodule=point)

    trainer.profiler = (
        pl.profilers.PassThroughProfiler()
    )  # turn profiling off during testing

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    if args.wandb:
        print("Best model also uploaded to W&B")

    # do we automatically test the best model?
    # trainer.test(seq_model, datamodule=data)

    # create a random number
    random_number = random.randint(1, 100000)
    wandb.log({"test_loss": random_number})
    # end wandb experiment
    wandb.finish()


if __name__ == "__main__":
    main()
