from argparse import ArgumentParser, Namespace
import importlib
from pathlib import Path
import torch
import pytorch_lightning as pl

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised
from finetuning_scheduler import FinetuningScheduler, fts_supporters


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'hyperiap.models.vit.SimpleVIT'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_from_args(args: Namespace, data_module: str, point_module: str):
    """setup data loaders"""
    data_class = import_class(f"{data_module}.{args.data_class}")
    point_class = import_class(f"{point_module}.{args.point_class}")

    data = data_class(args)
    point = point_class(args)

    return data, point


def setup_models_from_args(args: Namespace, data, ss_module: str, model_module: str):
    """setup pl models"""
    model_class = import_class(f"{model_module}.{args.model_class}")
    ssmodel_class = import_class(f"{ss_module}.{args.ssmodel_class}")

    model = model_class(data_config=data.config(), args=args)
    ssmodel = ssmodel_class(encoder=model, args=args)

    return model, ssmodel


def setup_transfer_from_args(
    args: Namespace, model: torch.nn.Module, data, model_module: str
):
    """setup pl model for trasnfer learning"""
    transfer_class = import_class(f"{model_module}.{args.transfer_class}")
    transfer = transfer_class(model, data_config=data.config())

    return transfer


def setup_parser(
    model_module: str, ss_module: str, data_module: str, point_module: str
):
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
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the data class, relative to {data_module}.",
    )
    # select point class
    parser.add_argument(
        "--point_class",
        type=str,
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the point data class, relative to {point_module}.",
    )
    # select model class
    parser.add_argument(
        "--model_class",
        type=str,
        default="vit.simpleVIT",
        help=f"String identifier for the model class, relative to {model_module}.",
    )
    # select selfsupervised class
    parser.add_argument(
        "--ssmodel_class",
        type=str,
        default="mae.MAE",
        help=f"String identifier for the selfsup model class, relative to {ss_module}.",
    )
    # select transfer class
    parser.add_argument(
        "--transfer_class",
        type=str,
        default="vit.TransferLearningVIT",
        help=f"String identifier for the transfer model class, relative to {model_module}.",
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
    # data_class = import_class(f"{data_module}.{temp_args.data_class}")
    # point_class = import_class(f"{point_module}.{temp_args.point_class}")
    model_class = import_class(f"{model_module}.{temp_args.model_class}")
    ssmodel_class = import_class(f"{ss_module}.{temp_args.ssmodel_class}")

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
    args: Namespace,
    log_dir: str,
    model: torch.nn.Module,
    finetune=False,
    append="",
    project="hyperiap",
):
    """Set up callbacks for training, including logging, checkpointing, and early stopping."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    if args.wandb:
        logger = pl.loggers.WandbLogger(
            log_model=True, save_dir=str(log_dir), job_type="train", project=project
        )
        logger.watch(model, log_freq=max(50, args.log_every_n_steps))
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
