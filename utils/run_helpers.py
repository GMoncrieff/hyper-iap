from argparse import ArgumentParser, Namespace
import yaml
import tempfile
import importlib
from pathlib import Path
import torch
import lightning as pl

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
    trainer_group = parser.add_argument_group("Trainer Args")
    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_group.add_argument(
        "--limit_val_batches",
        type=int,
        default=2,
        help="limit_val_batches",
    )
    trainer_group.add_argument(
        "--limit_train_batches",
        type=int,
        default=2,
        help="limit_train_batches",
    )
    trainer_group.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        help="max epochs",
    )
    trainer_group.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check_val_every_n_epoch",
    )
    trainer_group.add_argument(
        "--log_every_n_steps",
        type=int,
        default=2,
        help="--log_every_n_steps",
    )
    trainer_group.add_argument(
        "--precision",
        type=int,
        default=32,
        help="--precision",
    )

    setup_group = parser.add_argument_group("Setup Args")
    # wandb_logger
    setup_group.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    # pytorch profiling
    setup_group.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )

    # select data class
    setup_group.add_argument(
        "--data_class",
        type=str,
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the data class, relative to {data_module}.",
    )
    # select point class
    setup_group.add_argument(
        "--point_class",
        type=str,
        default="xarray_module.XarrayDataModule",
        help=f"String identifier for the point data class, relative to {point_module}.",
    )
    # select model class
    setup_group.add_argument(
        "--model_class",
        type=str,
        default="vit.simpleVIT",
        help=f"String identifier for the model class, relative to {model_module}.",
    )
    # select selfsupervised class
    setup_group.add_argument(
        "--ssmodel_class",
        type=str,
        default="mae.MAE",
        help=f"String identifier for the selfsup model class, relative to {ss_module}.",
    )
    # select transfer class
    setup_group.add_argument(
        "--transfer_class",
        type=str,
        default="vit.TransferLearningVIT",
        help=f"String identifier for the transfer model class, relative to {model_module}.",
    )
    # load ft shceudle
    # "hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml"
    setup_group.add_argument(
        "--ft_schedule",
        type=str,
        default="hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml",
        help="path to schedule for finetuing",
    )
    # lr finetune
    setup_group.add_argument(
        "--lr_ft",
        type=float,
        default=1000,
        help="finetuing learning rate",
    )
    # early stopping
    setup_group.add_argument(
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
    log_metric="val_loss",
):
    """Set up callbacks for training, including logging, checkpointing, and early stopping."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = pl.pytorch.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    if args.wandb:
        logger = pl.pytorch.loggers.WandbLogger(
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
            monitor=log_metric,
            mode="min",
            auto_insert_metric_name=False,
            dirpath=experiment_dir,
            every_n_epochs=args.check_val_every_n_epoch,
        )
    else:
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            save_top_k=1,
            filename=filename_run,
            monitor=log_metric,
            mode="min",
            auto_insert_metric_name=False,
            dirpath=experiment_dir,
            every_n_epochs=args.check_val_every_n_epoch,
        )

    summary_callback = pl.pytorch.callbacks.ModelSummary(max_depth=2)
    learning_rate_callback = pl.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    callbacks = [summary_callback, learning_rate_callback]

    if args.stop_early:
        early_stopping_callback = pl.pytorch.callbacks.EarlyStopping(
            monitor=log_metric, mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    if finetune:
        # open ft schedule and change lr
        with open(args.ft_schedule, "r") as file:
            data = yaml.safe_load(file)
            data[0]["max_transition_epoch"] = int(args.max_epochs / 2)
            data[1]["max_transition_epoch"] = args.max_epochs
            data[1]["lr"] = args.lr_ft

        # create tempfile with new lr
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_file:
            yaml.safe_dump(data, temp_file)

        # setup caback
        ft_callback = FinetuningScheduler(
            ft_schedule=temp_file.name,
            base_max_lr=args.lr,
            epoch_transitions_only=True,
        )
        callbacks.append(ft_callback)

    # -----------
    # profiling
    # -----------
    if args.profile:
        sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
        profiler = pl.pytorch.profilers.PyTorchProfiler(
            export_to_chrome=True, schedule=sched, dirpath=experiment_dir
        )
        profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training
    else:
        profiler = pl.pytorch.profilers.PassThroughProfiler()

    return callbacks, checkpoint_callback, profiler, logger
