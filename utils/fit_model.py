import torch
import lightning as pl
import argparse
import wandb
from typing import Tuple

from utils.run_helpers import setup_transfer_from_args, setup_callbacks


def fit(
    args: argparse.Namespace,
    arg_groups,
    data: pl.LightningDataModule,
    max_epoch: int,
    model: torch.nn.Module,
    log_dir: str,
    stage: str,
    lit_sup_model: pl.LightningModule,
    lit_ss_model: pl.LightningModule,
    checkpoint: str = "",
    lsmooth: float = None,
    module: str = "hyperiap.models",
    run_id: str = None,
    logmetric: str = "val_loss",
    mode: str = "min",
) -> Tuple[str, str]:
    """
    Trains a PyTorch model using PyTorch Lightning
    Args:
        args (argparse.Namespace): Arguments for the model and training.
        arg_groups: Argument groups for the trainer.
        data (pl.LightningDataModule): Data module containing the dataset used for training.
        run_id (str): Unique identifier for the Weights & Biases run.
        max_epoch (int): Maximum number of epochs for training.
        model (torch.nn.Module): PyTorch model to be trained.
        log_dir (str): Path to the directory for logging training information.
        stage (str): Stage of the training process, such as 'pretrain', 'finetune', etc.
        lit_sup_model (pl.LightningModule): PyTorch Lightning model class.
        lit_ss_model (pl.LightningModule): PyTorch Lightning self-supervised model class.
        module (str, optional): Import path for the model module. Defaults to "hyperiap.models".
        lsmooth (float): Label smoothing value.
        checkpoint (str): Path to the checkpoint file for fine-tuning a pre-trained model.
        logmetric (str): Metric to log for the model.
        mode (str): Mode for the metric to log, either 'min' or 'max'.

    Returns:
        Tuple[str, str]: A tuple containing the Path to the new checkpoint and best val loss.
    """
    # set lit model
    if stage == "ss":
        lit_model = lit_ss_model
    else:
        lit_model = lit_sup_model

    if lsmooth:
        # change label smoothing
        args.label_smooth = lsmooth

    # set loss to monitor
    args.monitor = f"{stage}_"

    # stage specifc epochs
    arg_groups["Trainer Args"].max_epochs = max_epoch
    args.max_epochs = max_epoch

    if args.wandb:
        wandb.init(id=run_id, resume="must")

    # setup transfer model if finetuning
    if bool(checkpoint):
        print("loading checkpoint: ", checkpoint)
        seq_model = lit_model.load_from_checkpoint(checkpoint, args=args, model=model)
        transfer = setup_transfer_from_args(
            args, seq_model.model, data, model_module=module
        )
        seq_model = lit_model(args=args, model=transfer)
    else:
        print("no checkpoint found, training from scratch")
        seq_model = lit_model(args=args, model=model)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args,
        log_dir=log_dir,
        model=seq_model,
        finetune=bool(checkpoint),
        append=f"_{stage}",
        log_metric=f"{stage}_{logmetric}",
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    # train
    trainer = pl.Trainer(
        **vars(arg_groups["Trainer Args"]), callbacks=callbacks, logger=logger
    )
    trainer.profiler = profiler
    trainer.fit(seq_model, datamodule=data)

    # valiadate best model
    valid = []
    if stage == "clean":
        valid = trainer.validate(seq_model, datamodule=data, ckpt_path="best")

    # get checkpoint details
    new_checkpoint = checkpoint_callback.best_model_path
    best_val_target = checkpoint_callback.best_model_score.item()

    # resave checkpoint in format useable for finetuning if training is self supervised
    if stage == "ss":
        seq_model = lit_model.load_from_checkpoint(
            new_checkpoint, args=args, model=model
        )
        litclass = lit_sup_model(seq_model.model.encoder)
        trainer = pl.Trainer(
            limit_val_batches=0, enable_checkpointing=False, logger=False
        )
        trainer.validate(litclass, datamodule=data)
        if args.wandb:
            new_checkpoint = logger.experiment.dir + "/ss_classifier.ckpt"
        else:
            new_checkpoint = logger.log_dir + "/ss_classifier.ckpt"
        trainer.save_checkpoint(new_checkpoint)

    return new_checkpoint, best_val_target, valid
