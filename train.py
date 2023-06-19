from pathlib import Path

import argparse
import lightning as pl
import torch
import numpy as np
import wandb
import random
from datetime import datetime

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

from utils.run_helpers import (
    setup_data_from_args,
    setup_model_from_args,
    setup_ssmodel_from_args,
    setup_parser,
)
from utils.fit_model import fit

DATA_CLASS_MODULE = "hyperiap.datasets"
MODEL_CLASS_MODULE = "hyperiap.models"

# for reproducibility
pl.seed_everything(1234)
torch.set_float32_matmul_precision("medium")


def main():
    """
     Run an experiment.
     Sample command:
     ```
     python train.py --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml
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
        # a simple run
        python train.py --model_class=vit.simpleVIT \
            --limit_val_batches=5 --limit_train_batches=10 --max_epochs=5

        # a run with all stages
        python train.py --model_class=vit.simpleVIT \
            --limit_val_batches=3 --limit_train_batches=3 --val_check_interval=1.0\
            --lr=0.001 --lr_ss=0.001 --lr_ft=0.0001 \
            --max_epochs_ss=2 --max_epochs_noisy=2 --max_epochs_clean=5 --log_every_n_steps=5\
            --ft_schedule=hyperiap/litmodels/LitClassifier_vit_ft_schedule.yaml \
            --run_ss --run_noisy --run_clean --precision=16

        # a run with tempcnn
        python train.py --model_class=tempcnn.TEMPCNN \
            --limit_val_batches=2 --limit_train_batches=5 \
            --lr_ft=0.0000001 \
            --max_epochs_noisy=10 --max_epochs_clean=6 --log_every_n_steps=1 \
            --transfer_class=tempcnn.TransferLearningTempCNN \
            --ft_schedule=hyperiap/litmodels/LitClassifier_tempcnn_ft_schedule.yaml \
            --wandb --run_noisy --run_clean
    """

    # seed random with datetime
    random.seed(datetime.now())
    parser = setup_parser(
        model_module=MODEL_CLASS_MODULE,
        ss_module=MODEL_CLASS_MODULE,
        data_module=DATA_CLASS_MODULE,
        point_module=DATA_CLASS_MODULE,
    )

    args = parser.parse_args()

    # split args into groups
    arg_groups = {}

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    if args.ft_schedule is None:
        raise ValueError("Must provide a finetuning schedule")

    data, point = setup_data_from_args(
        args, data_module=DATA_CLASS_MODULE, point_module=DATA_CLASS_MODULE
    )

    model = setup_model_from_args(args, data, model_module=MODEL_CLASS_MODULE)
    point_model = setup_model_from_args(args, point, model_module=MODEL_CLASS_MODULE)

    if args.run_ss:
        ssmodel = setup_ssmodel_from_args(args, model, ss_module=MODEL_CLASS_MODULE)

    # -----------
    # setup models
    # -----------
    seq_model_class = LitClassifier
    ss_model_class = LitSelfSupervised
    log_dir = Path("training") / "logs"
    checkpoint = args.checkpoint
    run_id = None

    if args.wandb:
        run_id = wandb.util.generate_id()
        wandb.init(
            project="hyperiap",
            id=run_id,
            dir=log_dir,
            allow_val_change=True,
            resume="allow",
        )

    if args.run_ss:
        # -----------
        # ss model
        # -----------

        # stage specifc epochs
        if args.max_epochs_ss > 0:
            max_epoch = args.max_epochs_ss
        else:
            max_epoch = arg_groups["Trainer Args"].max_epochs

        checkpoint, best_val_loss = fit(
            args=args,
            arg_groups=arg_groups,
            data=data,
            max_epoch=max_epoch,
            model=ssmodel,
            log_dir=log_dir,
            stage="ss",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
            run_id=run_id,
        )

        print("ss model checkpoint: ", checkpoint)

    if args.run_noisy:
        # -----------
        # noisy training
        # -----------

        # stage specifc epochs
        if args.max_epochs_noisy > 0:
            max_epoch = args.max_epochs_noisy
        else:
            max_epoch = arg_groups["Trainer Args"].max_epochs

        checkpoint, best_val_loss = fit(
            args=args,
            arg_groups=arg_groups,
            data=data,
            max_epoch=max_epoch,
            model=model,
            log_dir=log_dir,
            stage="noisy",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
            checkpoint=checkpoint,
            lsmooth=args.ls_modifier,
            run_id=run_id,
        )

        print("noisy model checkpoint: ", checkpoint)

    if args.run_clean:
        # -----------
        # clean training
        # -----------

        # stage specifc epochs
        if args.max_epochs_clean > 0:
            max_epoch = args.max_epochs_clean
        else:
            max_epoch = arg_groups["Trainer Args"].max_epochs

        if bool(checkpoint):
            # if there we are finetuning, we need to load weights into original model
            clean_model = model
        else:
            # otherwise we create a fresh model
            clean_model = point_model

        checkpoint, best_val_loss = fit(
            args=args,
            arg_groups=arg_groups,
            data=point,
            max_epoch=max_epoch,
            model=clean_model,
            log_dir=log_dir,
            stage="clean",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
            checkpoint=checkpoint,
            lsmooth=args.ls_modifier,
            run_id=run_id,
        )

        print("clean model checkpoint: ", checkpoint)

    if args.wandb:
        # log best validation loss at end of pipeline
        wandb.log({"final_loss": best_val_loss})
        # end wandb experiment
        wandb.finish()


if __name__ == "__main__":
    main()
