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
np.random.seed(42)
torch.manual_seed(42)
pl.seed_everything(1234)


def main():
    """
     Run an experiment.
     Sample command:
     ```
     python run_all.py --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml
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
            --log_every_n_steps=2 \
            --lr_ft=10000 \
            --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml \
            --wandb

        python run_all.py --model_class=vit.simpleVIT \
            --limit_val_batches=5 --limit_train_batches=10 \
            --max_epochs_ss=6 --max_epochs_noisy=6 --max_epochs_clean=6 \
            --log_every_n_steps=1 \
            --lr_ft=0.01 \
            --ft_schedule=hyperiap/litmodels/LitClassifier_ft_schedule_final.yaml \
            --wandb
    """
    # seed random with datetime
    random.seed(datetime.now())
    parser = setup_parser(
        model_module=MODEL_CLASS_MODULE,
        ss_module=MODEL_CLASS_MODULE,
        data_module=DATA_CLASS_MODULE,
        point_module=DATA_CLASS_MODULE,
    )
    # labels moothing modifier
    parser.add_argument(
        "--ls_modifier",
        type=float,
        default=0.2,
        help="modifier for label smoothing when training on noisy labels",
    )
    # ss stage epochs
    parser.add_argument(
        "--max_epochs_ss",
        type=int,
        default=0,
        help="num epochs to train ss model",
    )
    # noisy stage epochs
    parser.add_argument(
        "--max_epochs_noisy",
        type=int,
        default=0,
        help="num epochs to train noisy model",
    )
    # clean stage epochs
    parser.add_argument(
        "--max_epochs_clean",
        type=int,
        default=0,
        help="num epochs to train clean model",
    )
    # do we run ss training
    parser.add_argument(
        "--run_ss",
        action="store_true",
        default=True,
        help="run ss training",
    )
    # do we run noisy training
    parser.add_argument(
        "--run_noisy",
        action="store_true",
        default=True,
        help="run noisy training",
    )
    # do we run clean training
    parser.add_argument(
        "--run_clean",
        action="store_true",
        default=True,
        help="run clean training",
    )
    # do we run test data
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="run test data",
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

    model = setup_model_from_args(
        args, data, ss_module=MODEL_CLASS_MODULE, model_module=MODEL_CLASS_MODULE
    )

    if args.run_ss:
        ssmodel = setup_ssmodel_from_args(args, model, ss_module=MODEL_CLASS_MODULE)

    # -----------
    # setup models
    # -----------
    seq_model_class = LitClassifier
    ss_model_class = LitSelfSupervised
    log_dir = Path("training") / "logs"

    if args.wandb:
        wandb.init(project="hyperiap")
        run_id = wandb.run.id

    if args.run_ss:
        # -----------
        # ss model
        # -----------

        # stage specifc epochs
        if args.max_epochs_ss > 0:
            max_epoch = args.max_epochs_ss
        else:
            max_epoch = arg_groups["Trainer Args"].max_epochs

        checkpoint = fit(
            args=args,
            arg_groups=arg_groups,
            data=data,
            run_id=run_id,
            max_epoch=max_epoch,
            model=ssmodel,
            log_dir=log_dir,
            stage="ss",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
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

        checkpoint = fit(
            args=args,
            arg_groups=arg_groups,
            data=data,
            run_id=run_id,
            max_epoch=max_epoch,
            model=model,
            log_dir=log_dir,
            stage="noisy",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
            checkpoint=checkpoint,
            lsmooth=args.ls_modifier,
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

        checkpoint = fit(
            args=args,
            arg_groups=arg_groups,
            data=data,
            run_id=run_id,
            max_epoch=max_epoch,
            model=model,
            log_dir=log_dir,
            stage="clean",
            lit_sup_model=seq_model_class,
            lit_ss_model=ss_model_class,
            checkpoint=checkpoint,
        )

        print("clean model checkpoint: ", checkpoint)

        # test dataset
        if args.test:
            # TODO create test function in utils
            # turn profiling off during testing
            # trainer.profiler = (pl.pytorch.profilers.PassThroughProfiler())
            # trainer.test(seq_model, datamodule=data)
            pass
        else:
            test_loss = random.randint(1, 100000)

    if args.wandb:
        wandb.log({"test_loss": test_loss})
        # end wandb experiment
        wandb.finish()


if __name__ == "__main__":
    main()
