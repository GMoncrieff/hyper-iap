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
    # learning rate modifier
    parser.add_argument(
        "--lr_modifier",
        type=float,
        default=1.0,
        help="modifier for learning rate",
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
    model, ssmodel = setup_models_from_args(
        args, data, ss_module=MODEL_CLASS_MODULE, model_module=MODEL_CLASS_MODULE
    )

    # -----------
    # setup models
    # -----------
    seq_model_class = LitClassifier
    ss_model_class = LitSelfSupervised
    log_dir = Path("training") / "logs"

    if args.wandb:
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
    trainer = pl.Trainer(
        **vars(arg_groups["Trainer Args"]), callbacks=callbacks, logger=logger
    )
    trainer.profiler = profiler
    trainer.fit(seq_ss_model, datamodule=data)

    # save best model in useable format
    seq_ss_model = ss_model_class.load_from_checkpoint(
        checkpoint_callback.best_model_path, args=args, model=ssmodel
    )
    litclass = seq_model_class(seq_ss_model.model.encoder)
    trainer = pl.Trainer(limit_val_batches=0, enable_checkpointing=False, logger=False)
    trainer.validate(litclass, datamodule=data)
    if args.wandb:
        ss_checkpoint = logger.experiment.dir + "/ss_classifier.ckpt"
    else:
        ss_checkpoint = logger.log_dir + "/ss_classifier.ckpt"

    trainer.save_checkpoint(ss_checkpoint)

    # end wandb experiment
    # wandb.finish()

    # -----------
    # noisy training
    # -----------
    if args.wandb:
        wandb.init(id=run_id, resume="must")

    seq_model = seq_model_class.load_from_checkpoint(
        ss_checkpoint, args=args, model=model
    )

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args, log_dir=log_dir, model=seq_model, finetune=False, append="_noisy"
    )
    if args.wandb:
        run_id = logger.version

    callbacks.append(checkpoint_callback)

    # train on noisy lables
    trainer = pl.Trainer(
        **vars(arg_groups["Trainer Args"]), callbacks=callbacks, logger=logger
    )
    trainer.profiler = profiler
    trainer.fit(seq_model, datamodule=data)

    noisy_checkpoint = checkpoint_callback.best_model_path
    # end wandb experiment
    # wandb.finish()

    # -----------
    # clean training
    # -----------

    args.monitor = "val_loss_final"

    # modify learning rate
    args.lr = args.lr * args.lr_modifier

    if args.wandb:
        wandb.init(id=run_id, resume="must")

    seq_model = seq_model_class.load_from_checkpoint(
        noisy_checkpoint, args=args, model=model
    )

    transfer = setup_transfer_from_args(
        args, seq_model.model, point, model_module=MODEL_CLASS_MODULE
    )
    seq_model = seq_model_class(args=args, model=transfer)

    # setup callbacks
    callbacks, checkpoint_callback, profiler, logger = setup_callbacks(
        args=args,
        log_dir=log_dir,
        model=seq_model,
        finetune=False,
        append="_clean",
        log_metric=args.monitor,
    )
    if args.wandb:
        run_id = logger.version

    callbacks.append(checkpoint_callback)

    # train on clean lables
    trainer = pl.Trainer(
        **vars(arg_groups["Trainer Args"]), callbacks=callbacks, logger=logger
    )
    trainer.profiler = profiler
    # trainer.fit(seq_model, datamodule=data)
    trainer.fit(seq_model, datamodule=point)

    trainer.profiler = (
        pl.pytorch.profilers.PassThroughProfiler()
    )  # turn profiling off during testing

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    if args.wandb:
        print("Best model also uploaded to W&B")

    # do we automatically test the best model?
    # trainer.test(seq_model, datamodule=data)

    # create a random number
    if args.wandb:
        random_number = random.randint(1, 100000)
        wandb.log({"test_loss": random_number})
        # end wandb experiment
        wandb.finish()


if __name__ == "__main__":
    main()
