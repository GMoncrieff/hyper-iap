import argparse
from pathlib import Path
from lightning import seed_everything

from hyperiap.litmodels.litclassifier import LitClassifier
from hyperiap.litmodels.litselfsupervised import LitSelfSupervised

from utils.run_helpers import (
    setup_data_from_args,
    setup_model_from_args,
    setup_parser,
)
from utils.fit_model import fit

seed_everything(1234)


def test_fit():
    DATA_CLASS_MODULE = "hyperiap.datasets"
    MODEL_CLASS_MODULE = "hyperiap.models"

    parser = setup_parser(
        model_module=MODEL_CLASS_MODULE,
        ss_module=MODEL_CLASS_MODULE,
        data_module=DATA_CLASS_MODULE,
        point_module=DATA_CLASS_MODULE,
    )

    args = parser.parse_args([])
    args.testdata = 1
    # split args into groups
    arg_groups = {}

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    if args.ft_schedule is None:
        raise ValueError("Must provide a finetuning schedule")

    data, _ = setup_data_from_args(
        args, data_module=DATA_CLASS_MODULE, point_module=DATA_CLASS_MODULE
    )

    model = setup_model_from_args(args, data, model_module=MODEL_CLASS_MODULE)

    seq_model_class = LitClassifier
    ss_model_class = LitSelfSupervised
    log_dir = Path("training") / "logs"
    max_epoch = 5
    checkpoint, best_val_loss, _ = fit(
        args=args,
        arg_groups=arg_groups,
        data=data,
        max_epoch=max_epoch,
        model=model,
        log_dir=log_dir,
        stage="",
        lit_sup_model=seq_model_class,
        lit_ss_model=ss_model_class,
    )

    assert checkpoint is not None
    assert best_val_loss > 0.0
