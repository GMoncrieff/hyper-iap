import wandb
import types

from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.datasets.point_module import PointDataModule

from hyperiap.litmodels.litclassifier import LitClassifier


def get_vit_model(run_id="dnxchxbq"):
    api = wandb.Api()
    run = wandb.init("hyperiap")
    artifact = run.use_artifact(
        f"glennwithtwons/hyperiap/model-{run_id}:best_k", type="model"
    )
    artifact_dir = artifact.download()
    run.finish()
    # get model config
    config = api.run(f"glennwithtwons/hyperiap/{run_id}").config
    args = types.SimpleNamespace(**config)
    # setup model
    xmod = PointDataModule()
    model = LitClassifier.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
        model=simpleVIT(data_config=xmod.config(), args=args),
        args=args,
    )
    return model, xmod


def get_tempcnn_model(run_id="007kslnc"):
    api = wandb.Api()
    run = wandb.init("hyperiap")
    artifact = run.use_artifact(
        f"glennwithtwons/hyperiap/model-{run_id}:best_k", type="model"
    )
    artifact_dir = artifact.download()
    run.finish()
    # get model config
    config = api.run(f"glennwithtwons/hyperiap/{run_id}").config
    args = types.SimpleNamespace(**config)
    # setup model
    xmod = PointDataModule()
    model = LitClassifier.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
        model=TEMPCNN(data_config=xmod.config(), args=args),
        args=args,
    )
    return model, xmod
