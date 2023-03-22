from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.litmodels.litclassifier import LitClassifier
import torch


def test_tempcnn():

    model = LitClassifier(
        TEMPCNN(data_config={"num_classes": 5, "num_bands": 20, "num_dim": 4})
    )

    # create rand torch tensor with shape (6, 4, 20)
    nrand = torch.rand(6, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [6, 5]


def test_vit():

    model = LitClassifier(
        simpleVIT(data_config={"num_classes": 5, "num_bands": 20, "num_dim": 4})
    )

    # create rand torch tensor with shape (6, 4, 20)
    nrand = torch.rand(6, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [6, 5]


def test_mae():

    v = simpleVIT(data_config={"num_classes": 5, "num_bands": 20, "num_dim": 4})

    mae = MAE(encoder=v)

    # create rand torch tensor with shape (6, 4, 20)
    nrand = torch.rand(6, 4, 20)

    pixel, mask, *other = mae(nrand)

    assert pixel.shape == mask.shape
