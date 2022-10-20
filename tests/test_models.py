from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.litmodels.litclassifier import LitClassifier
import torch


def test_tempcnn():

    model = LitClassifier(
        TEMPCNN(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})
    )

    # create rand torch tensor with shape (1, 10, 4, 20)
    nrand = torch.rand(1, 10, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [20, 5]


def test_vit():

    model = LitClassifier(
        simpleVIT(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})
    )

    # create rand torch tensor with shape (1, 10, 4, 20)
    nrand = torch.rand(1, 10, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [20, 5]


def test_mae():

    v = simpleVIT(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})

    mae = MAE(encoder=v)

    nrand = torch.randn(1, 10, 4, 20)

    pixel, mask = mae(nrand)

    assert pixel.shape == mask.shape
