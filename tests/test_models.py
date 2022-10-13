from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.models.baseclassifier import BaseClassifier
import torch


def test_tempcnn():

    model = BaseClassifier(
        TEMPCNN(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})
    )

    # create rand torch tensor with shape (1, 10, 4, 20)
    nrand = torch.rand(1, 10, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [20, 5]


def test_vit():

    model = BaseClassifier(
        simpleVIT(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})
    )

    # create rand torch tensor with shape (1, 10, 4, 20)
    nrand = torch.rand(1, 10, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == [20, 5]


def test_mae():

    v = simpleVIT(data_config={"num_classes": 5, "num_bands": 10, "num_dim": 4})

    mae = MAE(
        encoder=v,
        masking_ratio=0.75,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=6,  # anywhere from 1 to 8
    )

    nrand = torch.randn(1, 10, 4, 20)

    loss = mae(nrand)

    assert len(loss.shape) == 0
    assert isinstance(loss, torch.Tensor)
