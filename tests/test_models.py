from hyperiap.models.tempcnn import TEMPCNN
from hyperiap.models.vit import simpleVIT
from hyperiap.models.mae import MAE
from hyperiap.litmodels.litclassifier import LitClassifier
import torch
import pytest


def model_func_lit(model_constructor):
    return lambda: LitClassifier(model_constructor(data_config={"num_classes": 5, "num_bands": 20, "num_dim": 4}))

def model_func_mae(encoder_constructor):
    return lambda: MAE(encoder=encoder_constructor(data_config={"num_classes": 5, "num_bands": 20, "num_dim": 4}))

@pytest.mark.parametrize("model_func, expected_shape", [
    (model_func_lit(TEMPCNN), (6, 5)),
    (model_func_lit(simpleVIT), (6, 5)),
])
def test_model_output_shape(model_func, expected_shape):
    model = model_func()

    # create rand torch tensor with shape (6, 4, 20)
    nrand = torch.rand(6, 4, 20)

    pred = model(nrand)

    assert list(pred.shape) == list(expected_shape)


@pytest.mark.parametrize("encoder_constructor", [
    simpleVIT,
])
def test_mae_output_shape(encoder_constructor):
    model = model_func_mae(encoder_constructor)()

    # create rand torch tensor with shape (6, 4, 20)
    nrand = torch.rand(6, 4, 20)

    pixel, mask, *other = model(nrand)

    assert pixel.shape == mask.shape