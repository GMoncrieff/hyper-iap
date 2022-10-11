from hyperiap.models.tempcnn import TEMPCNN
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
