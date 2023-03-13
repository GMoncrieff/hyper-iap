import torch
import pytorch_lightning as pl
from argparse import Namespace

from abc import abstractmethod

OPTIMIZER = "AdamW"


class LitBaseModel(pl.LightningModule):
    def __init__(self, model, args: Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = "AdamW"
        self.optimizer_class = getattr(torch.optim, optimizer)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = self.optimizer_class(parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.T_0
        )
        # return {
        #    "optimizer": optimizer,
        #    "scheduler": [scheduler],
        #    "monitor": "val/loss",
        # }
        return [optimizer], [scheduler]
