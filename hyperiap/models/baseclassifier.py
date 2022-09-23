
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy

from argparse import ArgumentParser, Namespace

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100

class BaseClassifier(pl.LightningModule):
    def __init__(self, model, args: Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("transformer",):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding
    
    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        outputs = {"loss": loss}
        
        return outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}
        
        return outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.test_acc(logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser
