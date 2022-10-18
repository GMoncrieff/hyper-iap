import torch
import pytorch_lightning as pl

from argparse import Namespace

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "mse_loss"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseSelfSupervised(pl.LightningModule):
    def __init__(self, model, args: Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("transformer",):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

    def forward(self, x):
        # use forward for inference/predictions
        pred_pixels, mask_pixels = self.model(x)
        return pred_pixels, mask_pixels

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("train/loss", loss)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--optimizer",
            type=str,
            default=OPTIMIZER,
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument(
            "--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS
        )
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser
