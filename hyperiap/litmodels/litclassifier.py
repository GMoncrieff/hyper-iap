import torch
from torchmetrics import Accuracy
from einops import rearrange
from argparse import Namespace

from hyperiap.litmodels.litbasemodel import LitBaseModel

LR = 1e-3
T_0 = 2
LOSS = "cross_entropy"


class LitClassifier(LitBaseModel):
    def __init__(self, model, args: Namespace = None):
        super().__init__(args)
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.T_0 = self.args.get("T_0", T_0)

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )
        self.test_acc = Accuracy(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.test_acc(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--T_0", type=float, default=T_0)
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser
