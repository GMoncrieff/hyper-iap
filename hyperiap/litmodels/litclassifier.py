import torch
import numpy as np
import wandb
from torchmetrics import Accuracy, F1Score
from einops import rearrange
from argparse import Namespace
import torch.nn.functional as F
from hyperiap.litmodels.litbasemodel import LitBaseModel

LR = 1e-3
T_0 = 2
LOSS = "cross_entropy"
MONITOR = ""
LABEL_SMOOTH = 0.0


class LitClassifier(LitBaseModel):
    def __init__(self, model, args: Namespace = None):
        super().__init__(args)
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config
        # get class names
        self.class_names = self.data_config.get("class_names")
        self.lr = self.args.get("lr", LR)
        self.monitor = self.args.get("monitor", MONITOR)

        loss = self.args.get("loss", LOSS)
        self.label_smooth = self.args.get("label_smooth", LABEL_SMOOTH)
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
        self.train_f1 = F1Score(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=self.data_config.get("num_classes")
        )

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")
        logits = self(x)
        # outputs = np.stack((y.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy()))
        pred = F.softmax(logits, dim=-1).detach().cpu().numpy()
        y = y.cpu().numpy()
        return pred, y

    # def predict(self, x, y):
    #    logits = self.model(x)
    #    return y, torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y, label_smoothing=self.label_smooth)
        self.train_acc(logits, y)
        self.train_f1(logits, y)

        self.log(f"{self.monitor}train_loss", loss)
        self.log(
            f"{self.monitor}train_acc", self.train_acc, on_step=False, on_epoch=True
        )
        self.log(f"{self.monitor}train_f1", self.train_f1, on_step=False, on_epoch=True)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y, label_smoothing=self.label_smooth)
        self.val_acc(logits, y)
        self.val_f1(logits, y)

        self.log(f"{self.monitor}val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            f"{self.monitor}val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{self.monitor}val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # if ((self.monitor == "clean_") and (self.args.get("wandb",False))):
        #    self.logger.experiment.log(
        #        {
        #            f"{self.monitor}val_confmat": wandb.plot.confusion_matrix(
        #                probs=logits.cpu().numpy(),
        #                y_true=y.cpu().numpy(),
        #                class_names=self.class_names,
        #            )
        #        }
        #    )

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = rearrange(y, "b1 b2 -> (b1 b2)")
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        logits = self(x)
        loss = self.loss_fn(logits, y, label_smoothing=self.label_smooth)
        self.test_acc(logits, y)
        self.test_f1(logits, y)

        self.log(f"{self.monitor}test_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{self.monitor}test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log(f"{self.monitor}test_f1", self.test_f1, on_step=False, on_epoch=True)

        # if ((self.monitor == "clean_") and (self.args.get("wandb",False))):
        #    self.logger.experiment.log(
        #        {
        #            f"{self.monitor}test_confmat": wandb.plot.confusion_matrix(
        #                probs=logits.cpu().numpy(),
        #                y_true=y.cpu().numpy(),
        #                class_names=self.class_names,
        #            )
        #        }
        # )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--T_0", type=float, default=T_0)
        parser.add_argument("--monitor", type=str, default=MONITOR)
        parser.add_argument("--label_smooth", type=float, default=LABEL_SMOOTH)
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser
