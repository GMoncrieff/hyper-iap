import torch

from argparse import Namespace

from hyperiap.litmodels.litbasemodel import LitBaseModel

LR_SS = 1e-3
LOSS_SS = "mse_loss"
T_0_SS = 2


class LitSelfSupervised(LitBaseModel):
    def __init__(self, model, args: Namespace = None):
        super().__init__(args)
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr_ss", LR_SS)

        loss = self.args.get("loss_ss", LOSS_SS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.T_0 = self.args.get("T_0_ss", T_0_SS)

    def forward(self, x):
        # use forward for inference/predictions
        pred_pixels, mask_pixels = self.model(x)
        return pred_pixels, mask_pixels

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("train_loss", loss)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, _ = batch
        pred_pixel, masked_pixel = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

    @staticmethod
    def add_to_argparse(parser):
        LitBaseModel.add_to_argparse(parser)
        parser.add_argument("--lr_ss", type=float, default=LR_SS)
        parser.add_argument("--T_0_ss", type=float, default=T_0_SS)
        parser.add_argument(
            "--loss_ss",
            type=str,
            default=LOSS_SS,
            help="loss function from torch.nn.functional",
        )

        return parser
