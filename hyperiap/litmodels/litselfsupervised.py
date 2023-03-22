import torch
from einops import rearrange
import matplotlib.pyplot as plt
import wandb
import numpy as np

from argparse import Namespace

from hyperiap.litmodels.litbasemodel import LitBaseModel

LR_SS = 1e-3
LOSS_SS = "mse_loss"
T_0_SS = 2


class LitSelfSupervised(LitBaseModel):
    def __init__(self, model, args: Namespace = None):
        super().__init__(args)
        self.model = model
        self.data_config = self.model.encoder.data_config

        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr_ss", LR_SS)
        self.wandb = self.args.get("wandb", False)

        loss = self.args.get("loss_ss", LOSS_SS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.T_0 = self.args.get("T_0_ss", T_0_SS)

    def forward(self, x):
        # use forward for inference/predictions
        (
            pred_pixel,
            masked_pixel,
            patches,
            batch_range,
            masked_indices,
            unmasked_indices,
        ) = self.model(x)
        return (
            pred_pixel,
            masked_pixel,
            patches,
            batch_range,
            masked_indices,
            unmasked_indices,
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        pred_pixel, masked_pixel, *other = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("train_loss", loss)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        # pred_pixel, masked_pixel, *other = self(x)

        (
            pred_pixel,
            masked_pixel,
            patches,
            batch_range,
            masked_indices,
            unmasked_indices,
        ) = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if self.wandb:
            wandb_logger = self.logger.experiment
            patch = 4
            fig = self._plot_hyperspec(
                pred_pixel,
                masked_pixel,
                patches,
                batch_range,
                masked_indices,
                unmasked_indices,
                self.data_config["wl"],
                patch,
            )
            wandb_logger.log({"plot": wandb.Image(fig)})

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = rearrange(x, "b1 b2 z c -> (b1 b2) z c")

        pred_pixel, masked_pixel, *other = self(x)
        loss = self.loss_fn(pred_pixel, masked_pixel)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        outputs = {"loss": loss}

        return outputs

    def _plot_hyperspec(
        self,
        pred_pixel,
        masked_pixel,
        patches,
        batch_range,
        masked_indices,
        unmasked_indices,
        wl,
        patch,
    ):

        # nbands
        # zdim = number of pixels per samples
        vdim = int(patches.shape[2] / patch)

        # reshape wl
        ###########

        # calculate padding
        padding = (patches.shape[1] * patch) - (wl.shape[0])

        # pad wl to match patches
        wl = np.pad(wl, (0, padding), mode="edge")
        # repeat wl to match num pixels per sample
        wl = np.tile(wl, (vdim, 1))
        # rearrange like pixels
        wl = rearrange(wl, "z (c p) -> c (p z)", p=patch)

        # get wl of masked pathes
        wl_mask = wl[
            masked_indices[0],
        ]
        # getwl of unmaksed patches
        wl_unmask = wl[
            unmasked_indices[0],
        ]

        mask_wl = rearrange(wl_mask, "c (p z) -> z (c p)", p=patch)
        mask_wl = mask_wl[0, :]
        unmask_wl = rearrange(wl_unmask, "c (p z) -> z (c p)", p=patch)
        unmask_wl = unmask_wl[0, :]
        # unwrap pred masked pixel values
        p_mask = (
            rearrange(pred_pixel, "b c (p z) -> b z (c p)", p=patch).detach().numpy()
        )
        p_mask = p_mask[:, 0, :]
        # unwrap actual masked pixels values
        mask = (
            rearrange(masked_pixel, "b c (p z) -> b z (c p)", p=patch).detach().numpy()
        )
        mask = mask[:, 0, :]
        # unwrap notmasked pixel values
        unmask = patches[batch_range, unmasked_indices]
        unmask = rearrange(unmask, "b c (p z) -> b z (c p)", p=patch).detach().numpy()
        unmask = unmask[:, 0, :]

        # plot
        fig, ax = plt.subplots()
        ax.scatter(unmask_wl, unmask[0, :], label="unmasked", color="blue")
        ax.scatter(mask_wl, mask[0, :], label="masked", color="red")
        ax.scatter(mask_wl, p_mask[0, :], label="predicted", color="green")

        return fig

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr_ss", type=float, default=LR_SS)
        parser.add_argument("--T_0_ss", type=float, default=T_0_SS)
        parser.add_argument(
            "--loss_ss",
            type=str,
            default=LOSS_SS,
            help="loss function from torch.nn.functional",
        )

        return parser
