import torch
from einops import rearrange
import matplotlib.pyplot as plt
import wandb
import numpy as np
import random

from argparse import Namespace

from hyperiap.litmodels.litbasemodel import LitBaseModel

LR_SS = 1e-3
LOSS_SS = "mse_loss"
T_0_SS = 2
# number of pixles in plot
P_LENGTH = 6
SS_MONITOR = "ss_"


class LitSelfSupervised(LitBaseModel):
    def __init__(self, model, args: Namespace = None):
        super().__init__(args)
        self.model = model
        self.data_config = self.model.encoder.data_config

        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get("lr_ss", LR_SS)
        self.wandb = self.args.get("wandb", False)
        self.ss_monitor = self.args.get("ss_monitor", SS_MONITOR)
        loss = self.args.get("loss_ss", LOSS_SS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.T_0 = self.args.get("T_0_ss", T_0_SS)
        # patch len from transformer
        self.patch_length = self.args.get("patch_len")

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

        self.log(f"{self.ss_monitor}train_loss", loss)

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

        self.log(f"{self.ss_monitor}val_loss", loss, prog_bar=True, sync_dist=True)

        if self.wandb:
            wandb_logger = self.logger.experiment
            patch = self.patch_length
            fig = self._plot_hyperspec(
                pred_pixel.cpu(),
                masked_pixel.cpu(),
                patches.cpu(),
                batch_range.cpu(),
                masked_indices.cpu(),
                unmasked_indices.cpu(),
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

        self.log(f"{self.ss_monitor}val_loss", loss, on_step=False, on_epoch=True)

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
        length=P_LENGTH,
    ):
        # get random samples
        # batch len
        # blen = range(patches.shape[0])
        # samp = random.sample(blen, length)

        # or hardcode
        samp = [0, 24, 36, 58, 72, 88]

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
        wl_mask = wl[masked_indices[samp]]
        # getwl of unmaksed patches
        wl_unmask = wl[unmasked_indices[samp]]

        mask_wl = rearrange(wl_mask, "b c (p z) -> b z (c p)", p=patch)
        mask_wl = mask_wl[:, 0, :] * 1000
        unmask_wl = rearrange(wl_unmask, "b c (p z) -> b z (c p)", p=patch)
        unmask_wl = unmask_wl[:, 0, :] * 1000

        # unwrap pred masked pixel values
        p_mask = (
            rearrange(pred_pixel, "b c (p z) -> b z (c p)", p=patch)
            .to(dtype=torch.float)
            .detach()
            .numpy()
        )
        p_mask = p_mask[samp, 0, :] / 10000

        # unwrap actual masked pixels values
        mask = (
            rearrange(masked_pixel, "b c (p z) -> b z (c p)", p=patch)
            .to(dtype=torch.float)
            .detach()
            .numpy()
        )
        mask = mask[samp, 0, :] / 10000

        # unwrap notmasked pixel values
        unmask = patches[batch_range, unmasked_indices]
        unmask = rearrange(unmask, "b c (p z) -> b z (c p)", p=patch).detach().numpy()
        unmask = unmask[samp, 0, :] / 10000

        fig, axs = plt.subplots(
            nrows=2, ncols=length // 2, sharex=True, sharey=True, figsize=(12, 3)
        )
        axs = axs.flatten()

        for i in range(length):
            unmask_wl_plot = unmask_wl[i, :]
            mask_wl_plot = mask_wl[i, :]
            unmask_plot = unmask[i, :]
            mask_plot = mask[i, :]
            p_mask_plot = p_mask[i, :]

            unma = np.concatenate(
                (unmask_wl_plot.reshape(-1, 1), unmask_plot.reshape(-1, 1)), axis=1
            )
            ma = np.concatenate(
                (mask_wl_plot.reshape(-1, 1), mask_plot.reshape(-1, 1)), axis=1
            )
            pma = np.concatenate(
                (mask_wl_plot.reshape(-1, 1), p_mask_plot.reshape(-1, 1)), axis=1
            )

            unma = unma[np.argsort(unma[:, 0])]
            unma_wl = unma[:, 0]
            unma_refl = unma[:, 1]

            ma = ma[np.argsort(ma[:, 0])]
            ma_wl = ma[:, 0]
            ma_refl = ma[:, 1]

            pma = pma[np.argsort(pma[:, 0])]
            pma_wl = pma[:, 0]
            pma_refl = pma[:, 1]

            for j in range(unma_wl.shape[0] - 1):
                if abs(unma_wl[j + 1] - unma_wl[j]) < 7:
                    axs[i].plot(
                        unma_wl[j : j + 2],
                        unma_refl[j : j + 2],
                        label="unmasked",
                        color="blue",
                        linestyle="-",
                    )

            for j in range(ma_wl.shape[0] - 1):
                if abs(ma_wl[j + 1] - ma_wl[j]) < 7:
                    axs[i].plot(
                        ma_wl[j : j + 2],
                        ma_refl[j : j + 2],
                        label="masked",
                        color="red",
                        linestyle="-",
                    )

            for j in range(pma_wl.shape[0] - 1):
                if abs(pma_wl[j + 1] - pma_wl[j]) < 7:
                    axs[i].plot(
                        pma_wl[j : j + 2],
                        pma_refl[j : j + 2],
                        label="masked",
                        color="purple",
                        linestyle="-",
                    )

        return fig

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr_ss", type=float, default=LR_SS)
        parser.add_argument("--T_0_ss", type=float, default=T_0_SS)
        parser.add_argument("--ss_monitor", type=str, default=SS_MONITOR)
        parser.add_argument(
            "--loss_ss",
            type=str,
            default=LOSS_SS,
            help="loss function from torch.nn.functional",
        )

        return parser
