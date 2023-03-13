import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from hyperiap.models.vit import Transformer

from argparse import Namespace

DECODER_DIM = 128
MASKING_RATIO = 0.75
DECODER_DEPTH = 1
DECODER_HEADS = 4
DECODER_DIM_HEAD = 32


class MAE(nn.Module):
    def __init__(self, encoder, args: Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        decoder_dim = self.args.get("decoder_dim", DECODER_DIM)
        masking_ratio = self.args.get("masking_ratio", MASKING_RATIO)
        decoder_depth = self.args.get("decoder_depth", DECODER_DEPTH)
        decoder_heads = self.args.get("decoder_heads", DECODER_HEADS)
        decoder_dim_head = self.args.get("decoder_dim_head", DECODER_DIM_HEAD)

        self.encoder = encoder
        assert (
            masking_ratio > 0.0 and masking_ratio < 1.0
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.pad = encoder.embedding.pad
        num_patches, encoder_dim = encoder.embedding.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.embedding.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_head=decoder_dim * 4,
            dropout=0,
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches
        img = self.pad(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.embedding.pos_embedding[:, :num_patches]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.embedding.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return (pred_pixel_values, masked_patches)
        # calculate reconstruction loss

        # recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        # return recon_loss

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--decoder_dim", type=int, default=DECODER_DIM)
        parser.add_argument("--masking_ratio", type=float, default=MASKING_RATIO)
        parser.add_argument("--decoder_depth", type=int, default=DECODER_DEPTH)
        parser.add_argument("--decoder_heads", type=int, default=DECODER_HEADS)
        parser.add_argument("--decoder_dim_head", type=int, default=DECODER_DIM_HEAD)
        return parser
