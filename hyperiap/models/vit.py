from argparse import Namespace
from typing import Any, Dict

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# model params
DIM = 64
DEPTH = 5
HEADS = 4
DIM_HEAD = 16
MLP_DIM = 8
DROPOUT = 0.1
EMB_DROPOUT = 0.1
PATCH_LEN = 4
MODE = "ViT"


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_head, dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class simpleVITextractor(nn.Module):
    def __init__(
        self,
        seq_size,
        near_band,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        emb_dropout,
        patch_len,
    ) -> None:
        super().__init__()

        # pad seq len if not divisible to patch len
        padding = patch_len - (seq_size % patch_len)
        self.pad = nn.ReplicationPad1d((0, padding))

        num_patches = (seq_size + padding) // patch_len
        patch_dim = near_band * patch_len

        # self.flat_batch = Rearrange("b1 b2 z c -> (b1 b2) z c")

        # self.to_embed_patch = nn.Sequential(
        #    Rearrange('s t c b -> (s b) c t', s=1),
        #    nn.Conv1d(near_band,dim, kernel_size=patch_len,stride=patch_len),
        #    Rearrange('b d t -> b t d')
        # )
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b z (c p) -> b c (p z)", p=patch_len),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

    def forward(self, x):

        # combine batch dims
        # x = self.flat_batch(x)
        # pad if needed
        x = self.pad(x)

        # embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        # x = self.to_embed_patch(x)
        x = self.to_patch_embedding(x)  # [b,n,dim]

        # add position embedding
        pe = self.pos_embedding
        # pe = posemb_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.to_latent(x)


class TransferLearningVIT(nn.Module):
    def __init__(self, backbone, data_config):
        super().__init__()
        self.data_config = data_config
        # init a pretrained model
        dim = backbone.dim
        self.embedding = backbone.embedding

        num_classes = data_config["num_classes"]
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.linear_head(embeddings)


class simpleVIT(nn.Module):
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        # data params
        self.num_classes = data_config["num_classes"]
        self.dim = self.args.get("dim", DIM)

        seq_size = data_config["num_bands"]
        near_band = data_config["num_dim"]
        # model params

        depth = self.args.get("depth", DEPTH)
        heads = self.args.get("heads", HEADS)
        dim_head = self.args.get("dim_heads", DIM_HEAD)
        mlp_dim = self.args.get("mlp_dim", MLP_DIM)
        dropout = self.args.get("dropout", DROPOUT)
        emb_dropout = self.args.get("emb_dropout", EMB_DROPOUT)
        # mode = self.args.get("mode", MODE)
        patch_len = self.args.get("patch_len", PATCH_LEN)

        self.embedding = simpleVITextractor(
            seq_size,
            near_band,
            self.dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            emb_dropout,
            patch_len,
        )
        self.linear_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_classes)
        )

    def forward(self, x):

        x = self.embedding(x)
        return self.linear_head(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--dim", type=int, default=DIM)
        parser.add_argument("--depth", type=int, default=DEPTH)
        parser.add_argument("--heads", type=int, default=HEADS)
        parser.add_argument("--dim_head", type=int, default=DIM_HEAD)
        parser.add_argument("--mlp_dim", type=int, default=MLP_DIM)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument("--emb_dropout", type=float, default=EMB_DROPOUT)
        parser.add_argument("--mode", type=str, default=MODE)
        parser.add_argument("--patch_len", type=int, default=PATCH_LEN)
        return parser
