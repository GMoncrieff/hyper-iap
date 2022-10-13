from argparse import Namespace
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.utils.data

from einops import rearrange

INPUT_DIM = 27
NUM_CLASSES = 2
SEQ_LEN = 500
KERNEL_SIZE = 5
HIDDEN = 24
DROPOUT = 0.2


class TEMPCNN(torch.nn.Module):
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        # model params
        self.kernel_size = self.args.get("kernel_size", KERNEL_SIZE)
        self.hidden = self.args.get("hidden_dim", HIDDEN)
        self.dropout = self.args.get("dropout", DROPOUT)
        # data params
        self.num_classes = data_config["num_classes"]
        self.seq_len = data_config["num_bands"]
        self.input_dim = data_config["num_dim"]

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            self.input_dim,
            self.hidden,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout,
        )
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            self.hidden,
            self.hidden,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout,
        )
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            self.hidden,
            self.hidden,
            kernel_size=self.kernel_size,
            drop_probability=self.dropout,
        )
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            self.hidden * self.seq_len, 4 * self.hidden, drop_probability=self.dropout
        )
        self.out = nn.Linear(4 * self.hidden, self.num_classes)

    def forward(self, x):
        x = rearrange(x, "s t c b -> (s b) c t", s=1)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.out(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE)
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        return parser


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dims, kernel_size=KERNEL_SIZE, drop_probability=DROPOUT
    ):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=DROPOUT):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
