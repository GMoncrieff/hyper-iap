import torch
import torch.nn as nn
import torch.utils.data


class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=27, num_classes=2, sequencelength=500, kernel_size=5, hidden_dims=24, dropout=0.2):
        super(TempCNN, self).__init__()

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.out = nn.Linear(4 * hidden_dims, num_classes)

    def forward(self, x):
        # require NxTxD
        #x = x.transpose(0,2, 1)
        x = x.permute(0, 2, 1)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.out(x)


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class simpleCNN(torch.nn.Module):
    def __init__(self, input_dim=1, num_classes=2, sequencelength=500, kernel_size=5, hidden_dim=24, dropout=0.2):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size // 2))
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size // 2))
        self.flatten = Flatten()
        self.l1 = torch.nn.Linear(sequencelength * hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # require NxTxD
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.l1(x))
        return self.out(x)