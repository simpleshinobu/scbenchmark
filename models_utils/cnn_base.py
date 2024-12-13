import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=9, padding=4, stride=2):
        super(Conv1DLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, initial_dim, num_layers, dropout_rate, args):
        super(CNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()
        hidden_dim_channel = initial_dim

        for i in range(num_layers):
            if (i % 6) == 0:
                stride = 1
                pool_kernel_size = 1
            else:
                stride = 2
                pool_kernel_size = 2

            self.layers.append(Conv1DLayer(hidden_dim_channel, stride=stride))
            self.dropouts.append(nn.Dropout(dropout_rate))
            self.pools.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size))

        self.final_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, src, src_key_padding_mask):
        x = x.transpose(1, 2)
        for conv, drop, pool in zip(self.layers, self.dropouts, self.pools):
            identity = pool(x)
            x = conv(x)
            x = F.gelu(x)
            x = drop(x)
            x = x + identity
        x = self.final_pool(x)
        x = x.transpose(1, 2)
        return x.squeeze(1)
