import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear1DLayer(nn.Module):
    def __init__(self, in_channels, input_seq_length, output_seq_length):
        super(Linear1DLayer, self).__init__()
        self.linear_channel = nn.Linear(in_channels, in_channels)
        self.linear_seq = nn.Linear(input_seq_length, output_seq_length)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.linear_channel(x)
        x = x.transpose(1, 2)
        x = self.linear_seq(x)
        x = self.bn(x).transpose(1, 2)
        return x

class LinearModel(nn.Module):
    def __init__(self, initial_dim, num_layers, dropout_rate, args):
        super(LinearModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        seq_length = args.train_maxseq + 1

        for i in range(num_layers):
            output_seq_length = 1 if i == num_layers - 1 else seq_length // 2
            self.layers.append(Linear1DLayer(initial_dim, seq_length, output_seq_length))
            self.dropouts.append(nn.Dropout(dropout_rate))
            seq_length = output_seq_length

        self.final_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, src, src_key_padding_mask):
        for layer, drop in zip(self.layers, self.dropouts):
            identity = x
            x = layer(x)
            x = F.gelu(x)
            identity = F.adaptive_avg_pool1d(identity.transpose(1, 2), x.size(1)).transpose(1, 2)
            x = x + identity

        return x.squeeze(1)
