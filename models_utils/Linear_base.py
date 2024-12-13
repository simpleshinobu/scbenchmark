import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, feature_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(feature_size, feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)
    def forward(self, x_in):
        x = self.fc1(x_in)
        x = F.relu(x)
        x = self.fc2(x)
        return x
class LinearModel_Base(nn.Module):
    def __init__(self, in_features, out_features, num_layers, dropout_rate, args, ntoken):
        super(LinearModel_Base, self).__init__()
        self.hidden_size = in_features
        self.ffns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.ffns.append(FFN(out_features))
            self.layer_norms.append(nn.LayerNorm(self.hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
    def forward(self, x, src, src_key_padding_mask):
        for  ffns, ln, drop in zip(self.ffns, self.layer_norms, self.dropouts):
            identity = x
            x = ffns(x)
            x = ln(x)
            x = F.gelu(x)
            x = x + identity
        return x