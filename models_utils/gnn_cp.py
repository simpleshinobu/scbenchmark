import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, ntoken, k=64): #default
        super(GraphConvolution, self).__init__()
        self.k = k
        self.factor_linear = nn.Parameter(torch.rand(ntoken, self.k), requires_grad=True)
        self.factor = nn.Parameter(torch.rand(ntoken, self.k), requires_grad=True)
        self.linear = nn.Linear(in_features, out_features)
        self.linear_trans = nn.Sequential(*[nn.Linear(k, k), nn.GELU(), nn.Linear(k, k)])
        self.temperature = 1.0
    def forward(self, x, src, src_key_padding_mask):
        selected_factor = self.factor[src, :]
        selected_factor_trans = self.linear_trans(self.factor_linear[src, :])
        adj_selected = torch.matmul(selected_factor_trans, selected_factor_trans.transpose(1, 2)) +\
            torch.matmul(selected_factor, selected_factor.transpose(1, 2))
        adj_selected.masked_fill_(src_key_padding_mask.unsqueeze(1).expand(-1, adj_selected.size(1), -1), float('-inf'))
        adj_normalized = F.softmax(adj_selected/self.temperature, dim=2)
        x = torch.bmm(adj_normalized, x)
        x = self.linear(x)  # Linear transformation
        return x
class GNN_cp(nn.Module):
    def __init__(self, in_features, out_features, num_layers, dropout_rate, args, ntoken, k=100):
        super(GNN_cp, self).__init__()
        self.hidden_size = in_features
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolution(self.hidden_size, self.hidden_size, ntoken))
            self.layer_norms.append(nn.LayerNorm(self.hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
    def forward(self, x, src, src_key_padding_mask):
        for gnnconv, ln, drop in zip(self.layers, self.layer_norms, self.dropouts):
            identity = x
            x = gnnconv(x, src, src_key_padding_mask)
            x = ln(x)
            x = F.gelu(x)
            x = x + identity
        return x
