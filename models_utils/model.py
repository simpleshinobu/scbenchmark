import torch
from torch import nn, Tensor
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .cnn_base import CNNModel as CNNBase
from .Linear import LinearModel as LinearModel_Ours
from .Linear_base import LinearModel_Base
# from .mamba_model import Mamba #please download mamba before use it!
class scModel(nn.Module):
    def __init__(self, vocab, args=None):
        super().__init__()
        d_model = args.embsize
        ntoken = len(vocab)
        self.cell_emb_style = args.cell_emb_style
        self.model_type = "Transformer"
        self.max_value = 512
        self.args = args
        self.dict_size = 256
        self.embedding = nn.Embedding(ntoken, self.dict_size, padding_idx=vocab[args.pad_token])
        self.layer_norm = nn.LayerNorm(d_model)
        if self.dict_size != self.args.embsize:
            self.emb_trans = nn.Linear(self.dict_size, self.args.embsize)
        self.value_encoder = nn.Sequential(nn.Linear(1, d_model),nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(args.dropout)
        if self.args.model_structure in ["transformer", "transformer_ours"]:
            encoder_layers = TransformerEncoderLayer(d_model, args.nheads, d_model, args.dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, args.nlayers)
        elif self.args.model_structure == "mamba":
            self.encoder = Mamba(d_model, args.nlayers, args=args)
        elif self.args.model_structure in ["cnn_base", "cnn_ours"]:
            self.encoder = CNNBase(d_model, args.nlayers, args.dropout, args)
        elif self.args.model_structure in ["linear_ours"]:
            self.encoder = LinearModel_Ours(d_model, args.nlayers, args.dropout, args)
        elif self.args.model_structure in ["linear_base"]:
            self.encoder = LinearModel_Base(d_model, d_model, args.nlayers, args.dropout, args, ntoken)
        if self.args.model_structure in ["cnn_ours"]:
            self.idx_list = torch.arange(60697)
            mask = self.idx_list >= self.args.vocab["<pad>"]
            large_values = self.idx_list[mask]
            value_60695 = large_values[large_values == self.args.vocab["<cls>"]]
            value_60694 = large_values[large_values == self.args.vocab["<pad>"]]
            remaining_values = self.idx_list[~mask]
            self.idx_list = torch.cat((value_60695, remaining_values, value_60694))
            max_val = max(self.idx_list) + 1
            self.global_index_map = torch.full((max_val,), -1, dtype=torch.long).cuda()
            for i, val in enumerate(self.idx_list):
                self.global_index_map[val] = i
        out_feat_size = d_model
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model),nn.LeakyReLU(0.01),
            nn.Linear(d_model, d_model),nn.LeakyReLU(0.01),
            nn.Linear(d_model, 1))
        self.gene2query = nn.Sequential(nn.Linear(d_model, d_model),nn.LeakyReLU(0.01),
            nn.Linear(d_model, d_model),nn.LeakyReLU(0.01),
            nn.Linear(d_model, out_feat_size))
        self._init_weights()
    def generate_position_encoding(self, max_len, d_model):
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(max_len) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in [self.value_encoder, self.decoder, self.gene2query]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.layer_norm.weight, 1)
        nn.init.constant_(self.layer_norm.bias, 0)
    def _transfer_idx(self,src):
        positions = self.global_index_map[src]
        trans = positions.argsort(dim=1)
        reverse_trans = torch.argsort(trans, dim=1)
        return trans, reverse_trans
    def forward(self,src: Tensor,values: Tensor,src_key_padding_mask: Tensor):
        output = {}
        gene_emb = self.embedding(src)
        if self.dict_size != self.args.embsize:
            gene_emb = self.emb_trans(gene_emb)
        values_emb = self.value_encoder(torch.clamp(values.unsqueeze(-1), max=self.max_value))
        total_embs = gene_emb + values_emb
        if self.args.model_structure in ["cnn_base","linear_ours","linear_base"]:
            encoding_output = self.encoder(total_embs, src, src_key_padding_mask)
        elif self.args.model_structure in ["cnn_ours"]:
            trans, _ = self._transfer_idx(src)
            trans_expanded = trans.unsqueeze(-1).expand(-1, -1, total_embs.size(-1))
            total_embs = torch.gather(total_embs, 1, trans_expanded)
            encoding_output = self.encoder(total_embs, src, src_key_padding_mask)
        elif self.args.model_structure == "mamba":
            encoding_output = self.encoder(total_embs)
        else:
            encoding_output = self.encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        if self.args.model_structure in ["linear_ours","cnn_base", "cnn_ours"]:
            cell_emb = encoding_output
            output["cell_emb"] = cell_emb
            mvc_output = torch.bmm(self.gene2query(gene_emb.detach()), cell_emb.unsqueeze(2)).squeeze(2)
            output["mvc_output"] = mvc_output
            output["mlm_output"] = mvc_output.detach()
        else:
            mlm_output = self.decoder(encoding_output).squeeze(2)
            output["mlm_output"] = mlm_output
            if self.cell_emb_style == "cls":
                cell_emb = encoding_output[:, 0, :]
            elif self.cell_emb_style == "avg-pool":
                cell_emb = torch.mean(encoding_output, dim=1)
            output["cell_emb"] = cell_emb
            mvc_output = torch.bmm(self.gene2query(gene_emb.detach()), cell_emb.unsqueeze(2)).squeeze(2)
            output["mvc_output"] = mvc_output
        return output


