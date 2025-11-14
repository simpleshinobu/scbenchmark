import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import random
from datasets import load_dataset
from pathlib import Path
import re
import random

class ExpressionDataset(Dataset):
    def __init__(self, data_path, vocab, args, max_seq_len, mask_ratio=None):
        self.data_path = Path(data_path)
        self.vocab = vocab
        self.args = args
        self.cls_token = vocab["<cls>"]
        self.cls_value = args.cls_value
        self.preprocess_mode = args.preprocess_mode
        # Precompute for all operations
        self.pad_token_id = self.vocab["<pad>"]
        self.pad_value = self.args.pad_value
        self.max_length = max_seq_len
        self.mask_ratio = self.args.mask_ratio if mask_ratio is None else mask_ratio
        self.raw_dataset = self._load_datasets()
        self.non_log1p = (self.args.train_style == "others")
        self.tmp_vocab_idx = None
    def _load_datasets(self):
        try:
            raw_dataset = load_dataset("parquet", data_files=str(self.data_path),cache_dir=self.args.cache_dir)
        except:
            raw_dataset = load_dataset("parquet", data_files=str(self.data_path))
        return raw_dataset["train"]
    def _preprocess(self, genes, expressions):
        genes = torch.tensor(genes, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float)
        if self.tmp_vocab_idx is not None:
            insert_positions = torch.searchsorted(self.tmp_vocab_idx, genes)
            matches = insert_positions < len(self.tmp_vocab_idx)
            matches &= self.tmp_vocab_idx[insert_positions] == genes
            mask = matches.bool()
            genes = genes[mask]
            expressions = expressions[mask]
        if self.non_log1p:
            expressions = torch.log1p(expressions)
        if len(genes) < self.max_length:
            padding_length = self.max_length - len(genes)
            genes = torch.cat([genes, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)])
            original_exps = torch.cat([expressions, torch.full((padding_length,), self.pad_value, dtype=torch.float)])
        else:
            if self.args.use_weighted_sampling:
                if random.random() < 0.5:
                    weights = expressions - expressions.min() + 1e-5
                    indices = torch.multinomial(weights, self.max_length, replacement=False)
                else:
                    indices = torch.randperm(len(genes))[:self.max_length]
            else:
                indices = torch.randperm(len(genes))[:self.max_length]
            genes = genes[indices]
            original_exps = expressions[indices]
            padding_length = 0
        if self.preprocess_mode == "bin":
            expressions = self._binning(original_exps)
        elif self.preprocess_mode == "clamp":
            expressions = torch.clamp(original_exps, max=self.args.max_clamp)
        else:
            expressions = original_exps
        return genes, expressions, original_exps, padding_length

    def _binning(self, row: torch.Tensor):
        """Binning the row into n_bins using a quantile-based digitization."""
        if torch.all(row == 0):
            return torch.zeros_like(row)
        n_bins = self.args.n_bins
        bins = torch.quantile(row, torch.linspace(0, 1, n_bins - 1))
        left_digits = torch.bucketize(row, bins, right=False) - 1
        right_digits = torch.bucketize(row, bins, right=True) - 1
        rands = torch.rand_like(row)
        digits = rands * (right_digits - left_digits) + left_digits
        digits = torch.ceil(digits).to(torch.int64)
        return digits

    def unbinning(self, binned_row, original_row):
        binned_row = binned_row
        n_bins = self.args.n_bins
        bins = torch.quantile(original_row, torch.linspace(0, 1, n_bins))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        approximated_row = bin_centers[binned_row.long()]
        return approximated_row
    def unbinning_batch(self, binned_row, original_row):
        n_bins = self.args.n_bins
        batch_size, dimension = original_row.shape
        approximated_row = torch.empty_like(original_row)
        for i in range(batch_size):
            sample = original_row[i]
            valid_mask = sample != -2  # Mask to exclude -2 values
            valid_sample = sample[valid_mask]
            if valid_sample.nelement() == 0:
                approximated_row[i] = sample  # Potentially fill with default or unchanged values
                continue
            bins = torch.quantile(valid_sample, torch.linspace(0, 1, n_bins))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            valid_binned_row = binned_row[i][valid_mask]
            clamped_binned_row = torch.clamp(valid_binned_row.float(), 0, len(bin_centers) - 1).long()
            approximated_valid_row = bin_centers[clamped_binned_row]
            approximated_row[i][valid_mask] = approximated_valid_row
            approximated_row[i][~valid_mask] = sample[~valid_mask]  # Keep -2 values unchanged
        return approximated_row
    def __len__(self):
        return len(self.raw_dataset)
    def __getitem__(self, idx):
        example = self.raw_dataset[idx]
        genes, processed_exps, original_exps, padding_length = self._preprocess(example['genes'], example['expressions'])
        masked_exps = self.mask(processed_exps, padding_length, self.mask_ratio)
        return {"gene": genes, "expr": processed_exps, "masked_expr": masked_exps, "expr_orin": original_exps}

    def mask(self, expressions, padding_length, mask_ratio):
        #to preven mask padding positions
        if padding_length > 0:
            expressions, expressions_keep = torch.split(expressions, [expressions.size(0)-padding_length, padding_length], dim=0)
        mask = torch.rand(expressions.size(), dtype=torch.float) < mask_ratio
        expressions = expressions.clone()  # Clone only once
        expressions[mask] = self.args.mask_value
        if padding_length > 0:
            expressions = torch.cat((expressions, expressions_keep), dim=0)
        return expressions

from torch.utils.data import Subset
import numpy as np

def split_and_create_datasets(dataset, num_stages, base_ratio=0.5):
    if num_stages==1:
        total_size = len(dataset.raw_dataset)
        indices = list(range(total_size))
        full_dataset = Subset(dataset, indices)
        return [full_dataset]
    else:
        total_size = len(dataset.raw_dataset)
        base_size = int(total_size * base_ratio)
        remaining_size = total_size - base_size
        indices = np.random.permutation(total_size)
        base_indices = indices[:base_size].tolist()
        remaining_indices = indices[base_size:].tolist()
        base_dataset = Subset(dataset, base_indices)
        stage_sizes = [remaining_size // (num_stages - 1) for _ in range(num_stages - 1)]
        for i in range(remaining_size % (num_stages - 1)):
            stage_sizes[i] += 1
        datasets = [base_dataset]
        start_idx = 0
        for size in stage_sizes:
            end_idx = start_idx + size
            stage_indices = remaining_indices[start_idx:end_idx]
            stage_dataset = Subset(dataset, stage_indices)
            datasets.append(stage_dataset)
            start_idx = end_idx
        return datasets


def split_and_create_datasets_withsave_id(dataset, num_stages, base_ratio=0.5, id_load=None):
    total_size = len(dataset.raw_dataset)
    if id_load is not None:
        datasets = [Subset(dataset, indices) for indices in id_load]
        return datasets, id_load
    else:
        indices = np.random.permutation(total_size)
        if num_stages == 1:
            return [Subset(dataset, indices.tolist())], [indices.tolist()]
        else:
            base_size = int(total_size * base_ratio)
            remaining_size = total_size - base_size
            base_indices = indices[:base_size].tolist()
            remaining_indices = indices[base_size:].tolist()

            base_dataset = Subset(dataset, base_indices)
            datasets = [base_dataset]
            stage_sample_ids = [base_indices]

            stage_sizes = [remaining_size // (num_stages - 1) for _ in range(num_stages - 1)]
            for i in range(remaining_size % (num_stages - 1)):
                stage_sizes[i] += 1

            start_idx = 0
            for size in stage_sizes:
                end_idx = start_idx + size
                stage_indices = remaining_indices[start_idx:end_idx]
                stage_dataset = Subset(dataset, stage_indices)
                datasets.append(stage_dataset)
                stage_sample_ids.append(stage_indices)
                start_idx = end_idx

            return datasets, stage_sample_ids