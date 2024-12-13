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
        self._estimated_length = 0
        self.single_length = 2000000
        # Precompute for all operations
        self.pad_token_id = self.vocab["<pad>"]
        self.pad_value = self.args.pad_value
        self.max_length = max_seq_len
        self.mask_ratio = self.args.mask_ratio if mask_ratio is None else mask_ratio
        self.raw_dataset = self._load_datasets()
    def _load_datasets(self):
        if self.data_path.is_dir():
            all_files = self.data_path.glob("*.parquet")
            regex = re.compile(r"combined_data_\d+\.parquet$")
            parquet_files = [str(f) for f in all_files if regex.search(str(f))]
            raw_dataset = load_dataset("parquet", data_files=parquet_files, streaming=True)
            self._estimated_length = self.single_length  * len(parquet_files)
        else:
            try:
                raw_dataset = load_dataset("parquet", data_files=str(self.data_path),cache_dir=self.args.cache_dir)
            except:
                raw_dataset = load_dataset("parquet", data_files=str(self.data_path))
        return raw_dataset["train"]
    def _preprocess(self, genes, expressions):
        genes = torch.tensor(genes, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float)
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
    def unbinning_batch(self, binned_row, original_row):
        n_bins = self.args.n_bins
        batch_size, dimension = original_row.shape
        approximated_row = torch.empty_like(original_row)
        for i in range(batch_size):
            sample = original_row[i]
            valid_mask = sample != -2
            valid_sample = sample[valid_mask]
            if valid_sample.nelement() == 0:
                approximated_row[i] = sample
                continue
            bins = torch.quantile(valid_sample, torch.linspace(0, 1, n_bins))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            valid_binned_row = binned_row[i][valid_mask]
            clamped_binned_row = torch.clamp(valid_binned_row.float(), 0, len(bin_centers) - 1).long()
            approximated_valid_row = bin_centers[clamped_binned_row]
            approximated_row[i][valid_mask] = approximated_valid_row
            approximated_row[i][~valid_mask] = sample[~valid_mask]
        return approximated_row
    def __len__(self):
        if self._estimated_length == 0:
            return len(self.raw_dataset)
        else:
            return self._estimated_length

    def __getitem__(self, idx):
        example = self.raw_dataset[idx]
        genes, processed_exps, original_exps, padding_length = self._preprocess(example['genes'], example['expressions'])
        masked_exps = self.mask(processed_exps, padding_length, self.mask_ratio)
        return {"gene": genes, "expr": processed_exps, "masked_expr": masked_exps, "expr_orin": original_exps}

    def mask(self, expressions, padding_length, mask_ratio):
        if padding_length > 0:
            expressions, expressions_keep = torch.split(expressions, [expressions.size(0)-padding_length, padding_length], dim=0)
        mask = torch.rand(expressions.size(), dtype=torch.float) < mask_ratio
        expressions = expressions.clone()  # Clone only once
        expressions[mask] = self.args.mask_value
        if padding_length > 0:
            expressions = torch.cat((expressions, expressions_keep), dim=0)
        return expressions

