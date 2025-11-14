import os
import json
import pickle
import torch
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import os
import re
import random
import numpy as np
from collections import defaultdict
def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def select_exclude_1102(args, is_main_process=None, print_eli=True):
    all_counts = torch.load("analysis/imp_genes4cls.torch")
    selected_datasets = args.filter_names
    exclude_idx = []
    exclude_idx_origin = []
    dataset_counts = []
    for dataset in selected_datasets:
        if dataset in all_counts:
            templist = all_counts[dataset]
            tensor_list = torch.tensor(templist)
            _, sorted_indices = torch.sort(tensor_list, descending=True)
            filtered_indices = [idx for idx in sorted_indices.tolist()]
            top_indices = filtered_indices[:args.select_idxnum]
            exclude_idx.append(top_indices)
            exclude_idx_origin.append(filtered_indices[:args.select_idxnum])
    idx_positions = defaultdict(list)
    for list_idx, indices in enumerate(exclude_idx):
        for pos, idx in enumerate(indices):
            idx_positions[idx].append((list_idx, pos))
    outputs = []
    for idx, positions in idx_positions.items():
        if len(positions) > 1:
            all_positions = [pos for _, pos in positions]
            min_position = min(all_positions)
            output = [f"ID {idx}:"]

            if min_position > args.keep_rank2:
                # Rule 1: Remove from all lists if min position > args.keep_rank2
                for list_idx, pos in positions:
                    exclude_idx[list_idx][pos] = None
                    output.append(f"list{list_idx + 1}: [{pos}] (removed)")
            elif min_position < args.keep_rank1 and len([p for p in all_positions if p < args.keep_rank1]) >= 2:
                # Rule 2: Remove from all lists if min position < args.keep_rank1 and occurs two or more times
                for list_idx, pos in positions:
                    exclude_idx[list_idx][pos] = None
                    output.append(f"list{list_idx + 1}: [{pos}] (removed)")
            elif min_position < args.keep_rank1 and len([p for p in all_positions if p < args.keep_rank1]) == 1:
                # Rule 3: Keep in the list with min position < args.keep_rank1, remove from others
                for list_idx, pos in positions:
                    if pos == min_position:
                        output.append(f"list{list_idx + 1}: [{pos}] (keep)")
                    else:
                        exclude_idx[list_idx][pos] = None
                        output.append(f"list{list_idx + 1}: [{pos}] (removed)")
            elif args.keep_rank1 <= min_position <= args.keep_rank2:
                # Rule 4: Keep first occurrence between args.keep_rank1-args.keep_rank2, remove from others
                first_occurrence = True
                for list_idx, pos in positions:
                    if first_occurrence and args.keep_rank1 <= pos <= args.keep_rank2:
                        output.append(f"list{list_idx + 1}: [{pos}] (keep)")
                        first_occurrence = False
                    else:
                        exclude_idx[list_idx][pos] = None
                        output.append(f"list{list_idx + 1}: [{pos}] (removed)")
            outputs.append(" ".join(output))

    if print_eli:
        print("\t||".join(outputs))
    # Remove None values from exclude_idx lists
    exclude_idx = [[idx for idx in indices if idx is not None] for indices in exclude_idx]

    return sum(exclude_idx,[]), [len(templist) for templist in exclude_idx]


def create_vocab_access_stages(vocab, num_stages, incre_only=False, base_ratio=0.5, exclude_idx = [], keep_base_id=False):
    vocab_items = sorted(vocab.vocab.items(), key=lambda x: x[1])
    special_symbols = vocab_items[-3:]
    normal_items = [item for item in vocab_items[:-3] if item[1] not in exclude_idx] ##exclude some imp id
    if keep_base_id:
        base_vocab_indices = [item[1] for item in normal_items]
        base_vocab_indices.extend([item[1] for item in special_symbols])
        stage_tokens_indices = [base_vocab_indices[:] for _ in range(num_stages)]
        return stage_tokens_indices
    else:
        total_size = len(normal_items)
        if num_stages==1:
            full_vocab_indices = [item[1] for item in normal_items]
            full_vocab_indices.extend([item[1] for item in special_symbols])
            return [full_vocab_indices]
        base_size = int(total_size * base_ratio)
        remaining_size = total_size - base_size
        remaining_items = torch.Tensor([item[1] for item in normal_items]).long()
        indices = np.random.permutation(remaining_items.size(0))
        base_tokens_indices = remaining_items[indices[:base_size]].tolist()
        remaining_indices = remaining_items[indices[base_size:]].tolist()
        base_tokens_indices.extend([item[1] for item in special_symbols])
        stage_tokens_indices = [base_tokens_indices.copy()]
        increment_per_stage = remaining_size // (num_stages - 1)
        start_idx = 0
        for i in range(num_stages - 1):
            end_idx = start_idx + increment_per_stage + (1 if i < remaining_size % (num_stages - 1) else 0)
            additional_tokens_indices = remaining_indices[start_idx:end_idx]
            if incre_only:
                current_stage_tokens_indices = additional_tokens_indices + [item[1] for item in special_symbols]
            else:
                current_stage_tokens_indices = stage_tokens_indices[-1] + additional_tokens_indices
            stage_tokens_indices.append(current_stage_tokens_indices)
            start_idx = end_idx
    return stage_tokens_indices

def distribute_exclude_indices(stage_tokens_indices, exclude_idx, num_per_stage=None, incre_only=False):
    num_stages = len(stage_tokens_indices)
    num_exclude = len(exclude_idx)
    if num_per_stage is None:
        num_per_stage = [num_exclude // num_stages] * num_stages
        for i in range(num_exclude % num_stages):
            num_per_stage[i] += 1
    else:
        if len(num_per_stage) < num_stages:
            num_per_stage.extend([0] * (num_stages - len(num_per_stage)))
        elif len(num_per_stage) > num_stages:
            num_per_stage = num_per_stage[:num_stages]
        total_required = sum(num_per_stage)
        if total_required > num_exclude:
            raise ValueError("指定的 num_per_stage 超过了 exclude_idx 的可用数量。")
    end_idx, start_idx = 0, 0
    for i, count in enumerate(num_per_stage):
        if incre_only:
            end_idx = start_idx + count
            stage_tokens_indices[i].extend(exclude_idx[start_idx:end_idx])
            start_idx = end_idx
        else:
            end_idx = end_idx + count
            stage_tokens_indices[i].extend(exclude_idx[:end_idx])
    return stage_tokens_indices

def aggregate_counts(select_counts, num_stages):
    base_size = len(select_counts) // num_stages
    remainder = len(select_counts) % num_stages
    result = []
    start_index = 0
    for i in range(num_stages):
        end_index = start_index + base_size + (1 if i < remainder else 0)
        result.append(sum(select_counts[start_index:end_index]))
        start_index = end_index
    return result

def load_model_origin(model=None, args=None, model_path=None):
    model_dir = Path(model_path)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    args.embsize, args.nheads, args.d_hid = model_configs["embsize"],  model_configs["nheads"], model_configs["d_hid"]
    args.nlayers = model_configs["nlayers"]
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    loaded_dict = {}
    for k, v in pretrained_dict.items():
        if "Wqkv.weight" in k:
            k = k.replace('Wqkv.weight', 'in_proj_weight')
        if "Wqkv.bias" in k:
            k = k.replace('Wqkv.bias', 'in_proj_bias')
        if k in model_dict and v.shape == model_dict[k].shape:
            loaded_dict[k] = v
        else:
            print("missing ", k)
    original_keys = set(model_dict.keys())
    updated_keys = set(loaded_dict.keys())
    unloaded_keys = original_keys - updated_keys
    print("unload：", len(unloaded_keys), unloaded_keys)
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    return model

def load_model(model=None, args=None, model_path=None):
    if os.path.isdir(model_path):
        model_dir = Path(model_path)
        model_path = model_dir / "best_model.pt"
    if args.resume:
        print("Loaded model from {} for resuming".format(model_path))
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded parallel model from {}".format(model_path))
    return model



def find_latest_checkpoint(load_dir):
    """
    Finds the latest checkpoint in the specified directory and returns the file path and epoch.
    """
    max_epoch = -1
    latest_model_path = None
    checkpoint_pattern = re.compile(r"model-ep(\d+)\.pt")
    for filename in os.listdir(load_dir):
        match = checkpoint_pattern.search(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model_path = os.path.join(load_dir, filename)
    print("find the last epoch files", latest_model_path)
    return latest_model_path, max_epoch

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    mask = mask * (target != 0)
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
class GeneVocab(Vocab):
    """
    Vocabulary for genes.
    """
    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)
    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )
    @classmethod
    def from_dict(cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ):
        # initiate an empty vocabulary first
        _vocab = cls([])
        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab
    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])
    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab