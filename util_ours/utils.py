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