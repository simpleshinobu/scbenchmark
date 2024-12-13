import argparse
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from collections import OrderedDict
from models_utils.model_perturb import scModel
from util_ours.utils import GeneVocab
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import warnings
import datetime
import random
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
class ExpressionDataset(Dataset):
    def __init__(self, data, max_length, pad_token_id, pad_value, args=None):
        if 'cls_name' in data:
            del data['cls_name']
        self.data = data
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.args = args
        self.preprocess_mode = args.preprocess_mode
        self.class_label_map = {label: idx for idx, label in enumerate(set(self.data['single_ctrl']))}
        self.zero_testing = args.zero_testing
        self.filter_and_split_data()

    def filter_and_split_data(self):
        self.base_indices = [i for i, label in enumerate(self.data['single_ctrl']) if label == -1]
        valid_indices = [i for i, label in enumerate(self.data['single_ctrl']) if label != -1]
        max_index = max(valid_indices) if valid_indices else -1
        for key in self.data.keys():
            if max_index >= len(self.data[key]):
                raise IndexError(f"Index {max_index} is out of bounds for key '{key}' with size {len(self.data[key])}")
        if self.zero_testing:
            unique_classes = list(set(self.data['single_ctrl'][i] for i in valid_indices))
            np.random.shuffle(unique_classes)
            split_idx = int(len(unique_classes) * 0.7)
            training_classes = set(unique_classes[:split_idx])
            validation_classes = set(unique_classes[split_idx:])
            self.train_indices = [i for i in valid_indices if self.data['single_ctrl'][i] in training_classes]
            self.eval_indices = [i for i in valid_indices if self.data['single_ctrl'][i] in validation_classes]
        else:
            np.random.shuffle(valid_indices)
            split_idx = int(len(valid_indices) * 0.7)
            self.train_indices = valid_indices[:split_idx]
            self.eval_indices = valid_indices[split_idx:]
    def __len__(self):
        return len(self.data['single_ctrl'])
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
    def _preprocess(self, genes, expressions):
        genes = torch.tensor(genes, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float)
        if len(genes) < self.max_length:
            padding_length = self.max_length - len(genes)
            genes = torch.cat([genes, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)])
            original_exps = torch.cat([expressions, torch.full((padding_length,), self.pad_value, dtype=torch.float)])
        else:
            if self.args.use_weighted_sampling:
                weights = expressions - expressions.min() + 1e-5
                indices = torch.multinomial(weights, self.max_length, replacement=False)
            else:
                indices = torch.randperm(len(genes))[:self.max_length]
            genes = genes[indices]
            original_exps = expressions[indices]
            padding_length = 0
        if self.preprocess_mode == "bin":
            expressions = self._binning(original_exps)
        else:
            expressions = original_exps
        return genes, expressions, padding_length
    def __getitem__(self, index):
        genes = self.data['genes'][index]
        expressions = self.data['expressions'][index]
        cls_label = self.data['single_ctrl'][index]
        genes, expressions, padding_length = self._preprocess(genes, expressions)
        return {
            'genes': genes,
            'expr': expressions,
            'cls_label': cls_label
        }
def set_seed(use_cuda=True, seed_value = None):
    if seed_value is None:
        seed_value = random.randint(0, 10000)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False

def evaluate(args, vocab, model, eval_loader, embedding_base=None, perturb_trans=None):
    total_loss = 0
    num_batches = 0
    for i, data_dict in enumerate(eval_loader):
        with torch.no_grad():
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            batch_size = data_dict["genes"].size(0)
            input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
            src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
            output_dict = model(input_gene_ids, torch.zeros_like(target_values),
                                src_key_padding_mask=src_key_padding_mask, base_emb=embedding_base,
                                perturb_trans=perturb_trans,perturb_cls=data_dict["cls_label"])
            output_values = output_dict["mvc_output"]
            loss_mse = F.mse_loss(output_values[:, 1:], target_values[:, 1:])
            total_loss += loss_mse.item()
            num_batches += 1
    loss_eval = total_loss / num_batches
    return loss_eval
def prepare_inputs(vocab, args, data_dict, batch_size):
    input_gene_ids = torch.cat((torch.Tensor([vocab["<cls>"]] * batch_size).cuda().unsqueeze(1), data_dict["genes"]),
                               dim=1).long()
    target_values = torch.cat((torch.Tensor([args.cls_value] * batch_size).cuda().unsqueeze(1), data_dict["expr"]),
                              dim=1)
    return input_gene_ids, target_values
def run(args, vocab, dataset):
    num_train = int(len(dataset) * train_ratio)
    base_loader = DataLoader(Subset(dataset, dataset.base_indices), batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(Subset(dataset, dataset.train_indices), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, dataset.eval_indices), batch_size=args.batch_size, shuffle=False)
    model = scModel(vocab, args)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    except:
        state_dict = torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()
    model = model.cuda()
    embedding_base = nn.Linear(1, 256, bias=False).cuda()
    optimizer = optim.Adam([{'params': embedding_base.parameters(), 'lr': args.lr},])
    base_epoch = 50
    stage2_epoch = 50
    for epoch in range(base_epoch):
        total_loss = 0
        num_batches = 0
        for i, data_dict in enumerate(base_loader):
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            batch_size = data_dict["genes"].size(0)
            input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
            src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
            output_dict = model(input_gene_ids, torch.zeros_like(target_values),
                                src_key_padding_mask=src_key_padding_mask, base_emb=embedding_base)
            output_values = output_dict["mvc_output"]
            loss_mse = F.mse_loss(output_values[:,1:], target_values[:,1:])
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            total_loss += loss_mse.item()
            num_batches += 1
        if epoch % args.print_epoch == 0:
            print(f'End of Epoch {epoch}, Average Loss: {total_loss / num_batches}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    embedding_base.eval()
    random_s1 = evaluate(args, vocab, model, eval_loader)
    random_s2 = evaluate(args, vocab, model, eval_loader, embedding_base)
    print("randoms1", random_s1, "randoms2", random_s2)
    ####real training
    dict_size = 256
    perturb_trans = nn.ModuleList([nn.Linear(dict_size, dict_size),
                                  nn.Linear(1, 256, bias=False)]).cuda()
    if args.model_structure in ["cnn_trans"]:
        optimizer = optim.Adam([{'params': embedding_base.parameters(), 'lr': 0.001},
                                {'params': perturb_trans.parameters(), 'lr': 0.005}])
    else:
        optimizer = optim.Adam([{'params': perturb_trans.parameters(), 'lr': 0.005}, ])
    for epoch in range(stage2_epoch):
        total_loss = 0
        num_batches = 0
        for i, data_dict in enumerate(train_loader):
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            batch_size = data_dict["genes"].size(0)
            input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
            src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
            output_dict = model(input_gene_ids, torch.zeros_like(target_values),
                                src_key_padding_mask=src_key_padding_mask, base_emb=embedding_base,
                                perturb_trans=perturb_trans,perturb_cls=data_dict["cls_label"])
            output_values = output_dict["mvc_output"]
            loss_mse = F.mse_loss(output_values[:,1:], target_values[:,1:])
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            total_loss += loss_mse.item()
            num_batches += 1
        if epoch % args.print_epoch == 0:
            print(f'End of Epoch {epoch}, Average Loss: {total_loss / num_batches}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    final_loss = evaluate(args, vocab, model, eval_loader, embedding_base, perturb_trans)
    print("eval fin", final_loss)
    return {
        "random_s1": random_s1,
        "random_s2": random_s2,
        "final_loss": final_loss,
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./save_pretrain/0527base/best_model.pt", help="Path to the trained model file.")
    parser.add_argument("--vocab_path", type=str, default="vocab.json", help="Path to the vocabulary file.")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation.")
    parser.add_argument("--pad-token",type=str,default="<pad>",help="The token to use for padding. Default is <pad>.")
    parser.add_argument("--cell_emb_style",type=str,choices=["cls", "avg-pool"],default="cls",help="The style of the input embedding. Default is continuous.")
    parser.add_argument("--n-bins",type=int,default=51,help="The number of bins to use for the binned input style. Default is 51.")
    parser.add_argument("--train_maxseq",type=int,default=512)
    parser.add_argument("--test_maxseq",type=int,default=512)
    parser.add_argument("--vocab-path",type=str,default="vocab.json",help="Path to the vocabulary file.")
    parser.add_argument("--batch_size",type=int,default=128,help="The batch size for training. Default is 64.")
    parser.add_argument("--lr",type=float,default=0.005,help="The learning rate for training. Default is 1e-3.")
    parser.add_argument("--eval_knn",action="store_true")
    parser.add_argument("--debug",action="store_true",help="break train and eval")
    parser.add_argument("--fp16",action="store_true",help="Whether to train in automatic mixed precision. Default is False.")
    parser.add_argument("--nlayers",type=int,default=6,help="The number of layers for the transformer. Default is 4.")
    parser.add_argument("--nheads",type=int,default=8,help="The number of heads for the transformer. Default is 4.")
    parser.add_argument("--embsize",type=int,default=256,help="The embedding size for the transformer. Default is 64.")
    parser.add_argument("--dropout",type=float,default=0.15,help="The dropout rate. Default is 0.15.")
    parser.add_argument("--train_ratio",type=float,default=0.7)
    parser.add_argument("--num_workers",type=int,default=16)
    parser.add_argument("--num_trials",type=int,default=5)
    parser.add_argument("--num_epochs",type=int,default=50)
    parser.add_argument("--print_epoch",type=int,default=1)
    parser.add_argument("--eval_epoch",type=int,default=5)
    parser.add_argument("--preprocess_mode",type=str,choices=["none", "bin"],default="none")
    parser.add_argument("--model_structure",type=str,default="transformer")
    parser.add_argument("--filter_name",type=str,default="dixit")
    parser.add_argument("--optimizer",type=str,default="adam")
    parser.add_argument("--input_directory",type=str,default="./data/downstreams/perturbation/processed_data")
    parser.add_argument("--use_weighted_sampling", action="store_true")
    parser.add_argument("--train_from_features", action="store_true")
    parser.add_argument("--zero_testing", action="store_true")
    parser.add_argument("--add_note", type=str, default="")
    args = parser.parse_args()
    args.mask_value = -1
    args.pad_value = 0
    args.cls_value = 0
    args.fp16 = True
    train_ratio = args.train_ratio
    all_accuracies = []
    all_accuracies2 = []
    all_accuracies3 = []
    all_knn_accuracies = []
    for trial in range(args.num_trials):
        set_seed(seed_value=trial)
        vocab = GeneVocab.from_file(Path(args.vocab_path))
        args.vocab = vocab
        input_path = Path(args.input_directory)
        input_file = input_path / f"{args.filter_name}_data.pt"
        data_saved_all = torch.load(input_file)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset = ExpressionDataset(data_saved_all, args.test_maxseq, vocab["<pad>"], args.pad_value, args=args)
        results_dict = run(args, vocab, dataset)
        all_accuracies.append(results_dict["random_s1"])
        all_accuracies2.append(results_dict["random_s2"])
        all_accuracies3.append(results_dict["final_loss"])
        print("current best losses", all_accuracies, all_accuracies2, all_accuracies3)

    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_accuracy2 = np.mean(all_accuracies2)
    std_accuracy2 = np.std(all_accuracies2)
    mean_accuracy3 = np.mean(all_accuracies3)
    std_accuracy3 = np.std(all_accuracies3)
    print(f'random_s1 Loss across trials: {mean_accuracy:.3f}±{std_accuracy:.3f}')
    print(f'random_s2 Loss across trials: {mean_accuracy2:.3f}±{std_accuracy2:.3f}')
    print(f'final_loss Loss across trials: {mean_accuracy3:.3f}±{std_accuracy3:.3f}')



