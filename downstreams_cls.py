import scanpy as sc
import argparse
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from collections import OrderedDict
from models_utils.model import scModel
from util_ours.utils import GeneVocab
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
import datetime
import random
warnings.filterwarnings("ignore", category=UserWarning)
def read_data(h5ad_files, vocab, filter_name=""):
    data_saved_all = {
        "genes": [],
        "expressions": [],
        "cls_name": []
    }
    for h5ad_files_temp in h5ad_files:
        print("loading for {} ...".format(h5ad_files_temp))
        adata_temp = sc.read_h5ad(h5ad_files_temp)
        if filter_name == "pancread":
            tokens = adata_temp.var["Gene Symbol"].tolist()
            expressions = adata_temp.X
            class_labels = adata_temp.obs["Celltype"].tolist()
        elif filter_name == "Multiple_Sclerosis":
            tokens = adata_temp.var["gene_name"].tolist()
            expressions = adata_temp.X.toarray()
            class_labels = adata_temp.obs["celltype"].tolist()
        elif filter_name == "Myeloid":
            expressions = adata_temp.X
            tokens = list(adata_temp.var.T)
            class_labels = adata_temp.obs["cell_type"].tolist()
        elif filter_name == "lupus":
            expressions = adata_temp.X
            tokens = adata_temp.var_names.tolist()
            class_labels = adata_temp.obs["ct_cov"].tolist()
        elif filter_name in ["ircolitis"]:
            expressions = adata_temp.X.toarray()
            class_labels = adata_temp.obs["cancer_type"].tolist()
            genes_origin = adata_temp.var_names.tolist()
            tokens = [gene.split('|')[1] for gene in genes_origin]
        elif filter_name in ["myasthenia"]:
            expressions = adata_temp.X.toarray()
            class_labels = adata_temp.obs["Self_defining_Celltype"].tolist()
            tokens = adata_temp.var_names.tolist()
        genes = np.array([vocab[item] for item in tokens])
        match_ratio = sum([1 for t in tokens if t in vocab]) / len(tokens)
        unknown_values = vocab["<unknown>"]

        for i in range(expressions.shape[0]):
            expression_row = expressions[i, :]
            expr_indices = np.where((expression_row > 0) & (genes != unknown_values))[0]
            if expr_indices.size > 0 and isinstance(class_labels[i], str):
                data_saved_all["genes"].append(genes[expr_indices])
                data_saved_all["expressions"].append(expression_row[expr_indices])
                data_saved_all["cls_name"].append(class_labels[i])
    return data_saved_all

class ExpressionDataset(Dataset):
    def __init__(self, data, max_length, pad_token_id, pad_value, args = None):
        self.data = data
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.args = args
        self.preprocess_mode = args.preprocess_mode
        self.class_label_map = {label: idx for idx, label in enumerate(set(self.data['cls_name']))}
    def __len__(self):
        return len(self.data['cls_name'])
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
        cls_label = self.data['cls_name'][index]
        genes, expressions, padding_length = self._preprocess(genes, expressions)
        cls_label_idx = self.class_label_map[cls_label]
        return {
            'genes': genes,
            'expr': expressions,
            'cls_label': cls_label_idx
        }
def extract_features(model, data_loader, vocab, args):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data_dict in data_loader:
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            batch_size = data_dict["genes"].size(0)
            input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
            src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
            output_dict = model(input_gene_ids, target_values, src_key_padding_mask=src_key_padding_mask)
            cell_embeddings = output_dict['cell_emb']
            embeddings.append(cell_embeddings.cpu().numpy())
            labels.append(data_dict['cls_label'].cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels
def evaluate_knn(train_embeddings, train_labels, eval_embeddings, eval_labels, k=10):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', n_jobs=4)
    knn.fit(train_embeddings, train_labels)
    predicted_labels = knn.predict(eval_embeddings)
    accuracy = accuracy_score(eval_labels, predicted_labels)
    del knn, predicted_labels
    return accuracy * 100


def set_seed(use_cuda=True):
    seed_value = random.randint(0, 10000)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False
def prepare_inputs(vocab, args, data_dict, batch_size):
    input_gene_ids = torch.cat((torch.Tensor([vocab["<cls>"]] * batch_size).cuda().unsqueeze(1), data_dict["genes"]),
                               dim=1).long()
    target_values = torch.cat((torch.Tensor([args.cls_value] * batch_size).cuda().unsqueeze(1), data_dict["expr"]),
                              dim=1)
    return input_gene_ids, target_values
def evaluate(model, classifier, eval_loader, vocab, args):
    model.eval()
    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data_dict in eval_loader:
            if args.train_from_features:
                cell_embeddings, cls_labels = data_dict
                cell_embeddings, cls_labels = cell_embeddings.cuda(), cls_labels.cuda()
            else:
                data_dict = {k: v.cuda() for k, v in data_dict.items()}
                batch_size = data_dict["genes"].size(0)
                input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
                src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
                output_dict = model(input_gene_ids, target_values, src_key_padding_mask=src_key_padding_mask)
                cell_embeddings = output_dict['cell_emb']
                cls_labels = data_dict['cls_label']
            logits = classifier(cell_embeddings)
            _, predicted = torch.max(logits.data, 1)
            total += cls_labels.size(0)
            correct += (predicted == cls_labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
def run(args, vocab, dataset):

    num_train = int(len(dataset) * train_ratio)
    train_dataset, eval_dataset = random_split(dataset, [num_train, len(dataset) - num_train])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
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
    if args.eval_knn:
        train_embeddings, train_labels = extract_features(model, train_loader, vocab, args)
        eval_embeddings, eval_labels = extract_features(model, eval_loader, vocab, args)
        knn_accuracy = evaluate_knn(train_embeddings, train_labels, eval_embeddings, eval_labels)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{current_time}:KNN Accuracy on evaluation data: {knn_accuracy:.2f}%')
    else:
        knn_accuracy = 0.0
        train_embeddings, train_labels = extract_features(model, train_loader, vocab, args)
        eval_embeddings, eval_labels = extract_features(model, eval_loader, vocab, args)
    classifier = nn.Linear(args.embsize, len(dataset.class_label_map.keys())).cuda()
    if args.optimizer == "adam":
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    classifier.train()
    best_accuracy = 0
    best_epoch = 0
    if args.train_from_features:
        train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32).cuda()
        train_labels = torch.tensor(train_labels, dtype=torch.long).cuda()
        eval_embeddings = torch.tensor(eval_embeddings, dtype=torch.float32).cuda()
        eval_labels = torch.tensor(eval_labels, dtype=torch.long).cuda()
        train_dataset = TensorDataset(train_embeddings, train_labels)
        eval_dataset = TensorDataset(eval_embeddings, eval_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    for epoch in range(1,args.num_epochs+1):
        total_loss = 0
        num_batches = 0

        for i, data_dict in enumerate(train_loader):
            if args.train_from_features:
                cell_embeddings, cls_labels = data_dict
                cell_embeddings, cls_labels = cell_embeddings.cuda(), cls_labels.cuda()
            else:
                data_dict = {k: v.cuda() for k, v in data_dict.items()}
                batch_size = data_dict["genes"].size(0)
                with torch.no_grad():
                    input_gene_ids, target_values = prepare_inputs(vocab, args, data_dict, batch_size)
                    src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
                    output_dict = model(input_gene_ids, target_values, src_key_padding_mask=src_key_padding_mask)
                    cell_embeddings = output_dict['cell_emb']
                cls_labels = data_dict['cls_label']
            logits = classifier(cell_embeddings)
            loss = criterion(logits, cls_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if args.debug:
                break
        if epoch % args.print_epoch == 0:
            print(f'End of Epoch {epoch}, Average Loss: {total_loss / num_batches}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
        if epoch % args.eval_epoch == 0:
            accuracy = evaluate(model, classifier, eval_loader, vocab, args)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
    return {
        "best_acc": best_accuracy,
        "knn_acc": knn_accuracy
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./save_pretrain/0527base/best_model.pt", help="Path to the trained model file.")
    parser.add_argument("--vocab_path", type=str, default="vocab.json", help="Path to the vocabulary file.")
    parser.add_argument("--pad-token",type=str,default="<pad>",help="The token to use for padding. Default is <pad>.")
    parser.add_argument("--cell_emb_style",type=str,choices=["cls", "avg-pool"],default="avg-pool",help="The style of the input embedding. Default is continuous.")
    parser.add_argument("--n-bins",type=int,default=51,help="The number of bins to use for the binned input style. Default is 51.")
    parser.add_argument("--train_maxseq",type=int,default=512)
    parser.add_argument("--test_maxseq",type=int,default=512)
    parser.add_argument("--vocab-path",type=str,default="vocab.json",help="Path to the vocabulary file.")
    parser.add_argument("--batch_size",type=int,default=64,help="The batch size for training. Default is 64.")
    parser.add_argument("--lr",type=float,default=0.005,help="The learning rate for training. Default is 1e-3.")
    parser.add_argument("--eval_knn",action="store_true")
    parser.add_argument("--debug",action="store_true",help="break train and eval")
    parser.add_argument("--fp16",action="store_true",help="Whether to train in automatic mixed precision. Default is False.")
    parser.add_argument("--nlayers",type=int,default=6,help="The number of layers for the transformer. Default is 4.")
    parser.add_argument("--nheads",type=int,default=8,help="The number of heads for the transformer. Default is 4.")
    parser.add_argument("--embsize",type=int,default=256,help="The embedding size for the transformer. Default is 64.")
    parser.add_argument("--dropout",type=float,default=0.15,help="The dropout rate. Default is 0.15.")
    parser.add_argument("--train_ratio",type=float,default=0.7)
    parser.add_argument("--num_workers",type=int,default=6)
    parser.add_argument("--num_trials",type=int,default=5)
    parser.add_argument("--num_epochs",type=int,default=50)
    parser.add_argument("--print_epoch",type=int,default=10)
    parser.add_argument("--eval_epoch",type=int,default=5)
    parser.add_argument("--preprocess_mode",type=str,choices=["none", "bin"],default="none")
    parser.add_argument("--model_structure",type=str,default="transformer")
    parser.add_argument("--filter_name",type=str,default="pancread")
    parser.add_argument("--optimizer",type=str,default="adam")
    parser.add_argument("--input_directory",type=str,default="./data/downstreams/classification/processed_data")
    parser.add_argument("--use_weighted_sampling", action="store_true")
    parser.add_argument("--train_from_features", action="store_true")
    parser.add_argument("--add_note", type=str, default="")
    args = parser.parse_args()
    args.mask_value = -1
    args.pad_value = 0
    args.cls_value = 0
    args.fp16 = True
    train_ratio = args.train_ratio
    all_accuracies = []
    all_knn_accuracies = []
    for trial in range(args.num_trials):
        set_seed()
        vocab = GeneVocab.from_file(Path(args.vocab_path))
        args.vocab = vocab
        input_path = Path(args.input_directory)
        input_file = input_path / f"{args.filter_name}_data.pt"
        data_saved_all = torch.load(input_file)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset = ExpressionDataset(data_saved_all, args.test_maxseq, vocab["<pad>"], args.pad_value, args=args)
        results_dict = run(args, vocab, dataset)
        all_accuracies.append(results_dict["best_acc"])
        all_knn_accuracies.append(results_dict["knn_acc"])
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    print(all_accuracies)
    mean_knn_accuracy = np.mean(all_knn_accuracies)
    std_knn_accuracy = np.std(all_knn_accuracies)
    print(f'Average KNN Accuracy/ cls Accuracy across trials: {mean_knn_accuracy:.3f}±{std_knn_accuracy:.3f}/{mean_accuracy:.3f}±{std_accuracy:.3f}')
    print("#" * 80, "\n", "#" * 80)


