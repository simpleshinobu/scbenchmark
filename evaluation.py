import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
from dataset_utils.dataset import ExpressionDataset
from models_utils.model import scModel
from util_ours.utils import GeneVocab, masked_relative_error
import math
import torch.distributed as dist
import os

def evaluate(model, loader, args = None, is_main_process=True, test_dataset=None):
    total_losses = {
        "total_loss": 0,
        "total_wloss": 0,
        "total_relative_error": 0,
        "total_loss_log2": 0,
        "total_loss_log2_log11": 0,
        "total_loss_log11": 0
    }
    counts = {
        "count_log2": 0,
        "count_log2_log11": 0,
        "count_log11": 0
    }
    is_distributed = dist.is_initialized()
    model.eval()
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(loader, total=len(loader), desc=f"Evaluating")):
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            input_gene_ids = torch.cat((torch.Tensor(
                [args.vocab["<cls>"]] * data_dict["gene"].size(0)).cuda().unsqueeze(1), data_dict["gene"]),
                                       dim=1).long()
            target_values = torch.cat(
                (torch.Tensor([args.cls_value] * input_gene_ids.size(0)).cuda().unsqueeze(1), data_dict["expr"]),
                dim=1)
            input_values = torch.cat((torch.Tensor([args.cls_value] * input_gene_ids.size(0)).cuda().unsqueeze(1),
                                      data_dict["masked_expr"]), dim=1)
            src_key_padding_mask = input_gene_ids.eq(args.vocab["<pad>"])
            output_dict = model(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask)
            output = output_dict["mlm_output"]
            positions_to_match = input_values.eq(-1)

            if args.preprocess_mode == "bin":
                output = test_dataset.unbinning_batch(output[:, 1:].long().cpu(),
                                                      data_dict["expr_orin"].cpu()).cuda()
                positions_to_match = positions_to_match[:, 1:]
                target_values = data_dict["expr_orin"].cuda()
            try:
                diff = F.mse_loss(output[positions_to_match], target_values[positions_to_match], reduction="none")
            except:
                print("output.size()", output.size(),
                      "target_values.size()", target_values.size(),
                      "input_gene_ids.size()", input_gene_ids.size(),
                      "src_key_padding_mask.size()", src_key_padding_mask.size(),
                      "input_values.size()", input_values.size(),
                      "positions_to_match.size()", positions_to_match.size())
                os.exit(1)
            losses = {
                "loss": diff.sum() / positions_to_match.sum(),
                "loss_weighted": (diff * target_values[positions_to_match]).sum() / positions_to_match.sum(),
                "loss_relative": masked_relative_error(output, target_values, positions_to_match)
            }
            total_losses["total_loss"] += losses["loss"]
            total_losses["total_wloss"] += losses["loss_weighted"]
            total_losses["total_relative_error"] += losses["loss_relative"]
            loss_ranges = [(0, math.log(2)), (math.log(2), math.log(11)), (math.log(11), float('inf'))]
            labels = ['log2', 'log2_log11', 'log11']
            for idx, (start, end) in enumerate(loss_ranges):
                positions = (target_values[positions_to_match] > start) & (target_values[positions_to_match] <= end)
                if positions.sum() > 0:
                    total_losses[f"total_loss_{labels[idx]}"] += diff[positions].sum().item()
                    counts[f"count_{labels[idx]}"] += positions.sum().item()
            if args.debug and i == 10:
                break
        loss_tensors = torch.tensor([total_losses[k] for k in total_losses]).cuda()
        count_tensors = torch.tensor([counts[k] for k in counts]).cuda()
        if is_distributed:
            dist.all_reduce(loss_tensors, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensors, op=dist.ReduceOp.SUM)
            num_samples = len(loader) * dist.get_world_size()
        else:
            num_samples = len(loader)
        if is_main_process:
            avg_loss = loss_tensors[0] / num_samples
            avg_wloss = loss_tensors[1] / num_samples
            avg_relative_error = loss_tensors[2] / num_samples
            avg_loss_log2 = loss_tensors[3] / count_tensors[0] if count_tensors[0] > 0 else 0
            avg_loss_log2_log11 = loss_tensors[4] / count_tensors[1] if count_tensors[1] > 0 else 0
            avg_loss_log11 = loss_tensors[5] / count_tensors[2] if count_tensors[2] > 0 else 0
            print(f"\nMean Loss\tWeighted Loss\tRelative Loss\tlog2 Loss\tlog2-log11 Loss\tbigger than log11 Loss")
            print(f"{avg_loss:.5f}             {avg_wloss:.5f}             {avg_relative_error:.5f}             "
                  f"{avg_loss_log2:.5f}             {avg_loss_log2_log11:.5f}             {avg_loss_log11:.5f}")
        return avg_loss if is_main_process else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default="./data/test_data.parquet")
    parser.add_argument("--model_path", type=str, default="./save_pretrain/baseline/best_model.pt")
    parser.add_argument("--vocab_path", type=str, default="vocab.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pad-token", type=str, default="<pad>")
    parser.add_argument("--cell_emb_style", type=str, choices=["cls", "avg-pool"], default="cls")
    parser.add_argument("--n-bins", type=int, default=51)
    parser.add_argument("--train_maxseq", type=int, default=512)
    parser.add_argument("--test_maxseq", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.40)
    parser.add_argument("--eval_mask_ratio", type=float, default=0.40)
    parser.add_argument("--vocab-path", type=str, default="vocab.json")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_weighted_sampling", action="store_true")
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--embsize", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--preprocess_mode", type=str, choices=["none", "bin"], default="none")
    parser.add_argument("--model_structure", type=str, default="transformer")

    args = parser.parse_args()
    args.mask_value = -1
    args.pad_value = 0
    args.cls_value = 0
    args.fp16 = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = GeneVocab.from_file(Path(args.vocab_path))
    args.vocab = vocab
    print("evaluating for ", args.eval_path)
    test_dataset = ExpressionDataset(args.eval_path, vocab, args, args.test_maxseq, mask_ratio=args.eval_mask_ratio)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, shuffle=False)
    model = scModel(vocab, args)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        state_dict = torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.cuda()
    evaluate(model, test_loader, args)