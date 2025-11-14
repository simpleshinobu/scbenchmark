import json
import time
from pathlib import Path
from collections import Counter
from torchtext.vocab import Vocab
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset_utils.dataset_incre import ExpressionDataset, split_and_create_datasets_withsave_id
from models_utils.model import scModel
from util_ours.utils import masked_mse_loss, load_model, find_latest_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from util_ours.argparse_config import parse_arguments
from util_ours.utils import (create_vocab_access_stages,distribute_exclude_indices,aggregate_counts,select_exclude_1102,
                             select_exclude_fin, select_exclude_fin2, set_seed)

args = parse_arguments()
set_seed(args.seed)
is_main_process = (args.local_rank == 0) if args.local_rank != -1 else True
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()
args.batch_size = args.batch_size // world_size
if args.save_dir != "" and is_main_process:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(log_dir=save_dir / "tensorboard")
elif is_main_process:
    writer = SummaryWriter()
# Load data
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocab_dict = json.load(file)
    counter = Counter(vocab_dict)
    return Vocab(counter)

vocab = load_vocab(args.vocab_path)
args.vocab = vocab

if args.save_dir != "":
    if args.resume:
        latest_model_path, max_epoch = find_latest_checkpoint(args.save_dir)
ntokens = len(vocab)  # size of vocabulary
device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
model = scModel(vocab, args=args).to(device)
if args.local_rank != -1:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
if args.resume and max_epoch > 0:
    model = load_model(model, args, model_path=latest_model_path)
    start_epoch = max_epoch
else:
    start_epoch = 0
if args.load_model is not None:
    model = load_model(model, args, args.load_model)

def train(model: nn.Module, train_loader: DataLoader,  epoch: int, stage: int) -> None:
    model.train()
    total_loss, total_mse,  total_error= 0.0, 0.0, 0.0
    global total_iters
    log_interval = args.log_interval
    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        if total_iters <= args.warmup_iters and epoch == 1 and stage == 0:
            optimizer.param_groups[0]["lr"] = args.lr * total_iters / args.warmup_iters
        else:
            optimizer.param_groups[0]["lr"] = args.lr
        global_iter = epoch * num_batches + batch
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        input_gene_ids = data_dict["gene"]
        input_values = data_dict["masked_expr"]
        target_values = data_dict["expr"]
        ##add cls token
        input_gene_ids = torch.cat((torch.Tensor([vocab["<cls>"]]*input_gene_ids.size(0)).to(device).unsqueeze(1),input_gene_ids),dim=1).long()
        target_values = torch.cat((torch.Tensor([args.cls_value]*input_gene_ids.size(0)).to(device).unsqueeze(1),target_values),dim=1)
        input_values = torch.cat((torch.Tensor([args.cls_value]*input_gene_ids.size(0)).to(device).unsqueeze(1),input_values),dim=1)
        src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])
        output_dict = model(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask)
        output_values = output_dict["mlm_output"]
        positions_to_match = input_values.eq(args.mask_value)
        loss_mse = masked_mse_loss(output_values, target_values, positions_to_match)
        loss_mvc = masked_mse_loss(output_dict["mvc_output"], target_values, positions_to_match)
        loss = loss_mse + loss_mvc
        loss.backward()
        optimizer.step()
        total_iters += 1
        optimizer.zero_grad()
        total_loss += loss.item()
        total_mse += loss_mse.item()
        if is_main_process:
            writer.add_scalar("Loss/train", loss.item(), global_iter)
            writer.add_scalar("MSE/train", loss_mse.item(), global_iter)
        if batch % log_interval == 0 and batch != 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if is_main_process:
                print(f"{current_time} | epoch {epoch:3d} | batch {batch:3d}/{num_batches:3d} | "
                      f"lr: {optimizer.param_groups[0]['lr']:.5f} | loss: {total_loss / log_interval:.3f} | "
                      f"mse: {total_mse / log_interval:.3f}  | ")
                      # f"mre: {total_error / log_interval:.3f}")
            total_loss, total_mse,  total_error = 0.0, 0.0, 0.0
        if args.debug and batch == 50:
            break

train_dataset = ExpressionDataset(args.train_path, vocab, args, args.train_maxseq)
valid_dataset = ExpressionDataset(args.valid_path, vocab, args, args.test_maxseq, mask_ratio=args.eval_mask_ratio)
num_stages = args.num_stages  #default add base stage

valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if args.local_rank != -1 else None
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                          num_workers=args.num_workers, pin_memory=True)
best_val_loss = float("inf")
if is_main_process:
    print("Start training")

if args.load_dir:
    load_dir = Path(args.load_dir)
    base_model_path = load_dir / "model-stage0.pt"
    if base_model_path.exists():
        model = load_model(model, args, str(base_model_path))
        print("Base model loaded from", base_model_path)
    stage_sample_ids_path = load_dir / "stage_sample_ids.pt"
    stage_sample_ids = torch.load(stage_sample_ids_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stage_sample_ids, save_dir / "stage_sample_ids.pt")
    train_datasets, stage_sample_ids = split_and_create_datasets_withsave_id(train_dataset, num_stages,
                base_ratio=args.base_ratio_dataset, id_load=stage_sample_ids)  # list of dataset
else:
    if args.excludefin:
        exclude_idx, select_counts = select_exclude_1102(args, is_main_process)
    aggregate_exclude = aggregate_counts(select_counts, args.num_stages-1)
    aggregate_exclude = [0] + aggregate_exclude


    train_datasets, stage_sample_ids  = split_and_create_datasets_withsave_id(train_dataset, num_stages, base_ratio=args.base_ratio_dataset) #list of dataset
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stage_sample_ids, save_dir / "stage_sample_ids.pt")
    stage_tokens_indices = create_vocab_access_stages(vocab, num_stages, incre_only=args.incre_only,
                                                      base_ratio=args.base_ratio,exclude_idx=exclude_idx, keep_base_id=args.keep_base_id)

    stage_tokens_indices = distribute_exclude_indices(stage_tokens_indices, exclude_idx,
                                                      num_per_stage=aggregate_exclude,incre_only=args.incre_only)  ##跳过base
    torch.save(stage_tokens_indices, save_dir / "stage_tokens_indices.pt")
    if is_main_process:
        print("exclude", len(exclude_idx), "genes in the base training", "true length", len(list(set(exclude_idx))))
        print("id numbers", [len(item) for item in stage_tokens_indices], "add number", aggregate_exclude)
###optimization
for stage, dataset in enumerate(train_datasets): ##incremental stage, the first stage is base
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ##重新定义optimizer
    if args.load_dir and stage == 0 and args.skip_stage0: ##only load base model
        model_path = f"{args.save_dir}/model-stage{stage}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"BASE Model (NO TRAINING) saved for Stage {stage} at {model_path}")
        continue
    temp_token_idx = torch.Tensor(stage_tokens_indices[stage]).long()
    temp_token_idx = torch.sort(temp_token_idx,descending=False)[0]
    dataset.dataset.tmp_vocab_idx = temp_token_idx
    train_sampler = DistributedSampler(dataset, shuffle=True) if args.local_rank != -1 else None
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True)
    if is_main_process:
        print("*" * 40)
        print("stage", stage, "dataset length", len(train_loader))
    total_iters = 0
    best_loss_tensor = torch.tensor([float('inf')], device='cuda')
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader,  epoch=epoch, stage=stage)
        rank = dist.get_rank()
    if is_main_process and args.save_dir:
        model_path = f"{args.save_dir}/model-stage{stage}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved for Stage {stage} at {model_path}")

if is_main_process:
    writer.flush()
    writer.close()
if args.local_rank != -1:
    dist.destroy_process_group()

