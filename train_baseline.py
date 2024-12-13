import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset_utils.dataset import ExpressionDataset
from models_utils.model import scModel
from util_ours.utils import masked_mse_loss, GeneVocab, load_model, find_latest_checkpoint
from evaluation import evaluate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="./data/train_data.parquet")
parser.add_argument("--valid_path", type=str, default="./data/test_data.parquet")

parser.add_argument("-s", "--save-dir", type=str, default="")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--pad-token", type=str, default="<pad>")
parser.add_argument("--cell_emb_style", type=str, choices=["cls", "avg-pool"], default="cls")
parser.add_argument("--n-bins", type=int, default=51)
parser.add_argument("--train_maxseq", type=int, default=512)
parser.add_argument("--test_maxseq", type=int, default=512)
parser.add_argument("--mask_ratio", type=float, default=0.40)
parser.add_argument("--eval_mask_ratio", type=float, default=0.40)
parser.add_argument("--vocab-path", type=str, default="vocab.json")
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--use_mvc", action="store_true")
parser.add_argument("--cache_dir", type=str, default="")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--nlayers", type=int, default=6)
parser.add_argument("--nheads", type=int, default=8)
parser.add_argument("--embsize", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--save_interval", type=int, default=1)
parser.add_argument("--log-interval", type=int, default=500)
parser.add_argument("--warmup_iters", type=int, default=10000)
parser.add_argument("--preprocess_mode", type=str, choices=["none", "bin"], default="none")
parser.add_argument("--model_structure", type=str, default="transformer")
parser.add_argument("--use_weighted_sampling", action="store_true")
parser.add_argument("--add_note", type=str, default="")
args = parser.parse_args()
args.mask_value = -1
args.pad_value = 0
args.cls_value = 0
args.fp16 = True


#settings
is_main_process = (args.local_rank == 0) if args.local_rank != -1 else True
if is_main_process:
    print(args)
def setup_distributed():
    # Set the device according to local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
def cleanup_distributed():
    dist.destroy_process_group()
if args.local_rank != -1:
    setup_distributed()
world_size = dist.get_world_size()
rank = dist.get_rank()
##adjust batch_size for parallel
args.batch_size = args.batch_size // world_size


if args.save_dir != "" and is_main_process:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(log_dir=save_dir / "tensorboard")
elif is_main_process:
    writer = SummaryWriter()

vocab = GeneVocab.from_file(Path(args.vocab_path))
args.vocab = vocab
if args.save_dir != "":
    if is_main_process:
        with open(save_dir / "vocab.json", "w") as f:
            json.dump({token: index for token, index in vocab.get_stoi().items()},
                f,indent=2,)
    if args.resume:
        latest_model_path, max_epoch = find_latest_checkpoint(args.save_dir)

# Load the full dataset
train_dataset = ExpressionDataset(args.train_path, vocab, args, args.train_maxseq)
valid_dataset = ExpressionDataset(args.valid_path, vocab, args, args.test_maxseq, mask_ratio=args.eval_mask_ratio)
train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else None
valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if args.local_rank != -1 else None
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)

# # Create and train model
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(model: nn.Module, train_loader: DataLoader, epoch: int) -> None:
    model.train()
    total_loss, total_mse,  total_error= 0.0, 0.0, 0.0
    global total_iters
    log_interval = args.log_interval
    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        if total_iters <= args.warmup_iters and epoch == 1:
            optimizer.param_groups[0]["lr"] = args.lr * total_iters / args.warmup_iters
        global_iter = epoch * num_batches + batch
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        input_gene_ids = data_dict["gene"]
        input_values = data_dict["masked_expr"]
        target_values = data_dict["expr"]
        ##add cls token for all inputs
        input_gene_ids = torch.cat((torch.Tensor([vocab["<cls>"]]*input_gene_ids.size(0)).to(device).unsqueeze(1),input_gene_ids),dim=1).long()
        target_values = torch.cat((torch.Tensor([args.cls_value]*input_gene_ids.size(0)).to(device).unsqueeze(1),target_values),dim=1)
        input_values = torch.cat((torch.Tensor([args.cls_value]*input_gene_ids.size(0)).to(device).unsqueeze(1),input_values),dim=1)

        src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])
        output_dict = model(input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask)
        output_values = output_dict["mlm_output"]
        positions_to_match = input_values.eq(args.mask_value)
        loss_mse = masked_mse_loss(output_values, target_values, positions_to_match)
        loss_mvc = masked_mse_loss(output_dict["mvc_output"], target_values, positions_to_match)
        if args.use_mvc:
            loss = loss_mse + loss_mvc
        else:
            loss = loss_mse
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
            total_loss, total_mse,  total_error = 0.0, 0.0, 0.0
        if args.debug and batch == 20:
            break

best_val_loss = float("inf")

if is_main_process:
    print("Start training")
total_iters = 0
for epoch in range(start_epoch+1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    rank = dist.get_rank()
    total_loss_mlm = evaluate(model, valid_loader, args, is_main_process, test_dataset=train_loader.dataset)
    if is_main_process and total_loss_mlm is not None:
        if total_loss_mlm < best_val_loss:
            best_val_loss = total_loss_mlm
            print(f"New best model found at Epoch {epoch} with Loss {best_val_loss:.4f}")
            if args.save_dir:
                torch.save(model.state_dict(), f"{args.save_dir}/best_model.pt")
    best_loss_tensor = torch.tensor([best_val_loss], device='cuda')
    dist.broadcast(best_loss_tensor, src=0)
    if not is_main_process:
        best_val_loss = best_loss_tensor.item()
    if epoch != -1 and (epoch) % args.save_interval == 0 and is_main_process and not args.debug:
        torch.save(model.state_dict(), args.save_dir + f"/model-ep{epoch}.pt")

if is_main_process:
    writer.flush()
    writer.close()
if args.local_rank != -1:
    cleanup_distributed()

