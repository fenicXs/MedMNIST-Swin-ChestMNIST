from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.data.chestmnist import make_chestmnist_dataloaders
from src.models import create_model
from src.utils.checkpoint import save_checkpoint
from src.utils.dist import broadcast_tensor, init_distributed, is_main_process
from src.utils.metrics import multilabel_auc_and_acc
from src.utils.seed import seed_everything

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    p.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (torchrun).")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cosine_warmup_lr(
    epoch: int,
    step_in_epoch: int,
    steps_per_epoch: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> float:
    """Per-step cosine schedule with linear warmup."""
    cur_step = epoch * steps_per_epoch + step_in_epoch
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if warmup_steps > 0 and cur_step < warmup_steps:
        return base_lr * float(cur_step) / float(max(1, warmup_steps))

    # cosine decay
    progress = float(cur_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def compute_pos_weight_from_dataset(dataset) -> torch.Tensor:
    """Compute pos_weight = neg/pos for each label from the full training dataset."""
    if not hasattr(dataset, "labels"):
        raise RuntimeError("Dataset does not expose `labels`; cannot compute pos_weight.")
    y = np.asarray(dataset.labels)  # (N, L)
    pos = y.sum(axis=0).astype(np.float64)
    total = float(y.shape[0])
    neg = total - pos
    eps = 1e-6
    pw = neg / (pos + eps)
    pw = np.clip(pw, 1.0, 50.0)  # keep stable
    return torch.tensor(pw, dtype=torch.float32)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    *,
    epoch: int,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
):
    model.train()
    running_loss = 0.0

    base_lr = float(cfg["optim"]["lr"])
    min_lr = float(cfg["sched"]["min_lr"])
    warmup_epochs = int(cfg["sched"]["warmup_epochs"])
    total_epochs = int(cfg["train"]["epochs"])

    # for DistributedSampler
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        lr = cosine_warmup_lr(
            epoch,
            step,
            steps_per_epoch,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=bool(cfg["train"]["amp"])):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if cfg["train"].get("grad_clip_norm", 0) and cfg["train"]["grad_clip_norm"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, threshold: float, distributed: bool) -> Tuple[float, float]:
    """Evaluate on val/test.

    loader yields (x, y, idx). If distributed, indices can be duplicated due to sampler padding,
    so we de-duplicate by idx after gathering.
    """
    model.eval()
    scores_list = []
    targets_list = []
    idxs_list = []

    for batch in loader:
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        scores_list.append(scores)
        targets_list.append(y.numpy())
        idxs_list.append(idx.numpy())

    y_score = np.concatenate(scores_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)
    idxs = np.concatenate(idxs_list, axis=0)

    if distributed:
        # Gather arrays from all ranks via all_gather_object
        import torch.distributed as dist

        world = dist.get_world_size()
        gathered_scores = [None for _ in range(world)]
        gathered_true = [None for _ in range(world)]
        gathered_idxs = [None for _ in range(world)]
        dist.all_gather_object(gathered_scores, y_score)
        dist.all_gather_object(gathered_true, y_true)
        dist.all_gather_object(gathered_idxs, idxs)

        y_score = np.concatenate(gathered_scores, axis=0)
        y_true = np.concatenate(gathered_true, axis=0)
        idxs = np.concatenate(gathered_idxs, axis=0)

    # De-duplicate by idx (important when DistributedSampler pads)
    order = np.argsort(idxs)
    idxs = idxs[order]
    y_score = y_score[order]
    y_true = y_true[order]
    uniq, uniq_pos = np.unique(idxs, return_index=True)
    y_score = y_score[uniq_pos]
    y_true = y_true[uniq_pos]

    m = multilabel_auc_and_acc(y_true=y_true, y_score=y_score, threshold=threshold)
    return m.auc, m.acc


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dist_info = init_distributed()
    distributed = bool(args.ddp and dist_info.is_distributed)

    seed_everything(int(cfg["run"]["seed"]) + dist_info.rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if distributed:
        device = torch.device("cuda", dist_info.local_rank)

    # Outputs
    run_name = cfg["run"]["name"]
    out_root = Path(cfg["run"]["output_dir"]) / run_name
    ckpt_dir = out_root / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(out_root)) if is_main_process() else None

    # Data
    dcfg = cfg["data"]
    loaders = make_chestmnist_dataloaders(
        root=dcfg["root"],
        image_size=int(dcfg["image_size"]),
        batch_size=int(dcfg["batch_size"]),
        num_workers=int(dcfg["num_workers"]),
        pin_memory=bool(dcfg["pin_memory"]),
        distributed=distributed,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
    )

    # Model
    mcfg = cfg["model"]
    model = create_model(
        name=str(mcfg["name"]),
        num_classes=int(mcfg["num_classes"]),
        pretrained=bool(mcfg["pretrained"]),
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[dist_info.local_rank], output_device=dist_info.local_rank, find_unused_parameters=False)

    # Loss
    use_pos_weight = bool(cfg["train"].get("use_pos_weight", False))
    pos_weight = None
    if use_pos_weight:
        num_classes = int(mcfg["num_classes"])
        if is_main_process():
            pw = compute_pos_weight_from_dataset(loaders.train.dataset).to(device)
        else:
            pw = torch.empty(num_classes, dtype=torch.float32, device=device)
        if distributed:
            pw = broadcast_tensor(pw, src=0)
        pos_weight = pw

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optim
    ocfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ocfg["lr"]),
        weight_decay=float(ocfg["weight_decay"]),
        betas=tuple(float(b) for b in ocfg.get("betas", [0.9, 0.999])),
    )

    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

    # Resume
    start_epoch = 0
    best_auc = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        best_auc = float(ckpt.get("best_metric", best_auc))

    # Train loop
    epochs = int(cfg["train"]["epochs"])
    threshold = float(cfg["eval"]["threshold"])
    steps_per_epoch = len(loaders.train)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            cfg=cfg,
            steps_per_epoch=steps_per_epoch,
        )

        val_auc, val_acc = evaluate(model, loaders.val, device, threshold=threshold, distributed=distributed)

        if is_main_process():
            console.print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/auc", val_auc, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)

            # Save last
            save_checkpoint(
                ckpt_dir / "last.pt",
                model.module if isinstance(model, DDP) else model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                best_metric=best_auc,
            )

            # Save best
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model.module if isinstance(model, DDP) else model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                    best_metric=best_auc,
                )

    # Final test on best (all ranks compute; rank0 prints)
    import torch.distributed as dist

    if distributed:
        dist.barrier()

    best_path = ckpt_dir / "best.pt"
    ckpt = torch.load(best_path, map_location="cpu")
    (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])

    test_auc, test_acc = evaluate(model, loaders.test, device, threshold=threshold, distributed=distributed)

    if is_main_process():
        console.print(f"Training complete. Best val_auc={best_auc:.4f}")
        console.print(f"Test  | auc={test_auc:.4f} | acc={test_acc:.4f}")
        writer.add_scalar("test/auc", test_auc, epochs)
        writer.add_scalar("test/acc", test_acc, epochs)
        writer.close()


if __name__ == "__main__":
    main()
