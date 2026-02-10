from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import torch
from rich.console import Console
import yaml

from src.data.chestmnist import make_chestmnist_dataloaders
from src.models import create_model
from src.utils.metrics import multilabel_auc_and_acc
from src.utils.seed import seed_everything

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, prefer EMA weights from the checkpoint when available (overrides config).",
    )
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_split(model, loader, device, threshold: float):
    model.eval()
    ys, yts, idxs = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        score = torch.sigmoid(logits).cpu().numpy()
        ys.append(score)
        yts.append(y.numpy())
        idxs.append(idx.numpy())

    y_score = np.concatenate(ys, 0)
    y_true = np.concatenate(yts, 0)
    idxs = np.concatenate(idxs, 0)

    # Ensure correct ordering + unique (even though non-distributed eval shouldn't have dups)
    order = np.argsort(idxs)
    y_score = y_score[order]
    y_true = y_true[order]
    uniq, uniq_pos = np.unique(idxs[order], return_index=True)
    y_score = y_score[uniq_pos]
    y_true = y_true[uniq_pos]

    m = multilabel_auc_and_acc(y_true=y_true, y_score=y_score, threshold=threshold)
    return m


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["run"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = make_chestmnist_dataloaders(
        root=cfg["data"]["root"],
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        distributed=False,
    )

    model = create_model(
        name=str(cfg["model"]["name"]),
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=False,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    use_ema_cfg = bool(cfg.get("eval", {}).get("use_ema", False))
    use_ema = bool(args.use_ema or use_ema_cfg)

    if use_ema and ("ema" in ckpt):
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)

    loader = loaders.val if args.split == "val" else loaders.test
    m = eval_split(model, loader, device, threshold=float(cfg["eval"]["threshold"]))
    console.print(f"{args.split.upper()} | AUC={m.auc:.4f} | ACC={m.acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
