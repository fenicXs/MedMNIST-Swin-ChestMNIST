# Swin-Base on MedMNIST v2: ChestMNIST (multi-label)

Train and evaluate a **Swin-Base** model (via [timm](https://github.com/huggingface/pytorch-image-models)) on **ChestMNIST** from **MedMNIST v2**.

This repo is intentionally **minimal, reproducible, and HPC-friendly** (single GPU or DDP).

---

## What this covers

- Dataset: **ChestMNIST** (NIH ChestX-ray14-derived), **14-label multi-label classification**
- Model: `swin_base_patch4_window7_224` (ImageNet pretrained)
- Input: **upsampled to 224×224** ("larger size" training)
- Loss: `BCEWithLogitsLoss` (optional `pos_weight`)
- Metrics:
  - **AUC**: mean of per-label ROC-AUC
  - **ACC**: micro accuracy over all label entries using threshold 0.5

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> **PyTorch**: install the correct CUDA build for your environment (SOL / A100).

---

## Data

We use the official `medmnist` python package, which downloads ChestMNIST automatically.

No manual preprocessing is required. Images are stored as 28×28 in the dataset and resized to 224×224 by transforms.

---

## Quickstart (single GPU)

```bash
python -m src.train --config configs/chestmnist_swinb_224.yaml
```

Checkpoints and logs are written to:

- `runs/<run_name>/`
- `runs/<run_name>/checkpoints/`

---

## Quickstart (DDP, 4 GPUs)

```bash
torchrun --nproc_per_node=4 -m src.train --config configs/chestmnist_swinb_224.yaml --ddp
```

---

## Evaluate a checkpoint

```bash
python -m src.eval   --config configs/chestmnist_swinb_224.yaml   --ckpt runs/chestmnist_swinb_224/checkpoints/best.pt   --split test
```

---

## Repro tips

- Use a fixed seed (`seed` in the config)
- Compare using the same metrics (AUC + ACC)
- For best results: train 3 seeds and report mean ± std

---

## Project layout

```text
.
├── configs/
│   └── chestmnist_swinb_224.yaml
├── scripts/
│   └── slurm_train_chestmnist_swinb.sh
└── src/
    ├── __init__.py
    ├── train.py
    ├── eval.py
    ├── data/
    │   ├── __init__.py
    │   └── chestmnist.py
    ├── models/
    │   ├── __init__.py
    │   └── timm_factory.py
    └── utils/
        ├── __init__.py
        ├── checkpoint.py
        ├── dist.py
        ├── metrics.py
        └── seed.py
```

---

## License

MIT (see `LICENSE`).
