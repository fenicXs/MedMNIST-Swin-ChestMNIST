from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import numpy as np

class IndexedDataset(Dataset):
    """Wrap a dataset to also return the sample index."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # ImageNet stats are standard for timm pretrained backbones.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def make_chestmnist_dataloaders(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoaders:
    """Create train/val/test dataloaders for ChestMNIST via medmnist.

    If distributed=True, uses DistributedSampler (train shuffled, val/test sequential).
    Val/test datasets are wrapped to return indices so we can de-duplicate padded samples.
    """
    try:
        from medmnist import ChestMNIST
    except Exception as e:
        raise RuntimeError("medmnist is not installed. Please `pip install medmnist`.") from e

    train_tfms, eval_tfms = build_transforms(image_size)

    train_ds = ChestMNIST(split="train", root=root, download=True, transform=train_tfms, as_rgb=True)
    val_ds = ChestMNIST(split="val", root=root, download=True, transform=eval_tfms, as_rgb=True)
    test_ds = ChestMNIST(split="test", root=root, download=True, transform=eval_tfms, as_rgb=True)

    # Wrap val/test to also return indices (needed for exact evaluation under DistributedSampler padding).
    val_ds_i = IndexedDataset(val_ds)
    test_ds_i = IndexedDataset(test_ds)

    # labels are numpy arrays; convert to float tensors inside collate
    def collate_train(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        labels = torch.from_numpy(np.stack(labels, axis=0)).float()
        return imgs, labels

    def collate_eval(batch):
        imgs, labels, idxs = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        labels = torch.from_numpy(np.stack(labels, axis=0)).float()
        idxs = torch.as_tensor(idxs, dtype=torch.int64)
        return imgs, labels, idxs

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds_i, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_ds_i, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds_i,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_eval,
    )
    test_loader = DataLoader(
        test_ds_i,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_eval,
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)
