"""
dataloader.py — PyTorch DataLoader for fake image detection
============================================================
Usage:
    from dataloader import build_dataloaders
    loaders = build_dataloaders(processed_root="./data/processed", batch_size=64)
    for images, labels in loaders["train"]:
        ...  # images: [B,3,224,224]  labels: [B] (0=real, 1=fake)
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NORMALIZATION = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def get_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            NORMALIZATION,
        ])
    return transforms.Compose([transforms.ToTensor(), NORMALIZATION])


class FakeImageDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None, source_name: str = "unknown"):
        self.root = Path(root) / split
        self.transform = transform or get_transform(split)
        self.source_name = source_name
        self.samples: List = []
        for label_name, label in (("real", 0), ("fake", 1)):
            folder = self.root / label_name
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((p, label))
        if not self.samples:
            raise FileNotFoundError(f"No images found at {self.root}. Run: python dataset.py")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), label

    @property
    def labels(self): return [l for _, l in self.samples]

    def class_counts(self): labels = self.labels; return {"real": labels.count(0), "fake": labels.count(1)}

    def __repr__(self):
        c = self.class_counts()
        return f"FakeImageDataset(source={self.source_name!r}, split={self.root.name!r}, n={len(self)}, real={c['real']}, fake={c['fake']})"


def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    all_labels = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets: all_labels.extend(ds.labels)
    else:
        all_labels = dataset.labels
    counts = [all_labels.count(c) for c in (0, 1)]
    total  = len(all_labels)
    weights = [total / (2 * counts[l]) for l in all_labels]
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


def build_dataloaders(
    processed_root: str = "./data/processed",
    datasets: Optional[List[str]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_classes: bool = True,
) -> Dict[str, DataLoader]:
    root = Path(processed_root)
    if datasets is None:
        datasets = [d.name for d in root.iterdir() if d.is_dir()] if root.exists() else []

    loaders = {}
    for split in ("train", "val", "test"):
        split_datasets = []
        for ds_name in datasets:
            ds_path = root / ds_name
            if not (ds_path / split).exists(): continue
            try:
                ds = FakeImageDataset(ds_path, split, source_name=ds_name)
                split_datasets.append(ds)
                print(f"  Loaded: {ds}")
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")

        if not split_datasets:
            print(f"  [WARN] No data for split '{split}'")
            continue

        combined = ConcatDataset(split_datasets) if len(split_datasets) > 1 else split_datasets[0]
        sampler  = make_balanced_sampler(combined) if split == "train" and balance_classes else None
        loaders[split] = DataLoader(
            combined, batch_size=batch_size,
            sampler=sampler, shuffle=(split == "train" and sampler is None),
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
    return loaders


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", default="./data/processed")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print("Building DataLoaders...")
    loaders = build_dataloaders(args.processed_root, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    for split, loader in loaders.items():
        imgs, labels = next(iter(loader))
        assert imgs.shape[1:] == (3, 224, 224)
        print(f"  [{split}] batches={len(loader)}, shape={tuple(imgs.shape)}, labels={labels.tolist()}")
    print("\n✓ DataLoader OK")
