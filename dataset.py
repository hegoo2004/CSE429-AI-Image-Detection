"""
dataset.py — Download and preprocess all datasets
==================================================
Run:  python dataset.py --data_root ./data --out_root ./data/processed
"""

import argparse, json, os, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_SIZE   = (224, 224)
MEAN         = (0.485, 0.456, 0.406)
STD          = (0.229, 0.224, 0.225)
VALID_EXT    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42


# ── Download helpers ──────────────────────────────────────────────────────────

def download_cifake(data_root: Path):
    print("\n[1/3] CIFAKE")
    dest = data_root / "cifake"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            "bird-jordan/cifake-real-and-ai-generated-synthetic-images",
            path=str(dest), unzip=True
        )
        print("  ✓ CIFAKE downloaded")
    except Exception as e:
        print(f"  ✗ CIFAKE: {e}")
        print("  → Install kaggle CLI: pip install kaggle")
        print("  → Place API key at: ~/.kaggle/kaggle.json")
        print("  → Then re-run: python dataset.py --download_only")


def download_faceforensics(data_root: Path):
    print("\n[2/3] FaceForensics++")
    dest = data_root / "faceforensics"
    dest.mkdir(parents=True, exist_ok=True)
    script = dest / "download_FaceForensics.py"
    if not script.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download_FaceForensics.py",
            str(script)
        )
    print("  FaceForensics++ requires manual access request.")
    print("  Request at: https://github.com/ondyari/FaceForensics#access")
    print(f"  Then run:  python {script} {dest} -d all -c c23 -t videos --server EU2")


def download_genimage(data_root: Path):
    print("\n[3/3] GenImage (ProGAN + SDv1.4 subsets)")
    dest = data_root / "genimage"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        for gen in ["imagenet_progan", "stable_diffusion_v_1_4"]:
            out = dest / gen
            out.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading {gen}...")
            snapshot_download(
                repo_id="HKUST-LongGroup/GenImage",
                repo_type="dataset",
                allow_patterns=[f"{gen}/*"],
                local_dir=str(out),
                max_workers=4,
            )
            print(f"  ✓ {gen}")
    except Exception as e:
        print(f"  ✗ GenImage: {e}")
        print("  → pip install huggingface_hub")


# ── Sample collectors ─────────────────────────────────────────────────────────

def collect_cifake(root: Path) -> List[Tuple[Path, int]]:
    samples = []
    base = root / "cifake"
    for split in ("train", "test"):
        for name, label in (("REAL", 0), ("FAKE", 1)):
            folder = base / split / name
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in VALID_EXT:
                    samples.append((p, label))
    print(f"  CIFAKE:         {len(samples):>8,} images")
    return samples


def collect_faceforensics(root: Path) -> List[Tuple[Path, int]]:
    samples = []
    base = root / "faceforensics"
    real = base / "original_sequences" / "youtube" / "c23" / "frames"
    if real.exists():
        for p in real.rglob("*.png"):
            samples.append((p, 0))
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        fake = base / "manipulated_sequences" / method / "c23" / "frames"
        if fake.exists():
            for p in fake.rglob("*.png"):
                samples.append((p, 1))
    print(f"  FaceForensics++:{len(samples):>8,} images")
    return samples


def collect_fakeface(root: Path) -> List[Tuple[Path, int]]:
    """
    Real and Fake Face Detection dataset.
    Looks in both real_and_fake_face and real_and_fake_face_detection subfolders.
    """
    samples = []
    base = root / "fakeface"
    for subfolder in ("real_and_fake_face", "real_and_fake_face_detection/real_and_fake_face"):
        for label_name, label in (("training_real", 0), ("training_fake", 1)):
            folder = base / subfolder / label_name
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.suffix.lower() in VALID_EXT:
                    samples.append((p, label))
    print(f"  FakeFace:       {len(samples):>8,} images")
    return samples


def collect_genimage(root: Path) -> List[Tuple[Path, int]]:
    samples = []
    base = root / "genimage"
    if not base.exists():
        print(f"  GenImage:              0 images (folder missing)")
        return samples
    for gen_dir in base.iterdir():
        if not gen_dir.is_dir():
            continue
        for split_dir in gen_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for class_dir in split_dir.iterdir():
                label = 1 if class_dir.name.lower() in ("ai", "fake") else 0
                for p in class_dir.rglob("*"):
                    if p.suffix.lower() in VALID_EXT:
                        samples.append((p, label))
    print(f"  GenImage:       {len(samples):>8,} images")
    return samples


# ── Core preprocessing ────────────────────────────────────────────────────────

def preprocess_image(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="PNG", optimize=True)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def process_dataset(samples: List[Tuple[Path, int]], name: str, out_root: Path) -> Dict:
    print(f"\n  Processing {name} ...")
    labels = [l for _, l in samples]
    train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=RANDOM_SEED)
    tv_labels = [l for _, l in train_val]
    train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=tv_labels, random_state=RANDOM_SEED)
    splits = {"train": train, "val": val, "test": test}
    stats = {"dataset": name, "splits": {}}
    skipped = 0
    for split_name, subset in splits.items():
        nr = nf = ok = 0
        for src, label in tqdm(subset, desc=f"    {split_name}", leave=False):
            ldir = "real" if label == 0 else "fake"
            dst = out_root / name / split_name / ldir / src.name
            if dst.exists():
                dst = dst.with_stem(f"{src.stem}_{random.randint(0,99999):05d}")
            if preprocess_image(src, dst):
                ok += 1
                if label == 0: nr += 1
                else: nf += 1
            else:
                skipped += 1
        stats["splits"][split_name] = {"total": ok, "real": nr, "fake": nf}
        print(f"    {name}/{split_name}: {ok:,} (real={nr:,}, fake={nf:,})")
    stats["skipped"] = skipped
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="./data")
    parser.add_argument("--out_root",     default="./data/processed")
    parser.add_argument("--datasets",     nargs="+", default=["cifake", "faceforensics", "genimage"])
    parser.add_argument("--download",     action="store_true", help="Download datasets first")
    parser.add_argument("--download_only",action="store_true", help="Download only, no preprocessing")
    parser.add_argument("--max_per_class",type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if args.download or args.download_only:
        print("=" * 50)
        print("Downloading datasets...")
        print("=" * 50)
        if "cifake"        in args.datasets: download_cifake(data_root)
        if "faceforensics" in args.datasets: download_faceforensics(data_root)
        if "genimage"      in args.datasets: download_genimage(data_root)
        if args.download_only:
            return

    collectors = {"cifake": collect_cifake, "faceforensics": collect_faceforensics, "fakeface": collect_fakeface, "genimage": collect_genimage}
    all_stats  = []

    print("\n" + "=" * 50)
    print("Collecting image paths...")
    for name in args.datasets:
        if name not in collectors:
            continue
        samples = collectors[name](data_root)
        if not samples:
            print(f"  [SKIP] No images for {name}")
            continue
        if args.max_per_class:
            real = [(p, l) for p, l in samples if l == 0][:args.max_per_class]
            fake = [(p, l) for p, l in samples if l == 1][:args.max_per_class]
            samples = real + fake
        stats = process_dataset(samples, name, out_root)
        all_stats.append(stats)

    stats_path = out_root / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Dataset':<18} {'Split':<8} {'Total':>8} {'Real':>8} {'Fake':>8}")
    print("-" * 60)
    for ds in all_stats:
        for split, c in ds["splits"].items():
            print(f"{ds['dataset']:<18} {split:<8} {c['total']:>8,} {c['real']:>8,} {c['fake']:>8,}")
    print("=" * 60)
    print(f"\n✓ Preprocessing complete. Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
