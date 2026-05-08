"""
extension.py — Zero-Shot Evaluation on Unseen Generators
=========================================================
Tests the trained model on DALL-E 3, SDXL, Midjourney v6 (no fine-tuning).

Setup:
    Place ~200 images per generator in:
        extension_data/dalle3/fake/
        extension_data/sdxl/fake/
        extension_data/midjourney_v6/fake/

    Place real images in:
        data/processed/cifake/test/real/   (already done by dataset.py)

Run:
    python extension.py --checkpoint ./checkpoints/best.pt

See collection guide:
    python extension.py --show_guide
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from model import LinearProbingDetector

GENERATORS = ["dalle3", "sdxl", "midjourney_v6"]

COLLECTION_GUIDE = """
HOW TO COLLECT EXTENSION IMAGES
=================================

DALL-E 3
  → https://labs.openai.com  (OpenAI account required)
  → Generate ~200 varied images, save to: extension_data/dalle3/fake/

SDXL (Stable Diffusion XL)
  → pip install diffusers accelerate
  → python extension.py --generate_sdxl    (auto-generates 200 images)
  → Or use https://clipdrop.co/stable-diffusion
  → Save to: extension_data/sdxl/fake/

Midjourney v6
  → Discord bot: https://www.midjourney.com
  → Generate ~200 images with /imagine, download, save to:
  → extension_data/midjourney_v6/fake/

Real images (reuse from preprocessing):
  → Already at: data/processed/cifake/test/real/
"""

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class FlatImageDataset(Dataset):
    def __init__(self, folder: Path, label: int = 1):
        self.paths = sorted([
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])
        self.label = label
        if not self.paths:
            raise FileNotFoundError(f"No images in {folder}")

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return TRANSFORM(Image.open(self.paths[idx]).convert("RGB")), self.label


@torch.no_grad()
def evaluate_generator(model, fake_folder, real_folder, device, batch_size=32, name=""):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    fake_ds = FlatImageDataset(fake_folder, label=1)
    real_ds = FlatImageDataset(real_folder, label=0)
    loader  = DataLoader(ConcatDataset([real_ds, fake_ds]), batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in tqdm(loader, desc=f"  {name}", leave=False):
        all_probs.append(model.predict_proba(imgs.to(device)).cpu().numpy())
        all_labels.append(labels.numpy())
    labels_arr = np.concatenate(all_labels)
    probs_arr  = np.concatenate(all_probs)
    preds_arr  = (probs_arr >= 0.5).astype(int)
    return {
        "generator":    name,
        "n_real":       int((labels_arr == 0).sum()),
        "n_fake":       int((labels_arr == 1).sum()),
        "accuracy":     float(accuracy_score(labels_arr, preds_arr)),
        "f1":           float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "auc":          float(roc_auc_score(labels_arr, probs_arr)),
        "ap":           float(average_precision_score(labels_arr, probs_arr)),
        "fake_recall":  float(((preds_arr == 1) & (labels_arr == 1)).sum() / max((labels_arr == 1).sum(), 1)),
    }


def generate_sdxl(out_dir: Path, n: int = 200):
    """Auto-generate SDXL images if diffusers is installed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers import StableDiffusionXLPipeline
        import torch as _torch
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=_torch.float16
        ).to("cuda" if _torch.cuda.is_available() else "cpu")

        prompts = [
            "a photorealistic portrait of a young person outdoors",
            "a detailed landscape photograph at golden hour",
            "professional product photo on white background",
            "macro photograph of a flower with morning dew",
            "a busy city street at night with neon lights",
        ] * (n // 5 + 1)

        for i, prompt in enumerate(prompts[:n]):
            img = pipe(prompt, num_inference_steps=30).images[0]
            img.save(out_dir / f"sdxl_{i:04d}.png")
            if i % 10 == 0:
                print(f"  Generated {i+1}/{n}", end="\r")
        print(f"\n✓ SDXL images saved to {out_dir}")
    except Exception as e:
        print(f"  ✗ SDXL generation failed: {e}")
        print("  → pip install diffusers accelerate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=False, default="./checkpoints/best.pt")
    parser.add_argument("--ext_root",     default="./extension_data")
    parser.add_argument("--real_root",    default="./data/processed/cifake/test/real")
    parser.add_argument("--out_dir",      default="./extension_results")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--show_guide",   action="store_true")
    parser.add_argument("--generate_sdxl",action="store_true")
    args = parser.parse_args()

    if args.show_guide:
        print(COLLECTION_GUIDE); sys.exit(0)

    if args.generate_sdxl:
        generate_sdxl(Path(args.ext_root) / "sdxl" / "fake"); sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LinearProbingDetector().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"✓ Loaded checkpoint (epoch {ckpt.get('epoch','?')})")

    real_folder = Path(args.real_root)
    if not real_folder.exists():
        print(f"✗ Real folder not found: {real_folder}")
        print("  Run: python dataset.py first")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Zero-Shot Evaluation — Unseen Generators")
    print(f"{'='*60}")

    results = []
    for gen in GENERATORS:
        fake_folder = Path(args.ext_root) / gen / "fake"
        if not fake_folder.exists():
            print(f"\n  [SKIP] {gen}: no images at {fake_folder}")
            print(f"         Run: python extension.py --show_guide")
            results.append({"generator": gen, "status": "not_collected"})
            continue
        try:
            r = evaluate_generator(model, fake_folder, real_folder, device, args.batch_size, gen)
            r["status"] = "evaluated"
            results.append(r)
            print(f"  {gen:<22}  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  FakeRecall={r['fake_recall']:.4f}")
        except Exception as e:
            print(f"  ✗ {gen}: {e}")
            results.append({"generator": gen, "status": "error", "error": str(e)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"{'Generator':<22} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'FakeRecall':>12}")
    print("-"*65)
    for r in results:
        if r.get("status") == "evaluated":
            print(f"{r['generator']:<22} {r['auc']:>8.4f} {r['accuracy']:>10.4f} "
                  f"{r['f1']:>8.4f} {r['fake_recall']:>12.4f}")
        else:
            print(f"{r['generator']:<22} {'—':>8} {'(' + r.get('status','?') + ')':>10}")
    print("="*65)
    print(f"\n✓ Results saved to {out_dir}/zero_shot_results.json")
