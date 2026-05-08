"""
predict.py — Test any image: is it REAL or FAKE?
=================================================
Single image:
    python predict.py --image path/to/image.jpg

Folder of images:
    python predict.py --folder path/to/folder/

With confidence bar:
    python predict.py --image path/to/image.jpg --show
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import LinearProbingDetector

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(checkpoint: str, device: torch.device):
    model = LinearProbingDetector().to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: Path, device: torch.device):
    img   = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    prob  = model.predict_proba(tensor).item()
    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, prob, confidence


def confidence_bar(prob, width=40):
    """Visual bar showing real vs fake probability."""
    real_pct = (1 - prob)
    fake_pct = prob
    real_fill = int(real_pct * width)
    fake_fill = int(fake_pct * width)
    bar = "█" * real_fill + "░" * fake_fill
    return f"REAL |{bar}| FAKE   ({real_pct*100:.1f}% real  /  {fake_pct*100:.1f}% fake)"


def print_result(image_path, label, prob, confidence, show_bar=True):
    icon = "🟢" if label == "REAL" else "🔴"
    print(f"\n{'─'*55}")
    print(f"  Image : {Path(image_path).name}")
    print(f"  Result: {icon}  {label}  (confidence: {confidence*100:.1f}%)")
    print(f"  Score : P(fake) = {prob:.4f}")
    if show_bar:
        print(f"  {confidence_bar(prob)}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real or fake for any image")
    parser.add_argument("--image",      type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder",     type=str, default=None, help="Path to a folder of images")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--show",       action="store_true", help="Show confidence bar")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Usage:")
        print("  Single image:  python predict.py --image path/to/image.jpg")
        print("  Folder:        python predict.py --folder path/to/folder/")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint} ...")
    model  = load_model(args.checkpoint, device)
    print("✓ Model loaded\n")

    # ── Single image ──────────────────────────────────────────────────────────
    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"✗ File not found: {path}")
            exit(1)
        label, prob, conf = predict_image(model, path, device)
        print_result(path, label, prob, conf, show_bar=True)

    # ── Folder of images ──────────────────────────────────────────────────────
    elif args.folder:
        folder = Path(args.folder)
        images = [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXT]
        if not images:
            print(f"✗ No images found in {folder}")
            exit(1)

        print(f"Found {len(images)} images in {folder}\n")
        results = {"REAL": 0, "FAKE": 0}

        for img_path in sorted(images):
            label, prob, conf = predict_image(model, img_path, device)
            print_result(img_path, label, prob, conf, show_bar=True)
            results[label] += 1

        # Summary
        total = len(images)
        print(f"\n{'='*55}")
        print(f"  SUMMARY: {total} images processed")
        print(f"  🟢 REAL : {results['REAL']} ({results['REAL']/total*100:.1f}%)")
        print(f"  🔴 FAKE : {results['FAKE']} ({results['FAKE']/total*100:.1f}%)")
        print(f"{'='*55}")
