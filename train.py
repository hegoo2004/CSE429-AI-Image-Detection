"""
train.py — Training & Evaluation Pipeline
==========================================
Usage:
    python train.py                          # full run (baselines + main training)
    python train.py --baselines_only         # baselines only
    python train.py --epochs 5 --batch_size 32
    tensorboard --logdir ./runs              # monitor training
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloader import build_dataloaders
from model import LinearProbingDetector, build_model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds, zero_division=0)),
        "auc":      float(roc_auc_score(labels, probs)),
        "ap":       float(average_precision_score(labels, probs)),
    }


# ── Baselines ─────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    name = "SimpleCNN"
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,1))
    def forward(self, x): return self.head(self.features(x))


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, criterion, device, scaler=None):
    model.train()
    total_loss = n_correct = n_total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(model(imgs)) >= 0.5).long()
        n_correct += (preds.squeeze() == labels.long().squeeze()).sum().item()
        n_total   += imgs.size(0)
    return {"loss": total_loss / n_total, "accuracy": n_correct / n_total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = n_total = 0
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs_d  = imgs.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True).float().unsqueeze(1)
        logits = model(imgs_d)
        total_loss += criterion(logits, labels_d).item() * imgs.size(0)
        n_total    += imgs.size(0)
        all_labels.append(labels.numpy())
        all_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    labels_arr = np.concatenate(all_labels)
    probs_arr  = np.concatenate(all_probs)
    metrics = compute_metrics(labels_arr, probs_arr)
    metrics["loss"] = total_loss / n_total
    return metrics, labels_arr, probs_arr


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baselines(loaders, args, device):
    print("\n" + "="*50)
    print("Running Baselines")
    print("="*50)
    results = {}
    criterion = nn.BCEWithLogitsLoss()

    # Random
    print("\n[Baseline 1] Random Classifier")
    all_labels, all_probs = [], []
    for _, labels in loaders["test"]:
        all_labels.append(labels.numpy())
        all_probs.append(np.random.rand(len(labels)))
    m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
    print(f"  AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    results["RandomClassifier"] = m

    # Simple CNN
    print("\n[Baseline 2] Simple CNN")
    cnn = SimpleCNN().to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    for ep in range(args.cnn_epochs):
        tm = train_one_epoch(cnn, loaders["train"], opt, criterion, device)
        print(f"  Epoch {ep+1}/{args.cnn_epochs}  loss={tm['loss']:.4f}  acc={tm['accuracy']:.4f}")
    cm, _, _ = evaluate(cnn, loaders["test"], criterion, device)
    print(f"  Test: AUC={cm['auc']:.4f}  Acc={cm['accuracy']:.4f}  F1={cm['f1']:.4f}")
    results["SimpleCNN"] = cm

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Baseline results saved to {args.checkpoint_dir}/baseline_results.json")
    return results


# ── Main training ─────────────────────────────────────────────────────────────

def train(loaders, args, device):
    print("\n" + "="*50)
    print(f"Training LinearProbingDetector — {args.epochs} epochs")
    print("="*50)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=args.log_dir)
    criterion = nn.BCEWithLogitsLoss()

    model = LinearProbingDetector(dropout=0.2).to(device)
    print(f"  Trainable params: {model.param_count()['trainable']:,}")

    opt       = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_auc    = 0.0
    patience    = 5      # early stopping patience
    no_improve  = 0
    history     = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_one_epoch(model, loaders["train"], opt, criterion, device, scaler)
        vm, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        gap = tm["accuracy"] - vm["accuracy"]
        gap_warn = "  ⚠ overfitting" if gap > 0.10 else ""

        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss={tm['loss']:.4f}  val_loss={vm['loss']:.4f}  "
              f"train_acc={tm['accuracy']:.4f}  val_acc={vm['accuracy']:.4f}  "
              f"val_auc={vm['auc']:.4f}  "
              f"({time.time()-t0:.1f}s){gap_warn}")

        writer.add_scalars("Loss",     {"train": tm["loss"],     "val": vm["loss"]},     epoch)
        writer.add_scalars("Accuracy", {"train": tm["accuracy"], "val": vm["accuracy"]}, epoch)
        writer.add_scalar("Val/AUC",   vm["auc"],  epoch)
        writer.add_scalar("Val/F1",    vm["f1"],   epoch)
        writer.add_scalar("LR",        scheduler.get_last_lr()[0], epoch)
        history["train"].append(tm); history["val"].append(vm)

        if vm["auc"] > best_auc:
            best_auc   = vm["auc"]
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_metrics": vm},
                       ckpt_dir / "best.pt")
            print(f"  ✓ Best AUC: {best_auc:.4f} — checkpoint saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch} — best AUC: {best_auc:.4f}")
                break

    writer.close()

    # Final test evaluation
    print(f"\n{'='*50}\nFinal Test Evaluation\n{'='*50}")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m, _, _ = evaluate(model, loaders["test"], criterion, device)
    print(f"  Accuracy : {test_m['accuracy']:.4f}")
    print(f"  AUC      : {test_m['auc']:.4f}")
    print(f"  F1       : {test_m['f1']:.4f}")
    print(f"  AP       : {test_m['ap']:.4f}")

    results = {"model": "LinearProbingDetector", "test": test_m, "history": history}
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {ckpt_dir}/results.json")
    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def print_table(baseline_path, main_results):
    with open(baseline_path) as f:
        baselines = json.load(f)
    all_r = {**baselines, "LinearProbingDetector (ours)": main_results.get("test", {})}
    print("\n" + "="*72)
    print(f"{'Model':<35} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'AP':>8}")
    print("-"*72)
    for name, m in all_r.items():
        print(f"{name:<35} {m.get('auc',0):.4f}   {m.get('accuracy',0):.4f}"
              f"    {m.get('f1',0):.4f}  {m.get('ap',0):.4f}")
    print("="*72)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root",  default="./data/processed")
    parser.add_argument("--checkpoint_dir",  default="./checkpoints")
    parser.add_argument("--log_dir",         default="./runs/clip_lp")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--cnn_epochs",  type=int,   default=5)
    parser.add_argument("--baselines_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_dataloaders(
        processed_root=args.processed_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.baselines_only:
        run_baselines(loaders, args, device)
    else:
        baseline_results = run_baselines(loaders, args, device)
        main_results     = train(loaders, args, device)
        print_table(str(Path(args.checkpoint_dir) / "baseline_results.json"), main_results)
