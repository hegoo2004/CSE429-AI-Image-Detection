"""
visualise_results.py — Plot training results, confusion matrix, ROC curve
=========================================================================
Run AFTER training is complete:
    python visualise_results.py

Saves all plots to: ./results_plots/
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)

from dataloader import build_dataloaders
from model import LinearProbingDetector

OUT_DIR = Path("./results_plots")
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Loss & Accuracy curves ─────────────────────────────────────────────────

def plot_training_curves(results_path="./checkpoints/results.json"):
    with open(results_path) as f:
        results = json.load(f)

    train_loss = [e["loss"]     for e in results["history"]["train"]]
    val_loss   = [e["loss"]     for e in results["history"]["val"]]
    train_acc  = [e["accuracy"] for e in results["history"]["train"]]
    val_acc    = [e["accuracy"] for e in results["history"]["val"]]
    epochs     = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss,   "r-o", label="Val Loss",   linewidth=2)
    ax1.set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, "b-o", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, val_acc,   "r-o", label="Val Accuracy",   linewidth=2)
    ax2.set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves — LinearProbingDetector", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ── 2. Confusion Matrix + ROC + PR curves ────────────────────────────────────

@torch.no_grad()
def get_predictions(processed_root, checkpoint, device):
    loaders = build_dataloaders(
        processed_root=processed_root,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )

    model = LinearProbingDetector().to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_labels, all_probs = [], []
    for imgs, labels in loaders["test"]:
        probs = model.predict_proba(imgs.to(device)).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_confusion_matrix(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")

    # Annotate TP/TN/FP/FN
    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(f"Predicted Label\n\nTN={tn}  FP={fp}  FN={fn}  TP={tp}", fontsize=10)

    path = OUT_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2E75B6", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E75B6")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    path = OUT_DIR / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#ED7D31", lw=2)
    ax.fill_between(recall, precision, alpha=0.1, color="#ED7D31")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    path = OUT_DIR / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_score_distribution(labels, probs):
    """Show how fake and real scores are distributed."""
    real_scores = probs[labels == 0]
    fake_scores = probs[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_scores, bins=40, alpha=0.6, color="#70AD47", label="Real images", density=True)
    ax.hist(fake_scores, bins=40, alpha=0.6, color="#FF0000", label="Fake images", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision threshold (0.5)")
    ax.set_xlabel("P(fake) — Model Score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution — Real vs Fake (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    path = OUT_DIR / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", default="./data/processed")
    parser.add_argument("--checkpoint",     default="./checkpoints/best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nGenerating plots → {OUT_DIR}/\n")

    print("[1/5] Training curves...")
    plot_training_curves()

    print("[2/5] Getting test predictions...")
    labels, probs = get_predictions(args.processed_root, args.checkpoint, device)

    print("[3/5] Confusion matrix...")
    plot_confusion_matrix(labels, probs)

    print("[4/5] ROC curve...")
    plot_roc_curve(labels, probs)

    print("[5/5] PR curve + score distribution...")
    plot_pr_curve(labels, probs)
    plot_score_distribution(labels, probs)

    print(f"\n✓ All plots saved to {OUT_DIR}/")
    print("  Open the folder to view them.")
