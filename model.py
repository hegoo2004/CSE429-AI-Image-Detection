"""
model.py — UniversalFakeDetect re-implementation (Ojha et al., CVPR 2023)
=========================================================================
Usage:
    from model import build_model
    model = build_model("linear")   # LinearProbingDetector (trainable)
    model = build_model("nn")       # NearestNeighbourDetector (no training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPBackbone(nn.Module):
    """Frozen CLIP-ViT-L/14 visual encoder. Output: [B, 768] L2-normalised."""
    EMBED_DIM = 768

    def __init__(self):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError("pip install open-clip-torch")
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.visual(x)
        return F.normalize(feats, dim=-1)


class LinearProbingDetector(nn.Module):
    """
    Frozen CLIP backbone + single trainable Linear(768→1).
    Only 769 parameters are trained.
    """
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.backbone = CLIPBackbone()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 1, bias=True),
        )
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))          # [B, 1] logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)).squeeze(1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self):
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_params())
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class NearestNeighbourDetector(nn.Module):
    """
    Frozen CLIP backbone + cosine k-NN.
    No training needed — call build_feature_bank() before inference.
    """
    def __init__(self):
        super().__init__()
        self.backbone = CLIPBackbone()
        self.register_buffer("real_bank", torch.empty(0))
        self.register_buffer("fake_bank", torch.empty(0))
        self._bank_built = False

    @torch.no_grad()
    def build_feature_bank(self, real_loader, fake_loader, device, verbose=True):
        self.backbone.to(device)
        def extract(loader, name):
            feats = []
            for i, (imgs, _) in enumerate(loader):
                feats.append(self.backbone(imgs.to(device)).cpu())
                if verbose and i % 20 == 0:
                    print(f"  {name}: {i}/{len(loader)}", end="\r")
            print()
            return torch.cat(feats)
        if verbose: print("Building feature bank...")
        self.real_bank = extract(real_loader, "real").to(device)
        self.fake_bank = extract(fake_loader, "fake").to(device)
        self._bank_built = True
        if verbose:
            print(f"  Real: {self.real_bank.shape[0]:,}  Fake: {self.fake_bank.shape[0]:,}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._bank_built:
            raise RuntimeError("Call build_feature_bank() first.")
        q = self.backbone(x)
        dist_real = 1 - (q @ self.real_bank.T).max(dim=1).values
        dist_fake = 1 - (q @ self.fake_bank.T).max(dim=1).values
        return dist_real - dist_fake

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (self.forward(x) > 0).long()


def build_model(variant: str = "linear", **kwargs) -> nn.Module:
    if variant == "linear": return LinearProbingDetector(**kwargs)
    if variant == "nn":     return NearestNeighbourDetector(**kwargs)
    raise ValueError(f"Unknown variant '{variant}'. Use 'linear' or 'nn'.")
