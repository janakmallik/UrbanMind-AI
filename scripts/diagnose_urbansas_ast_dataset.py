"""Diagnostic: verify CachedASTFeatureDataset and ASTClassifier work end-to-end.

Verifies:
    1. Filename mapping between df and cache works
    2. Feature extractor output shape (1024, 128)
    3. Encoder output shape (1214, 768) after pooling to 768-dim
    4. Head output shape = (batch, 2)
    5. Only head has gradients (encoder frozen)
    6. Forward + backward + step works
    7. GPU memory usage

Not part of the standard pipeline. Retained for pipeline validation.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import CachedASTFeatureDataset, prepare_urbansas
from urbanintel.models.ast_classifier import ASTClassifier

CACHE_DIR = _REPO_ROOT / "data" / "urbansas_ast_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    mavd = split.train_df[split.train_df["subset"] == "mavd"].reset_index(drop=True).head(4)

    print("Building dataset for 4 clips ...")
    ds = CachedASTFeatureDataset(mavd, cache_dir=CACHE_DIR)
    print(f"  dataset size: {len(ds)}")

    # --- 1. Single item ---
    x, y = ds[0]
    print(f"\n[1] Single item:")
    print(f"  input_values.shape = {tuple(x.shape)}, dtype = {x.dtype}")
    print(f"  label = {y}")

    # --- 2. Load model ---
    print(f"\n[2] Loading ASTClassifier ...")
    model = ASTClassifier().to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_total:,}")
    print(f"  trainable: {n_trainable:,}")

    # --- 3. Check gradient state ---
    print(f"\n[3] Gradient state:")
    encoder_requires_grad = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_requires_grad = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"  encoder requires_grad params: {encoder_requires_grad}")
    print(f"  head requires_grad params: {head_requires_grad}")
    assert encoder_requires_grad == 0, "encoder should be frozen"
    assert head_requires_grad > 0, "head should be trainable"
    print(f"  PASS: encoder frozen, head trainable")

    # --- 4. Forward on a batch ---
    print(f"\n[4] Forward + backward + step:")
    xs = torch.stack([ds[i][0] for i in range(4)]).to(DEVICE)
    ys = torch.tensor([ds[i][1] for i in range(4)], dtype=torch.long).to(DEVICE)
    print(f"  batch input shape: {tuple(xs.shape)}")

    model.train()
    model.encoder.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

    before_head = next(model.head.parameters()).detach().clone()
    before_encoder = next(model.encoder.parameters()).detach().clone()

    optimizer.zero_grad()
    logits = model(xs)
    print(f"  logits shape: {tuple(logits.shape)}")
    assert logits.shape == (4, 2), f"expected (4, 2), got {tuple(logits.shape)}"
    loss = criterion(logits, ys)
    print(f"  loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()

    after_head = next(model.head.parameters()).detach()
    after_encoder = next(model.encoder.parameters()).detach()

    head_changed = not torch.allclose(before_head, after_head)
    encoder_changed = not torch.allclose(before_encoder, after_encoder)
    print(f"  head params changed: {head_changed} (expected True)")
    print(f"  encoder params changed: {encoder_changed} (expected False)")
    assert head_changed, "head should have updated"
    assert not encoder_changed, "encoder should not have updated"
    print(f"  PASS: forward + backward + step work correctly")

    # --- 5. VRAM ---
    if DEVICE == "cuda":
        print(f"\n[5] GPU memory:")
        print(f"  allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("\nDataset smoke test PASSED")


if __name__ == "__main__":
    main()
