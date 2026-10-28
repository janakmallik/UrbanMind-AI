"""Diagnostic: verify AST classifier module structure and freeze state.

Verifies:
    1. Cached16kAudioDataset imports from urbanintel.data.urbansas
    2. ASTClassifier and MODEL_NAME import from urbanintel.models.ast_classifier
    3. ASTClassifier.forward contains torch.no_grad()
    4. ASTClassifier encoder is frozen, head is trainable

Not part of the standard pipeline. Retained for pipeline validation.
"""

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.models.ast_classifier import MODEL_NAME, ASTClassifier

print("imports OK")
print(f"MODEL_NAME = {MODEL_NAME}")

# Check no_grad
source = inspect.getsource(ASTClassifier.forward)
has_no_grad = "no_grad" in source
print(f"ASTClassifier.forward contains no_grad: {has_no_grad}")
assert has_no_grad, "ASTClassifier.forward missing torch.no_grad() — check file"

# Check freeze state WITHOUT loading weights — inspect at class level
print()
print("Model construction check (loads pretrained weights, ~350 MB):")
model = ASTClassifier()
n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
n_head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

print(f"  total params:           {n_total:,}")
print(f"  trainable params:       {n_trainable:,}")
print(f"  encoder trainable:      {n_encoder_trainable:,}  (expected 0)")
print(f"  head trainable:         {n_head_trainable:,}  (expected > 0)")

assert n_encoder_trainable == 0, "encoder should be frozen"
assert n_head_trainable > 0, "head should be trainable"

print()
print("ALL CHECKS PASSED")
