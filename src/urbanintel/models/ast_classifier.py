"""AST classifier with frozen encoder for small-dataset transfer learning.

The AudioSet-pretrained encoder is frozen. Only the linear head is trainable.
The forward pass wraps the encoder in torch.no_grad() to avoid autograd
overhead on the 86M frozen parameters, while preserving gradients on the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTModel

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTClassifier(nn.Module):
    """Frozen AST encoder + 2-class linear head."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Linear(768, 2)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(input_values=input_values)
            pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)
