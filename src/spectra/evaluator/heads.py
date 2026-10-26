"""Heads of the automated evaluator on frozen RAD-DINO tokens.

One attentive-pooling head per task: a learned query attends over the CLS and grid tokens, and a two-layer MLP maps
the pooled vector to the task's classes. Tasks from different sources train together with a masked loss (a film
contributes only to the tasks its source labels). Finding tasks are ordinal grades (readout phi_c = expected grade);
device tasks are presence (readout psi_d = P(present)).
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveHead(nn.Module):
    def __init__(self, dim: int, n_out: int, n_heads: int = 8, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        pooled, _ = self.attn(self.query.expand(x.shape[0], -1, -1), x, x, need_weights=False)
        return self.mlp(pooled[:, 0])


class Heads(nn.Module):
    def __init__(self, tasks: Mapping[str, int], dim: int = 768, **kw):
        super().__init__()
        self.tasks = dict(tasks)
        self.heads = nn.ModuleDict({t: AttentiveHead(dim, n, **kw) for t, n in self.tasks.items()})

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {t: h(tokens) for t, h in self.heads.items()}


def masked_loss(logits: Mapping[str, torch.Tensor], labels: torch.Tensor, mask: torch.Tensor,
                order) -> torch.Tensor:
    """Mean cross-entropy over the labelled (film, task) entries; labels/mask are [B, T] in `order`."""
    total = labels.new_zeros((), dtype=torch.float32)
    n = 0
    for i, t in enumerate(order):
        m = mask[:, i]
        if m.any():
            total = total + F.cross_entropy(logits[t][m].float(), labels[m, i], reduction="sum")
            n += int(m.sum())
    return total / max(n, 1)


def expected_grade(logits: torch.Tensor) -> torch.Tensor:
    p = logits.float().softmax(-1)
    return (p * torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)).sum(-1)


def readouts(logits: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """phi_c (expected grade) for ordinal tasks, psi_d = P(present) for two-class tasks."""
    return {t: (l.float().softmax(-1)[:, 1] if l.shape[-1] == 2 else expected_grade(l)) for t, l in logits.items()}
