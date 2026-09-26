"""Masked BCE with hard-negative mining plus masked dice, DBNet's probability-map loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEGATIVE_RATIO = 3.0  # hardest background pixels kept per text pixel


def text_map_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    positive = (target * mask).flatten()
    negative = ((1 - target) * mask).flatten()
    bce = bce.flatten()
    n_pos = int(positive.sum().item())
    n_neg = min(int(negative.sum().item()), int(max(n_pos, 1) * NEGATIVE_RATIO))
    # max(n_pos, 1), so a text-free crop still pushes its background down.
    pos_loss = (bce * positive).sum()
    neg_loss = torch.topk(bce * negative, n_neg).values.sum() if n_neg else bce.new_zeros(())
    mined = (pos_loss + neg_loss) / max(1, n_pos + n_neg)

    prob = torch.sigmoid(logits).flatten()
    m = mask.flatten()
    t = target.flatten()
    intersection = (prob * t * m).sum()
    dice = 1 - (2 * intersection + 1) / ((prob * m).sum() + (t * m).sum() + 1)

    total = mined + dice
    return {"loss": total, "bce": mined.detach(), "dice": dice.detach()}
