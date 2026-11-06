"""Training-only OOF selection of a scalar logit fusion coefficient."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score


def fuse_logits(z_a: np.ndarray, z_b: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Fusion alpha must lie in [0, 1]")
    return (1 - alpha) * z_a + alpha * z_b


def select_alpha(y: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray, candidates: list[float]):
    if not candidates:
        raise ValueError("Fusion alpha candidate list is empty")
    history = [{"alpha": float(a), "inner_oof_AP": float(average_precision_score(y, fuse_logits(oof_a, oof_b, a)))}
               for a in candidates]
    selected = max(history, key=lambda row: row["inner_oof_AP"])
    return selected["alpha"], history


def fused_score(z_a: np.ndarray, z_b: np.ndarray, alpha: float) -> np.ndarray:
    return expit(fuse_logits(z_a, z_b, alpha))
