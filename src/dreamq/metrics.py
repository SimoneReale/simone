"""Imbalance-aware metrics, tie-aware screening, group bootstrap, and plots."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)


def top_k_metrics(y: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Expected hits under uniform random ordering within a score tie at k."""
    y, scores = np.asarray(y), np.asarray(scores)
    k = min(max(1, int(k)), len(y))
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > boundary, scores == boundary
    hits = float(y[above].sum() + (k - above.sum()) * y[tied].mean())
    positives, prevalence = y.sum(), y.mean()
    return {"k": k, "precision": hits / k,
            "recall": hits / positives if positives else None,
            "EF": hits / k / prevalence if prevalence else None}


def evaluate_metrics(y: np.ndarray, scores: np.ndarray, top_k: list[int] | None = None,
                     threshold: float = 0.5) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) == 0 or len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Metrics need matching nonempty arrays and finite scores")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Metrics require binary labels")
    both = len(np.unique(y)) == 2
    predictions = (scores >= threshold).astype(int)
    result = {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
              "AP": float(average_precision_score(y, scores)) if both else None,
              "ROC_AUC": float(roc_auc_score(y, scores)) if both else None,
              "undefined_reason": None if both else "Only one activity class is represented",
              "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both else None,
              "MCC": float(matthews_corrcoef(y, predictions)) if both else None,
              "threshold": threshold, "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist()}
    for fraction in [0.01, 0.02, 0.05, 0.10]:
        stats = top_k_metrics(y, scores, int(np.ceil(fraction * len(y))))
        result[f"EF@{int(fraction * 100)}%"] = stats["EF"]
    for k in top_k or [10, 50, 100]:
        stats = top_k_metrics(y, scores, k)
        result[f"recall@{k}"] = stats["recall"]
        result[f"precision@{k}"] = stats["precision"]
    return result


def cluster_bootstrap(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                      samples: int = 1000, seed: int = 2035, top_k: list[int] | None = None) -> dict:
    """Resample whole chemical entities; skip and count single-class replicates."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    estimates = []
    for _ in range(samples):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) == 2:
            estimates.append(evaluate_metrics(y[idx], scores[idx], top_k))
    names = ["AP", "ROC_AUC", "EF@1%", "EF@2%", "EF@5%", "EF@10%"]
    names += [f"recall@{k}" for k in top_k or [10, 50, 100]]
    intervals = {name: np.quantile([row[name] for row in estimates], [0.025, 0.975]).tolist()
                 for name in names} if estimates else {}
    return {"method": "chemical-entity cluster percentile bootstrap, fixed OOF predictions",
            "requested_replicates": samples, "valid_replicates": len(estimates), "intervals_95": intervals,
            "limitations": "Conditional on fitted OOF models; excludes model-selection/training uncertainty. Only 39 positives in the full dataset; intervals may be unstable."}


def plot_predictions(y: np.ndarray, model_logits: dict[str, np.ndarray], directory: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for kind in ["pr", "roc", "enrichment"]:
        fig, ax = plt.subplots(figsize=(9, 6) if len(model_logits) > 8 else (6, 5))
        for model, logits in model_logits.items():
            scores = expit(logits)
            if len(np.unique(y)) < 2:
                continue
            if kind == "pr":
                p, r, _ = precision_recall_curve(y, scores)
                ax.plot(r, p, label=model)
                ax.set(xlabel="Recall", ylabel="Precision")
            elif kind == "roc":
                fpr, tpr, _ = roc_curve(y, scores)
                ax.plot(fpr, tpr, label=model)
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            else:
                fractions = np.linspace(0.01, 1, 100)
                enrichment = [top_k_metrics(y, scores, int(np.ceil(len(y) * f)))["EF"] for f in fractions]
                ax.plot(fractions, enrichment, label=model)
                ax.set(xlabel="Fraction screened", ylabel="Enrichment factor")
        if kind == "pr":
            ax.axhline(y.mean(), ls="--", c="gray", label=f"Prevalence {y.mean():.4f}")
        elif kind == "roc":
            ax.plot([0, 1], [0, 1], "--", c="gray")
        else:
            ax.axhline(1, ls="--", c="gray")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8) if len(model_logits) > 8 else ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{kind}.png", dpi=160)
        plt.close(fig)
