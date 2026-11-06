"""Small training-only rank ensembles; score scales fitted to inner OOF logits."""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score


class EmpiricalRank:
    def fit(self, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Rank reference must be a finite nonempty vector")
        self.reference_ = np.sort(values)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Rank queries must be finite one-dimensional vectors")
        left = np.searchsorted(self.reference_, values, side="left")
        right = np.searchsorted(self.reference_, values, side="right")
        return (left + right + 1) / (2 * (len(self.reference_) + 1))


class LogitFusion:
    def __init__(self, model_a, model_b, alpha: float):
        self.model_a, self.model_b, self.alpha = model_a, model_b, float(alpha)
        self.training_structure_ids_ = model_a.training_structure_ids_

    def decision_function(self, inputs):
        return (1 - self.alpha) * self.model_a.decision_function(inputs) + self.alpha * self.model_b.decision_function(inputs)

    def predict_score(self, inputs):
        return expit(self.decision_function(inputs))


class RankEnsemble:
    def __init__(self, models: dict, normalizers: dict, weights: dict):
        self.models, self.normalizers, self.weights = models, normalizers, weights
        self.training_structure_ids_ = next(iter(models.values())).training_structure_ids_

    def predict_score(self, inputs):
        return sum(self.weights[name] * self.normalizers[name].transform(model.decision_function(inputs))
                   for name, model in self.models.items())

    def decision_function(self, inputs):
        return logit(np.clip(self.predict_score(inputs), 1e-7, 1 - 1e-7))


def build_rank_ensemble(models: dict, inner_logits: dict, y: np.ndarray, members: list[str],
                        selected: bool = False, alphas: list[float] | None = None,
                        max_correlation: float = 1.0):
    """Selection statistics belong exclusively to the caller's training partition."""
    if not members or len(set(members)) != len(members):
        raise ValueError("Ensemble members must be a nonempty unique list")
    missing = set(members) - set(models)
    if missing:
        raise ValueError(f"Ensemble members not fitted: {sorted(missing)}")
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Ensemble selection requires both binary activity classes")
    if set(members) - set(inner_logits):
        raise ValueError("Missing training-only inner OOF logits for an ensemble member")
    if any(np.asarray(inner_logits[name]).shape != y.shape or not np.isfinite(inner_logits[name]).all() for name in members):
        raise ValueError("Inner OOF logits must be finite and match target row counts")
    if not 0 <= max_correlation <= 1:
        raise ValueError("max_correlation must lie in [0,1]")
    normalizers = {name: EmpiricalRank().fit(inner_logits[name]) for name in members}
    ranked = {name: normalizers[name].transform(inner_logits[name]) for name in members}
    candidates = [{name: 1 / len(members) for name in members}]
    if selected:
        candidates = [{name: 1.0} for name in members]
        for a, b in combinations(members, 2):
            correlation = np.corrcoef(ranked[a], ranked[b])[0, 1]
            if np.isfinite(correlation) and correlation > max_correlation:
                continue
            for alpha in alphas or [0.25, 0.5, 0.75]:
                if not 0 <= alpha <= 1:
                    raise ValueError("Ensemble coefficient must lie in [0,1]")
                if 0 < alpha < 1:
                    candidates.append({a: 1 - alpha, b: alpha})
    best, best_value, best_scores, history = None, -np.inf, None, []
    for weights in candidates:
        scores = sum(weight * ranked[name] for name, weight in weights.items())
        value = float(average_precision_score(y, scores))
        history.append({"weights": weights, "inner_oof_AP": value})
        if value > best_value:
            best, best_value, best_scores = weights, value, scores
    ensemble = RankEnsemble({name: models[name] for name in best},
                            {name: normalizers[name] for name in best}, best)
    return ensemble, logit(np.clip(best_scores, 1e-7, 1 - 1e-7)), {
        "selected": best, "search": history,
        "normalization": "empirical midpoint CDF fitted to training-only inner OOF logits",
        "calibrated": False,
    }
