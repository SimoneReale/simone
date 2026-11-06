"""L2 logistic branches and a training-label-only Tanimoto reference."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .features import FeatureBundle, descriptor_transformer


class LogitClassifier:
    def __init__(self, branch: str = "M1", C: float = 0.1, positive_weight: float = 1,
                 seed: int = 2035, max_iter: int = 3000):
        if branch not in {"M0", "M1", "M2"}:
            raise ValueError(f"Unknown classifier {branch}")
        self.branch, self.C, self.positive_weight = branch, C, positive_weight
        self.seed, self.max_iter = seed, max_iter

    def _matrix(self, features: FeatureBundle, embeddings: np.ndarray | None, fit: bool):
        if self.branch == "M0":
            return features.ecfp.astype(np.float64)
        if fit:
            self.descriptor_transformer_ = descriptor_transformer().fit(features.descriptors)
        desc = self.descriptor_transformer_.transform(features.descriptors)
        if self.branch == "M1":
            return sparse.hstack([features.ecfp, sparse.csr_matrix(desc)], format="csr", dtype=np.float64)
        if embeddings is None or len(embeddings) != len(desc):
            raise ValueError("Branch M2 requires matching frozen molecular embeddings")
        if fit:
            self.embedding_scaler_ = StandardScaler().fit(embeddings)
        return np.column_stack([self.embedding_scaler_.transform(embeddings), desc])

    def fit(self, features: FeatureBundle, y: np.ndarray, embeddings: np.ndarray | None = None):
        if len(np.unique(y)) != 2:
            raise ValueError("Supervised training requires both activity classes")
        matrix = self._matrix(features, embeddings, fit=True)
        self.classifier_ = LogisticRegression(C=self.C, class_weight={0: 1, 1: self.positive_weight},
                                              solver="liblinear", max_iter=self.max_iter,
                                              random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier_.fit(matrix, y)
        self.feature_signature_ = features.signature
        self.training_structure_ids_ = features.structure_ids.copy()
        return self

    def decision_function(self, features: FeatureBundle, embeddings: np.ndarray | None = None) -> np.ndarray:
        if features.signature != self.feature_signature_:
            raise ValueError("Training and inference feature configurations differ")
        return self.classifier_.decision_function(self._matrix(features, embeddings, fit=False))

    def predict_score(self, features: FeatureBundle, embeddings: np.ndarray | None = None) -> np.ndarray:
        """Ranking sigmoid; class weighting does not establish calibration."""
        return expit(self.decision_function(features, embeddings))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> LogitClassifier:
        return joblib.load(path)


def select_classifier(branch: str, features: FeatureBundle, y: np.ndarray, embeddings: np.ndarray | None,
                      splits: list[tuple[np.ndarray, np.ndarray]], settings: dict, seed: int):
    """Train-only grid search; each candidate gets complete inner OOF logits."""
    best_value, best_model, best_oof, best_parameters = -np.inf, None, None, None
    history = []
    for C in settings["C"]:
        for weight in settings["positive_weight"]:
            oof = np.full(len(y), np.nan)
            for train, valid in splits:
                model = LogitClassifier(branch, C, weight, seed, settings["max_iter"])
                model.fit(features.take(train), y[train], None if embeddings is None else embeddings[train])
                oof[valid] = model.decision_function(features.take(valid), None if embeddings is None else embeddings[valid])
            if not np.isfinite(oof).all():
                raise ValueError("Inner folds must cover every training molecule exactly once")
            value = float(average_precision_score(y, oof))
            parameters = {"C": C, "positive_weight": weight}
            history.append({**parameters, "inner_oof_AP": value})
            if value > best_value:
                best_value, best_oof, best_parameters = value, oof.copy(), parameters
    best_model = LogitClassifier(branch, seed=seed, max_iter=settings["max_iter"], **best_parameters).fit(features, y, embeddings)
    return best_model, best_oof, {"selected": best_parameters, "search": history}


class SimilarityReference:
    """Score = max Tanimoto to actives minus max Tanimoto to inactives.

    Fixed logit scale 5 preserves ordering and places a neutral score at 0.5.
    Similarity scores are not calibrated probabilities.
    """
    def fit(self, features: FeatureBundle, y: np.ndarray):
        if len(np.unique(y)) != 2:
            raise ValueError("Similarity reference requires both training classes")
        self.reference_ = features.ecfp.astype(np.float32)
        self.y_ = np.asarray(y)
        self.counts_ = np.asarray(self.reference_.sum(axis=1)).ravel()
        self.training_structure_ids_ = features.structure_ids.copy()
        return self

    def decision_function(self, features: FeatureBundle, batch_size: int = 256) -> np.ndarray:
        output = []
        for start in range(0, features.ecfp.shape[0], batch_size):
            query = features.ecfp[start:start + batch_size].astype(np.float32)
            intersections = (query @ self.reference_.T).toarray()
            union = np.asarray(query.sum(axis=1)) + self.counts_[None, :] - intersections
            similarity = np.divide(intersections, union, out=np.zeros_like(intersections), where=union > 0)
            output.append(5 * (similarity[:, self.y_ == 1].max(axis=1) - similarity[:, self.y_ == 0].max(axis=1)))
        return np.concatenate(output)

    def predict_score(self, features: FeatureBundle) -> np.ndarray:
        return expit(self.decision_function(features))
