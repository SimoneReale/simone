"""Independent, reproducible molecular classifiers with training-only selection.

Binary and count fingerprint coordinates retain their original values. Only
continuous descriptors and frozen encoder coordinates are imputed/scaled.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import descriptor_transformer

if TYPE_CHECKING:
    from .representations import MolecularInputs


def _spec(label: str, estimator: str, representations: list[str], descriptors: bool = False,
          embeddings: bool = False) -> dict[str, Any]:
    return {"label": label, "estimator": estimator, "representations": representations,
            "descriptors": descriptors, "embeddings": embeddings}


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "M0": _spec("Morgan binary logistic regression", "logistic", ["ecfp"]),
    "M1": _spec("Morgan binary + descriptors logistic regression", "logistic", ["ecfp"], True),
    "M2": _spec("Frozen encoder + descriptors logistic regression", "logistic", [], True, True),
    "M4": _spec("Morgan count logistic regression", "logistic", ["counts"]),
    "M5": _spec("MACCS logistic regression", "logistic", ["maccs"]),
    "M6": _spec("Atom Pair logistic regression", "logistic", ["atompair"]),
    "M7": _spec("Direct Pharm2D logistic regression", "logistic", ["pharmacophore"]),
    "M8": _spec("Morgan + Pharm2D + descriptors logistic regression", "logistic", ["ecfp", "pharmacophore"], True),
    "M9": _spec("Morgan binary Random Forest", "random_forest", ["ecfp"]),
    "M10": _spec("Morgan count Extra Trees", "extra_trees", ["counts"]),
    "M11": _spec("Morgan count + descriptors XGBoost", "xgboost", ["counts"], True),
    "M12": _spec("Morgan count + descriptors LightGBM", "lightgbm", ["counts"], True),
    "M13": _spec("Multiple fingerprints Extra Trees", "extra_trees", ["ecfp", "maccs", "atompair", "rdkit", "avalon"]),
    "M14": _spec("Morgan binary Tanimoto SVM", "tanimoto_svm", ["ecfp"]),
    "similarity": _spec("Training active-minus-inactive maximum Tanimoto", "similarity", ["ecfp"]),
}


def require_model_dependencies(model_id: str) -> None:
    """Fail before long experiments when an explicitly requested extra is missing."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_id!r}; available: {', '.join(MODEL_SPECS)}")
    package = MODEL_SPECS[model_id]["estimator"]
    if package in {"xgboost", "lightgbm"} and importlib.util.find_spec(package) is None:
        raise ImportError(f"{model_id} requires optional dependency '{package}'. Install it with "
                          f"python -m pip install {package}; no substitute model was used.")


def tanimoto_kernel(query: sparse.csr_matrix, reference: sparse.csr_matrix,
                    batch_size: int = 256) -> np.ndarray:
    """Binary Tanimoto kernel, with identical empty bitsets assigned similarity 1.

    Only each intersection minibatch is materialized. The returned matrix is
    necessarily dense because scikit-learn's precomputed SVC needs that matrix.
    """
    if query.shape[1] != reference.shape[1] or batch_size < 1:
        raise ValueError("Tanimoto inputs must share feature width and have positive batch size")
    query, reference = query.astype(np.float32), reference.astype(np.float32)
    reference_counts = np.asarray(reference.sum(axis=1)).ravel()
    result = np.empty((query.shape[0], reference.shape[0]), dtype=np.float64)
    for start in range(0, query.shape[0], batch_size):
        current = query[start:start + batch_size]
        intersection = (current @ reference.T).toarray()
        union = np.asarray(current.sum(axis=1)) + reference_counts[None, :] - intersection
        values = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
        result[start:start + len(values)] = values
    return result


class ZooClassifier:
    """One separately serializable classifier; all learned transforms fit in fit()."""

    def __init__(self, model_id: str, parameters: dict[str, Any] | None = None,
                 seed: int = 2035, threads: int = 2):
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_id!r}")
        self.model_id = model_id
        self.parameters = deepcopy(parameters or {})
        self.seed, self.threads = seed, threads

    def _matrix(self, inputs: MolecularInputs, fit: bool = False) -> sparse.csr_matrix:
        spec = MODEL_SPECS[self.model_id]
        pieces = []
        for name in spec["representations"]:
            if name not in inputs.matrices:
                raise ValueError(f"{self.model_id} requires missing representation {name!r}")
            matrix = inputs.matrices[name]
            if matrix.shape[0] != len(inputs.structure_ids):
                raise ValueError(f"Representation {name!r} has a mismatched number of rows")
            if matrix.data.size and not np.isfinite(matrix.data).all():
                raise ValueError(f"Non-finite values in fingerprint representation {name!r}")
            pieces.append(matrix)
        if spec["descriptors"]:
            descriptors = np.asarray(inputs.descriptors, dtype=float).copy()
            descriptors[~np.isfinite(descriptors)] = np.nan
            if fit:
                self.descriptor_transformer_ = descriptor_transformer().fit(descriptors)
            pieces.append(sparse.csr_matrix(self.descriptor_transformer_.transform(descriptors)))
        if spec["embeddings"]:
            if inputs.embeddings is None or len(inputs.embeddings) != len(inputs.structure_ids):
                raise ValueError(f"{self.model_id} requires matching frozen molecular embeddings")
            if not np.isfinite(inputs.embeddings).all():
                raise ValueError("Frozen molecular embeddings must contain finite values")
            if fit:
                self.embedding_scaler_ = StandardScaler().fit(inputs.embeddings)
            pieces.append(sparse.csr_matrix(self.embedding_scaler_.transform(inputs.embeddings)))
        return sparse.hstack(pieces, format="csr", dtype=np.float64)

    def fit(self, inputs: MolecularInputs, y: np.ndarray) -> ZooClassifier:
        require_model_dependencies(self.model_id)
        y = np.asarray(y)
        if len(y) != len(inputs.structure_ids) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training requires matching rows with both binary activity classes")
        kind = MODEL_SPECS[self.model_id]["estimator"]
        params = deepcopy(self.parameters)
        weight = float(params.pop("positive_weight", 1))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("positive_weight must be finite and positive")
        matrix = self._matrix(inputs, fit=True)
        if kind == "logistic":
            defaults = {"C": 0.1, "max_iter": 3000}
            defaults.update(params)
            self.estimator_ = LogisticRegression(solver="liblinear", class_weight={0: 1, 1: weight},
                                                  random_state=self.seed, **defaults)
        elif kind in {"random_forest", "extra_trees"}:
            defaults = {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"}
            defaults.update(params)
            constructor = RandomForestClassifier if kind == "random_forest" else ExtraTreesClassifier
            self.estimator_ = constructor(class_weight={0: 1, 1: weight}, random_state=self.seed,
                                          n_jobs=self.threads, **defaults)
        elif kind == "xgboost":
            from xgboost import XGBClassifier
            defaults = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                        "reg_lambda": 10, "tree_method": "hist", "eval_metric": "logloss"}
            defaults.update(params)
            self.estimator_ = XGBClassifier(scale_pos_weight=weight, random_state=self.seed,
                                            n_jobs=self.threads, **defaults)
        elif kind == "lightgbm":
            from lightgbm import LGBMClassifier
            defaults = {"n_estimators": 200, "max_depth": 3, "num_leaves": 7,
                        "learning_rate": 0.05, "min_child_samples": 10, "reg_lambda": 10,
                        "deterministic": True, "force_col_wise": True, "verbosity": -1}
            defaults.update(params)
            self.estimator_ = LGBMClassifier(scale_pos_weight=weight, random_state=self.seed,
                                             n_jobs=self.threads, **defaults)
        elif kind == "tanimoto_svm":
            max_rows = int(params.pop("max_train_rows", 10000))
            max_mb = float(params.pop("max_kernel_mb", 768))
            self.batch_size_ = int(params.pop("batch_size", 256))
            required_mb = len(y) ** 2 * 8 / 1024 ** 2
            if len(y) > max_rows or required_mb > max_mb:
                raise ValueError(f"Tanimoto SVM needs {len(y)} training rows / {required_mb:.1f} MiB "
                                 f"for its dense kernel, exceeding max_train_rows={max_rows} "
                                 f"or max_kernel_mb={max_mb}. Configure these limits explicitly.")
            if matrix.data.size and not np.isin(matrix.data, [0, 1]).all():
                raise ValueError("M14 Tanimoto kernel requires binary Morgan fingerprints")
            self.reference_ = matrix.astype(np.float32)
            matrix = tanimoto_kernel(self.reference_, self.reference_, self.batch_size_)
            defaults = {"C": 1.0, "cache_size": 256}
            defaults.update(params)
            self.estimator_ = SVC(kernel="precomputed", class_weight={0: 1, 1: weight},
                                   random_state=self.seed, **defaults)
        elif kind == "similarity":
            if params:
                raise ValueError("The similarity reference has no fitted hyperparameters")
            self.reference_ = matrix.astype(np.float32)
            self.y_ = y.copy()
            self.batch_size_ = 256
        else:
            raise ValueError(f"Unsupported classifier family {kind}")
        if kind != "similarity":
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.estimator_.fit(matrix, y)
        self.signature_ = inputs.signature
        self.training_structure_ids_ = np.asarray(inputs.structure_ids).copy()
        self.n_features_in_ = self._matrix(inputs).shape[1]
        return self

    def decision_function(self, inputs: MolecularInputs) -> np.ndarray:
        if inputs.signature != self.signature_:
            raise ValueError("Model and inference representation signatures differ")
        matrix = self._matrix(inputs)
        kind = MODEL_SPECS[self.model_id]["estimator"]
        if kind == "similarity":
            output = []
            for start in range(0, matrix.shape[0], self.batch_size_):
                similarity = tanimoto_kernel(matrix[start:start + self.batch_size_], self.reference_, self.batch_size_)
                output.append(5 * (similarity[:, self.y_ == 1].max(axis=1) - similarity[:, self.y_ == 0].max(axis=1)))
            return np.concatenate(output) if output else np.empty(0)
        if kind == "tanimoto_svm":
            # Avoid materializing all arbitrary inference rows x training rows.
            return np.concatenate([self.estimator_.decision_function(tanimoto_kernel(
                matrix[start:start + self.batch_size_], self.reference_, self.batch_size_))
                for start in range(0, matrix.shape[0], self.batch_size_)]) if matrix.shape[0] else np.empty(0)
        if hasattr(self.estimator_, "decision_function"):
            return np.asarray(self.estimator_.decision_function(matrix), dtype=float)
        probabilities = self.estimator_.predict_proba(matrix)[:, 1]
        return logit(np.clip(probabilities, 1e-7, 1 - 1e-7))

    def predict_score(self, inputs: MolecularInputs) -> np.ndarray:
        """Return ranking scores; no probability-calibration claim is made."""
        return expit(self.decision_function(inputs))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> ZooClassifier:
        return joblib.load(path)


def parameter_grid(model_id: str, config: dict) -> list[dict[str, Any]]:
    """Return a small grid, with explicit per-model configuration taking precedence."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_id!r}")
    configured = config.get("zoo", {}).get("grids", {}).get(model_id)
    if configured is not None:
        grids = configured if isinstance(configured, list) else [configured]
        if not grids:
            raise ValueError(f"Empty parameter grid for {model_id}")
        normalized = [{key: value if isinstance(value, (list, tuple)) else [value] for key, value in row.items()}
                      for row in grids]
        result = list(ParameterGrid(normalized))
        if not result:
            raise ValueError(f"Empty parameter grid for {model_id}")
        return result
    settings = config.get("models", {})
    weights = settings.get("positive_weight", [1, 10])
    kind = MODEL_SPECS[model_id]["estimator"]
    if kind == "logistic":
        return list(ParameterGrid({"C": settings.get("C", [0.1, 1.0]), "positive_weight": weights,
                                   "max_iter": [settings.get("max_iter", 3000)]}))
    if kind == "similarity":
        return [{}]
    if kind == "tanimoto_svm":
        return list(ParameterGrid({"C": [0.1, 1.0], "positive_weight": weights}))
    return list(ParameterGrid({"max_depth": [3, 5], "positive_weight": weights}))


def select_model(model_id: str, inputs: MolecularInputs, y: np.ndarray,
                 splits: list[tuple[np.ndarray, np.ndarray]], config: dict, seed: int):
    """Select one model using complete inner OOF AP, then refit on all provided rows.

    The caller owns group-safe split construction. Coverage and structural
    disjointness are checked here to reject accidental leakage at the boundary.
    """
    require_model_dependencies(model_id)
    y = np.asarray(y)
    coverage = np.zeros(len(y), dtype=int)
    clean_splits = []
    for train, valid in splits:
        train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
        if (not len(train) or not len(valid) or np.any(train < 0) or np.any(valid < 0)
                or np.any(train >= len(y)) or np.any(valid >= len(y))):
            raise ValueError("Inner fold indices must be nonempty and in range")
        if len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid):
            raise ValueError("Inner folds contain repeated row indices")
        if set(inputs.structure_ids[train]) & set(inputs.structure_ids[valid]):
            raise ValueError("Chemical-structure leakage in inner model selection folds")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("Each inner training and validation partition requires both classes")
        coverage[valid] += 1
        clean_splits.append((train, valid))
    if not np.all(coverage == 1):
        raise ValueError("Inner folds must cover every training molecule exactly once")
    best_value, best_parameters, best_oof = -np.inf, None, None
    history = []
    for parameters in parameter_grid(model_id, config):
        oof = np.full(len(y), np.nan)
        for train, valid in clean_splits:
            model = ZooClassifier(model_id, parameters, seed, config.get("threads", 2))
            model.fit(inputs.take(train), y[train])
            oof[valid] = model.decision_function(inputs.take(valid))
        if not np.isfinite(oof).all():
            raise ValueError("Model returned non-finite inner OOF logits")
        value = float(average_precision_score(y, oof))
        history.append({"parameters": parameters, "inner_oof_AP": value})
        if value > best_value:
            best_value, best_parameters, best_oof = value, deepcopy(parameters), oof.copy()
    model = ZooClassifier(model_id, best_parameters, seed, config.get("threads", 2)).fit(inputs, y)
    return model, best_oof, {"selected": best_parameters, "search": history, "inner_oof_AP": best_value}
