"""Persisted group-safe folds and nested evaluation shared by all baselines."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .features import FeatureBundle
from .fusion import fuse_logits, select_alpha
from .metrics import cluster_bootstrap, evaluate_metrics, plot_predictions
from .models import SimilarityReference, select_classifier
from .utils import digest, versions, write_json

LOG = logging.getLogger(__name__)


def make_splits(y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 2035,
                attempts: int = 30) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    y, groups = np.asarray(y), np.asarray(groups)
    if len(np.unique(y)) != 2:
        raise ValueError("Cannot construct binary evaluation folds without both classes")
    maximum = min(folds, len(np.unique(groups)), *(len(np.unique(groups[y == label])) for label in [0, 1]))
    for count in range(maximum, 1, -1):
        for attempt in range(attempts):
            candidate_seed = seed + attempt
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=candidate_seed)
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
            valid = all(len(np.unique(y[t])) == 2 and len(np.unique(y[v])) == 2 and
                        not (set(groups[t]) & set(groups[v])) for t, v in splits)
            if valid:
                metadata = {"requested_folds": folds, "actual_folds": count, "seed": candidate_seed,
                            "positive_groups": len(np.unique(groups[y == 1])),
                            "limitation": None if count == folds else f"Reduced from {folds} to {count} folds to retain both classes"}
                return splits, metadata
    raise ValueError("No feasible grouped split with both classes in every train and validation partition. Inspect positive scaffolds; do not report misleading metrics.")


def fold_assignments(frame: pd.DataFrame, strategy: str, config: dict, directory: Path):
    if strategy not in {"stratified", "scaffold"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    directory.mkdir(parents=True, exist_ok=True)
    column = "structure_id" if strategy == "stratified" else "scaffold"
    y, groups = frame.label.to_numpy(), frame[column].to_numpy()
    signature = digest({"rows": frame[["compound_id", "structure_id", "scaffold", "label"]].values.tolist(),
                        "strategy": strategy, "folds": config["cv"]["folds"], "version": 1,
                        "repetitions": config["cv"]["repetitions"], "seed": config["seed"],
                        "attempts": config["cv"]["split_attempts"]})
    path = directory / f"{strategy}-{signature}.csv"
    metadata_path = path.with_suffix(".json")
    if path.exists():
        assignments = pd.read_csv(path)
        metadata = json.loads(metadata_path.read_text())
    else:
        rows, metadata = [], []
        for repetition in range(config["cv"]["repetitions"]):
            splits, info = make_splits(y, groups, config["cv"]["folds"], config["seed"] + 1009 * repetition,
                                       config["cv"]["split_attempts"])
            metadata.append(info)
            for fold, (_, valid) in enumerate(splits):
                rows.extend({"row_index": int(i), "repetition": repetition, "fold": fold,
                             "compound_id": frame.iloc[i].compound_id, "structure_id": frame.iloc[i].structure_id,
                             "scaffold": frame.iloc[i].scaffold} for i in valid)
        assignments = pd.DataFrame(rows)
        assignments.to_csv(path, index=False)
        write_json(metadata_path, metadata)
    # Check persisted assignments as well as generated ones.
    for _, repetition in assignments.groupby("repetition"):
        if sorted(repetition.row_index.tolist()) != list(range(len(frame))):
            raise ValueError("Stored folds do not cover each molecule once per repetition")
        for _, fold in repetition.groupby("fold"):
            valid = fold.row_index.to_numpy()
            train = np.setdiff1d(np.arange(len(frame)), valid)
            if set(groups[train]) & set(groups[valid]):
                raise ValueError("Group leakage in stored folds")
    return assignments, metadata, signature


def fit_branches(features: FeatureBundle, y: np.ndarray, embeddings: np.ndarray,
                 inner_splits, config: dict, seed: int, include_m0: bool = True):
    models, inner_logits, details = {}, {}, {}
    with threadpool_limits(limits=config.get("threads", 2)):
        for branch in (["M0", "M1", "M2"] if include_m0 else ["M1", "M2"]):
            models[branch], inner_logits[branch], details[branch] = select_classifier(
                branch, features, y, embeddings if branch == "M2" else None, inner_splits, config["models"], seed)
    alpha, history = select_alpha(y, inner_logits["M1"], inner_logits["M2"], config["models"]["alpha"])
    details["fusion"] = {"alpha": alpha, "search": history}
    return models, alpha, details, inner_logits


def evaluate(frame: pd.DataFrame, features: FeatureBundle, embeddings: np.ndarray, config: dict,
             encoder_metadata: dict) -> pd.DataFrame:
    root = Path(config["artifact_dir"]) / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    y = frame.label.to_numpy()
    all_results = []
    for strategy in config["cv"]["strategies"]:
        directory = root / strategy
        directory.mkdir(parents=True, exist_ok=True)
        assignments, split_info, signature = fold_assignments(frame, strategy, config, Path(config["artifact_dir"]) / "folds")
        groups = frame["structure_id" if strategy == "stratified" else "scaffold"].to_numpy()
        predictions, fold_results, selection = [], [], []
        for (repetition, fold), part in assignments.groupby(["repetition", "fold"]):
            valid = part.row_index.to_numpy()
            train = np.setdiff1d(np.arange(len(frame)), valid)
            seed = config["seed"] + int(repetition) * 1009 + int(fold) * 97
            inner, info = make_splits(y[train], groups[train], config["cv"]["inner_folds"], seed,
                                      config["cv"]["split_attempts"])
            LOG.info("Nested CV %s repetition=%d fold=%d train=%d valid=%d", strategy, repetition, fold, len(train), len(valid))
            models, alpha, details, inner_logits = fit_branches(features.take(train), y[train], embeddings[train], inner, config, seed)
            logits = {branch: model.decision_function(features.take(valid), embeddings[valid] if branch == "M2" else None)
                      for branch, model in models.items()}
            logits["M3"] = fuse_logits(logits["M1"], logits["M2"], alpha)
            similarity = SimilarityReference().fit(features.take(train), y[train])
            logits["similarity"] = similarity.decision_function(features.take(valid))
            fold_dir = directory / f"repeat{repetition}_fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"models": models, "alpha": alpha, "similarity": similarity,
                         "encoder_identity": encoder_metadata["identity"]}, fold_dir / "models.joblib")
            inner_rows = pd.DataFrame({"row_index": train, **{f"logit_{k}": v for k, v in inner_logits.items()}})
            inner_rows.to_csv(fold_dir / "inner_oof.csv", index=False)
            inner_assignments = [{"inner_fold": j, "row_index": int(train[i])} for j, (_, val) in enumerate(inner) for i in val]
            pd.DataFrame(inner_assignments).to_csv(fold_dir / "inner_folds.csv", index=False)
            selection.append({"repetition": int(repetition), "fold": int(fold), "inner_split": info, "selection": details})
            for model, values in logits.items():
                stats = evaluate_metrics(y[valid], expit(values), config["metrics"]["top_k"], config["metrics"]["threshold"])
                fold_results.append({"repetition": int(repetition), "fold": int(fold), "model": model, **stats})
                predictions.extend({"row_index": int(i), "compound_id": frame.iloc[i].compound_id,
                                    "structure_id": frame.iloc[i].structure_id, "label": int(y[i]),
                                    "repetition": int(repetition), "fold": int(fold), "model": model,
                                    "logit": float(z), "score": float(expit(z)), "alpha": float(alpha)} for i, z in zip(valid, values))
        oof = pd.DataFrame(predictions)
        oof.to_csv(directory / "oof_predictions.csv", index=False)
        write_json(directory / "fold_metrics.json", fold_results)
        write_json(directory / "model_selection.json", selection)
        write_json(directory / "provenance.json", {"split_signature": signature, "splits": split_info,
                   "encoder": encoder_metadata, "config": config, "versions": versions(),
                   "development_results": config.get("development", False)})
        # A molecule contributes once after averaging its logits over repetitions.
        pooled = oof.groupby(["model", "row_index"], as_index=False).agg(logit=("logit", "mean"), label=("label", "first"))
        pooled["score"] = expit(pooled.logit)
        pooled.to_csv(directory / "pooled_oof_predictions.csv", index=False)
        pooled_logits, metrics_output = {}, {}
        for model, subset in pooled.groupby("model"):
            subset = subset.sort_values("row_index")
            values = subset.logit.to_numpy()
            pooled_logits[model] = values
            stats = evaluate_metrics(y, expit(values), config["metrics"]["top_k"], config["metrics"]["threshold"])
            bootstrap = cluster_bootstrap(y, expit(values), frame.structure_id.to_numpy(), config["cv"]["bootstrap_samples"],
                                           config["seed"], config["metrics"]["top_k"])
            per_fold = [row for row in fold_results if row["model"] == model]
            fold_ap = [row["AP"] for row in per_fold if row["AP"] is not None]
            repetition_ap = []
            for _, rep in oof.loc[oof.model == model].groupby("repetition"):
                repetition_ap.append(evaluate_metrics(rep.label.to_numpy(), rep.score.to_numpy())["AP"])
            metrics_output[model] = {"pooled": stats, "bootstrap": bootstrap,
                                     "fold_AP_mean": float(np.mean(fold_ap)), "fold_AP_std": float(np.std(fold_ap, ddof=1)) if len(fold_ap) > 1 else 0.0,
                                     "repetition_AP": repetition_ap}
            all_results.append({"strategy": strategy, "model": model, "development": config.get("development", False),
                                **{k: stats[k] for k in ["AP", "ROC_AUC", "EF@1%", "EF@5%", *[f"recall@{k}" for k in config["metrics"]["top_k"]]]},
                                "AP_fold_std": metrics_output[model]["fold_AP_std"],
                                "AP_CI_low": bootstrap["intervals_95"].get("AP", [None, None])[0],
                                "AP_CI_high": bootstrap["intervals_95"].get("AP", [None, None])[1]})
        write_json(directory / "metrics.json", metrics_output)
        plot_predictions(y, pooled_logits, directory / "plots")
        pd.DataFrame(all_results).to_csv(root / "results.csv", index=False)
    return pd.DataFrame(all_results)
