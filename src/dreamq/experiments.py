"""Independent molecular models, nested comparisons, refits and diverse portfolios.

All selection and ensemble weights use inner training predictions only. Existing
M0--M3 artifacts and split assignments are never overwritten by this module.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import logging
from pathlib import Path
import re
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .cross_validation import fold_assignments, make_splits
from .data import audit, pretraining_entities
from .ensemble import LogitFusion, build_rank_ensemble
from .features import feature_signature
from .fusion import fuse_logits, select_alpha
from .inference import training_signature, validate_submission
from .metrics import cluster_bootstrap, evaluate_metrics, plot_predictions
from .model_zoo import MODEL_SPECS, parameter_grid, require_model_dependencies, select_model
from .pretraining import FrozenEncoder
from .representations import MolecularInputs, featurize_inputs
from .selection import select_diverse, selection_metrics
from .utils import digest, versions, write_json

LOG = logging.getLogger(__name__)
EXTRA_SPECS = {
    "M3": {"label": "Inner-OOF logit fusion of M1 and M2"},
    "M15": {"label": "Chemprop supervised molecular graph ensemble"},
    "M16": {"label": "Fixed equal-weight cross-fingerprint rank ensemble"},
    "M17": {"label": "Inner-OOF selected single/pair rank ensemble"},
}
DEFAULT_MODELS = ["M0", "M1", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M13", "M14", "similarity", "M16", "M17"]


def comparison_root(config: dict, predictions: bool = False) -> Path:
    root = Path(config["prediction_dir" if predictions else "artifact_dir"]) / "comparison"
    name = config.get("zoo", {}).get("output_name")
    if name:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(name)):
            raise ValueError("Run name must contain letters, digits, underscores or hyphens")
        root /= name
    return root


def protect_run(config: dict, names: list[str], frame: pd.DataFrame) -> None:
    root = comparison_root(config)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    expected = {"models": sorted(names), "training_data_signature": training_signature(frame)}
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError("This run directory belongs to different models or data; choose a new --run-name")
    write_json(path, expected)


def catalog() -> dict:
    result = deepcopy(MODEL_SPECS)
    result.update(deepcopy(EXTRA_SPECS))
    for name, backend in [("M11", "xgboost"), ("M12", "lightgbm"), ("M15", "chemprop")]:
        result[name]["optional_backend"] = backend
        result[name]["installed"] = importlib.util.find_spec(backend) is not None
    return result


def requested_models(config: dict, override: str | list[str] | None = None) -> list[str]:
    names = (override.split(",") if isinstance(override, str) else override)
    if names == ["all"]:
        names = list(catalog())
    names = list(names or config.get("zoo", {}).get("enabled", DEFAULT_MODELS))
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose a nonempty list of distinct model IDs")
    unknown = set(names) - set(catalog())
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; run list-models")
    return names


def required_models(names: list[str], config: dict) -> list[str]:
    result = list(names)
    if "M3" in result:
        result.extend(["M1", "M2"])
    if {"M16", "M17"} & set(result):
        members = config.get("zoo", {}).get("ensemble_members", ["M0", "M4", "M7", "M9", "M10"])
        if not members or set(members) & {"M3", "M16", "M17"}:
            raise ValueError("Ensemble members must be independent base models")
        result.extend(members)
    result = list(dict.fromkeys(result))
    for name in result:
        if name in MODEL_SPECS:
            require_model_dependencies(name)
        elif name == "M15":
            if importlib.util.find_spec("chemprop") is None:
                raise ImportError("M15 requires Chemprop: install .[graph] or use the .venv-chemprop interpreter; see docs/chemprop.md")
            from .chemprop_model import _chemprop_modules
            _chemprop_modules()  # Validate transitive graph dependencies before long comparisons.
        elif name not in EXTRA_SPECS:
            raise ValueError(f"Unknown ensemble dependency {name}")
    return result


def prepare_inputs(config: dict, names: list[str], arbitrary: pd.DataFrame | None = None, encoder_override: FrozenEncoder | None = None):
    tables, report = audit(config)
    required = required_models(names, config)
    frame = pd.concat([tables["labeled"], tables["leaderboard"], *([] if arbitrary is None else [arbitrary])], ignore_index=True)
    base = None
    expected = feature_signature(config["features"])
    wanted = set(frame.structure_id)
    # Reuse prior full-collection or development caches without re-extracting Pharm2D.
    for path in sorted((Path(config["cache_dir"]) / "features").glob("*.joblib"), key=lambda p: p.stat().st_size):
        candidate = joblib.load(path)
        if candidate.signature == expected and wanted <= set(candidate.structure_ids):
            base = candidate
            LOG.info("Reusing existing base molecular cache %s", path)
            break
    inputs = featurize_inputs(frame, config["features"], config["cache_dir"], base=base)
    encoder = None
    if "M2" in required:
        from .cli import load_encoder
        permitted = pretraining_entities(tables, config)
        encoder = encoder_override or load_encoder(config, permitted, expected)
        if encoder.metadata["feature_signature"] != expected:
            raise ValueError("Frozen encoder feature signature differs from requested representation")
        # Frozen encoding uses the unchanged original feature signature.
        if base is None:
            from .features import featurize
            base = featurize(frame, config["features"], config["cache_dir"])
        inputs.embeddings = encoder.transform(base.for_frame(frame))
    return tables, inputs, report, encoder


def _select_graph(inputs, y, splits, config, seed):
    from .chemprop_model import ChempropClassifier
    from sklearn.model_selection import ParameterGrid
    configured = config.get("zoo", {}).get("grids", {}).get("M15", {})
    grids = configured if isinstance(configured, list) else [configured]
    normalized = [{key: value if isinstance(value, (list, tuple)) else [value] for key, value in grid.items()} for grid in grids]
    candidates = list(ParameterGrid(normalized))
    if not candidates:
        raise ValueError("M15 parameter grid is empty")
    history, best, best_value, best_oof = [], None, -np.inf, None
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(inputs) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Graph selection needs matching binary labels with both classes")
    counts = np.zeros(len(y), dtype=int)
    clean_splits = []
    for train, valid in splits:
        train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
        if not len(train) or not len(valid) or np.any(train < 0) or np.any(valid < 0) or np.any(train >= len(y)) or np.any(valid >= len(y)):
            raise ValueError("Graph inner fold indices must be nonempty and in range")
        if len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid):
            raise ValueError("Graph inner folds contain repeated indices")
        if set(inputs.structure_ids[train]) & set(inputs.structure_ids[valid]):
            raise ValueError("Chemical group leakage in graph inner folds")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("Graph inner training and validation folds need both classes")
        counts[valid] += 1
        clean_splits.append((train, valid))
    if not np.all(counts == 1):
        raise ValueError("Graph inner OOF predictions must cover each training row once")
    for parameters in candidates:
        oof = np.full(len(y), np.nan)
        for train, valid in clean_splits:
            model = ChempropClassifier(parameters, seed, config.get("threads", 2)).fit(inputs.take(train), y[train])
            oof[valid] = model.decision_function(inputs.take(valid))
        if not np.isfinite(oof).all():
            raise ValueError("Non-finite graph inner OOF logits")
        value = float(average_precision_score(y, oof))
        history.append({"parameters": parameters, "inner_oof_AP": value})
        if value > best_value:
            best, best_value, best_oof = parameters, value, oof.copy()
    model = ChempropClassifier(best, seed, config.get("threads", 2)).fit(inputs, y)
    return model, best_oof, {"selected": best, "search": history}


def fit_models(inputs: MolecularInputs, y: np.ndarray, splits, config: dict, seed: int, names: list[str]):
    required = required_models(names, config)
    models, oof, details = {}, {}, {}
    with threadpool_limits(limits=config.get("threads", 2)):
        for name in required:
            if name in EXTRA_SPECS and name != "M15":
                continue
            LOG.info("Selecting and fitting %s", name)
            if name == "M15":
                models[name], oof[name], details[name] = _select_graph(inputs, y, splits, config, seed)
            else:
                models[name], oof[name], details[name] = select_model(name, inputs, y, splits, config, seed)
    if "M3" in required:
        alpha, search = select_alpha(y, oof["M1"], oof["M2"], config["models"]["alpha"])
        models["M3"] = LogitFusion(models["M1"], models["M2"], alpha)
        oof["M3"] = fuse_logits(oof["M1"], oof["M2"], alpha)
        details["M3"] = {"selected": {"alpha": alpha}, "search": search}
    for name in ["M16", "M17"]:
        if name in required:
            settings = config.get("zoo", {})
            members = settings.get("ensemble_members", ["M0", "M4", "M7", "M9", "M10"])
            models[name], oof[name], details[name] = build_rank_ensemble(
                models, oof, y, members, selected=name == "M17",
                alphas=settings.get("ensemble_alphas", [0.25, 0.5, 0.75]),
                max_correlation=settings.get("max_correlation", 1.0))
    return models, oof, details


def _selection_records(frame, inputs, scores, config):
    settings = config.get("zoo", {}).get("selection", {})
    if not settings.get("enabled", True):
        return [], {}
    # A fixed predeclared policy grid, never tuned against these reporting labels.
    k = min(int(settings.get("k", 50)), len(np.unique(inputs.structure_ids)))
    records, indices = [], {}
    for weight in settings.get("weights", [0.0, 0.25, 0.5]):
        chosen = select_diverse(inputs, scores, k=k, diversity_weight=weight, pool_size=settings.get("pool_size"))
        policy = f"diversity_{float(weight):g}"
        indices[policy] = chosen
        stats = selection_metrics(frame.label.to_numpy(), chosen) if "label" in frame else {"selected": len(chosen)}
        records.append({"policy": policy, "diversity_weight": float(weight), "requested_k": settings.get("k", 50), "effective_k": k, "budget_reduced": k < int(settings.get("k", 50)),
                        "unique_scaffolds": int(frame.iloc[chosen].scaffold.nunique()), **stats})
    return records, indices


def compare_models(frame, inputs: MolecularInputs, config: dict, names: list[str], encoder_metadata: dict | None = None):
    protect_run(config, names, frame)
    root = comparison_root(config)
    root.mkdir(parents=True, exist_ok=True)
    y = frame.label.to_numpy()
    results = []
    for strategy in config["cv"]["strategies"]:
        directory = root / strategy
        directory.mkdir(parents=True, exist_ok=True)
        assignments, split_info, signature = fold_assignments(frame, strategy, config, Path(config["artifact_dir"]) / "folds")
        groups = frame["structure_id" if strategy == "stratified" else "scaffold"].to_numpy()
        predictions, fold_metrics, portfolio_metrics, selections = [], [], [], []
        for (repetition, fold), part in assignments.groupby(["repetition", "fold"]):
            valid = part.row_index.to_numpy()
            train = np.setdiff1d(np.arange(len(frame)), valid)
            seed = config["seed"] + int(repetition) * 1009 + int(fold) * 97
            inner, info = make_splits(y[train], groups[train], config["cv"]["inner_folds"], seed, config["cv"]["split_attempts"])
            LOG.info("Model comparison %s repeat=%d fold=%d", strategy, repetition, fold)
            models, inner_logits, details = fit_models(inputs.take(train), y[train], inner, config, seed, names)
            fold_dir = directory / f"repeat{repetition}_fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(models, fold_dir / "models.joblib", compress=3)
            pd.DataFrame({"row_index": train, **inner_logits}).to_csv(fold_dir / "inner_oof.csv", index=False)
            pd.DataFrame(inner_logits).corr(method="spearman").to_csv(fold_dir / "inner_rank_correlations.csv")
            pd.DataFrame([{"inner_fold": i, "row_index": int(train[j])} for i, (_, val) in enumerate(inner) for j in val]).to_csv(fold_dir / "inner_folds.csv", index=False)
            selections.append({"repetition": int(repetition), "fold": int(fold), "inner_split": info, "models": details})
            for name in names:
                logits = np.asarray(models[name].decision_function(inputs.take(valid)), dtype=float)
                if logits.shape != (len(valid),) or not np.isfinite(logits).all():
                    raise ValueError(f"Invalid validation logits from {name}")
                scores = expit(logits)
                fold_metrics.append({"model": name, "repetition": int(repetition), "fold": int(fold),
                                     **evaluate_metrics(y[valid], scores, config["metrics"]["top_k"], config["metrics"]["threshold"])})
                predictions.extend({"model": name, "row_index": int(i), "structure_id": frame.iloc[i].structure_id,
                                    "compound_id": frame.iloc[i].compound_id, "label": int(y[i]),
                                    "repetition": int(repetition), "fold": int(fold), "logit": float(z), "score": float(s)}
                                   for i, z, s in zip(valid, logits, scores))
                records, portfolios = _selection_records(frame.iloc[valid].reset_index(drop=True), inputs.take(valid), scores, config)
                portfolio_metrics.extend({"model": name, "repetition": int(repetition), "fold": int(fold), **r} for r in records)
                for policy, chosen in portfolios.items():
                    pd.DataFrame({"row_index": valid[chosen], "compound_id": frame.iloc[valid[chosen]].compound_id.to_numpy()}).to_csv(fold_dir / f"{name}_{policy}.csv", index=False)
        oof = pd.DataFrame(predictions)
        oof.to_csv(directory / "oof_predictions.csv", index=False)
        pooled = oof.groupby(["model", "row_index"], as_index=False).agg(logit=("logit", "mean"), label=("label", "first"))
        pooled["score"] = expit(pooled.logit)
        pooled.to_csv(directory / "pooled_oof_predictions.csv", index=False)
        metrics, logits_by_model = {}, {}
        for name in names:
            part = pooled.loc[pooled.model == name].sort_values("row_index")
            scores = part.score.to_numpy()
            stats = evaluate_metrics(y, scores, config["metrics"]["top_k"], config["metrics"]["threshold"])
            uncertainty = cluster_bootstrap(y, scores, frame.structure_id.to_numpy(), config["cv"]["bootstrap_samples"], config["seed"], config["metrics"]["top_k"])
            fold_ap = [r["AP"] for r in fold_metrics if r["model"] == name and r["AP"] is not None]
            repetition_ap = [evaluate_metrics(rep.label.to_numpy(), rep.score.to_numpy())["AP"]
                             for _, rep in oof.loc[oof.model == name].groupby("repetition")]
            metrics[name] = {"pooled": stats, "bootstrap": uncertainty, "repetition_AP": repetition_ap,
                             "fold_AP_mean": float(np.mean(fold_ap)), "fold_AP_std": float(np.std(fold_ap, ddof=1)) if len(fold_ap) > 1 else 0.0}
            low, high = uncertainty["intervals_95"].get("AP", [None, None])
            results.append({"strategy": strategy, "model": name, "label": catalog()[name]["label"],
                            "development": config.get("development", False),
                            **{key: stats[key] for key in ["AP", "ROC_AUC", "EF@1%", "EF@5%", *[f"recall@{k}" for k in config["metrics"]["top_k"]]]},
                            "AP_fold_std": metrics[name]["fold_AP_std"], "AP_CI_low": low, "AP_CI_high": high})
            logits_by_model[name] = part.logit.to_numpy()
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "fold_metrics.json", fold_metrics)
        write_json(directory / "model_selection.json", selections)
        pd.DataFrame(portfolio_metrics).to_csv(directory / "portfolio_metrics.csv", index=False)
        write_json(directory / "provenance.json", {"config": config, "versions": versions(), "split_signature": signature,
                   "splits": split_info, "encoder": encoder_metadata, "requested_models": names,
                   "selection_note": "Fixed diversity policies; no policy selection against reporting labels. Scaffold counts are not official MCS chemical series."})
        plot_predictions(y, logits_by_model, directory / "plots")
        pd.DataFrame(results).to_csv(root / "results.csv", index=False)
    return pd.DataFrame(results)


def fit_final_models(frame, inputs: MolecularInputs, config: dict, names: list[str], encoder: FrozenEncoder | None = None):
    protect_run(config, names, frame)
    strategy = config["cv"].get("final_strategy", "stratified")
    if strategy not in {"stratified", "scaffold"}:
        raise ValueError("final_strategy must be stratified or scaffold")
    groups = frame["structure_id" if strategy == "stratified" else "scaffold"].to_numpy()
    splits, info = make_splits(frame.label.to_numpy(), groups, config["cv"]["inner_folds"], config["seed"], config["cv"]["split_attempts"])
    models, oof, details = fit_models(inputs, frame.label.to_numpy(), splits, config, config["seed"], names)
    metadata = {"config": config, "versions": versions(), "requested_models": names, "selection": details,
                "splits": info, "training_data_signature": training_signature(frame), "input_signature": inputs.signature,
                "encoder_identity": encoder.metadata["identity"] if encoder else None,
                "development": config.get("development", False), "calibrated": False}
    predictor = ComparisonPredictor(models, names, metadata, encoder)
    root = comparison_root(config) / "final"
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, root / "predictor.joblib", compress=3)
    for name in names:
        joblib.dump(models[name], root / f"{name}.joblib", compress=3)
    write_json(root / "metadata.json", metadata)
    pd.DataFrame({"row_index": np.arange(len(frame)), **oof}).to_csv(root / "inner_oof.csv", index=False)
    pd.DataFrame(oof).corr(method="spearman").to_csv(root / "inner_rank_correlations.csv")
    pd.DataFrame([{"fold": j, "row_index": int(i)} for j, (_, val) in enumerate(splits) for i in val]).to_csv(root / "inner_folds.csv", index=False)
    return predictor


class ComparisonPredictor:
    def __init__(self, models: dict, requested: list[str], metadata: dict, encoder: FrozenEncoder | None = None):
        self.models, self.requested, self.metadata, self.encoder = models, requested, metadata, encoder

    def predict(self, inputs: MolecularInputs, names: list[str] | None = None):
        if inputs.signature != self.metadata["input_signature"]:
            raise ValueError("Molecular representation configuration differs from fitted models")
        names = names or self.requested
        if not set(names) <= set(self.models):
            raise ValueError("Requested model was not fitted")
        return {name: np.asarray(self.models[name].predict_score(inputs)) for name in names}

    @classmethod
    def load(cls, config: dict):
        return joblib.load(comparison_root(config) / "final" / "predictor.joblib")


def predict_models(frame, inputs: MolecularInputs, predictor: ComparisonPredictor, config: dict,
                   names: list[str] | None = None, official: bool = True):
    root = comparison_root(config, predictions=True)
    root.mkdir(parents=True, exist_ok=True)
    predictions = predictor.predict(inputs, names)
    summary = []
    for name, scores in predictions.items():
        settings = config["submission"]
        submission = pd.DataFrame({settings["id_column"]: frame.compound_id.to_numpy(), settings["score_column"]: scores})
        validate_submission(submission, frame.compound_id.tolist(), settings,
                            config["data"].get("expected_leaderboard_rows") if official else len(frame))
        suffix = "submission" if official else "arbitrary_scores"
        submission.to_csv(root / f"{name}_{suffix}.csv", index=False)
        # No target column is ever passed to candidate selection or written to submissions.
        unlabeled_frame = frame.drop(columns=[c for c in ["label", "Label"] if c in frame])
        records, portfolios = _selection_records(unlabeled_frame, inputs, scores, config)
        for record in records:
            policy = record["policy"]
            chosen = portfolios[policy]
            output = submission.copy()
            output["Selected"] = np.isin(np.arange(len(frame)), chosen).astype(int)
            output.to_csv(root / f"{name}_{policy}_{suffix}.csv", index=False)
            summary.append({"model": name, **record})
    diagnostics = pd.DataFrame({"compound_id": frame.compound_id.to_numpy(), "structure_id": frame.structure_id.to_numpy(),
                                **{f"score_{name}": scores for name, scores in predictions.items()},
                                **{f"logit_{name}": predictor.models[name].decision_function(inputs) for name in predictions}})
    diagnostics.to_csv(root / ("leaderboard_diagnostics.csv" if official else "arbitrary_diagnostics.csv"), index=False)
    pd.DataFrame(summary).to_csv(root / "selection_summary.csv", index=False)
    write_json(root / "prediction_metadata.json", {"rows": len(frame), "models": list(predictions),
               "development": config.get("development", False), "calibrated": False,
               "official_schema_verified": False, "selection_column_is_diagnostic": True,
               "note": "Diversity variants change the selected portfolio, not the activity ranking. Scaffold counts are proxies, not official chemical series."})
    return predictions
