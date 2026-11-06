"""End-to-end comparison/refit/export without optional encoders or legacy mutation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dreamq.cross_validation import fold_assignments
from dreamq.experiments import (ComparisonPredictor, compare_models, fit_final_models,
                                predict_models)
from dreamq.representations import featurize_inputs


def test_comparison_refit_reload_and_individual_exports(config, molecules, features):
    names = ["M0", "M4", "M7", "M9", "M16", "M17"]
    config["cv"].update(strategies=["stratified"], folds=2, inner_folds=2,
                         repetitions=2, bootstrap_samples=5)
    config["metrics"]["top_k"] = [3, 10]
    config["zoo"] = {
        "enabled": names,
        "ensemble_members": ["M0", "M4", "M7", "M9"],
        "ensemble_alphas": [0.5],
        "selection": {"enabled": True, "k": 3, "weights": [0.0, 0.5], "pool_size": 10},
        "grids": {
            "M0": {"C": [0.1], "positive_weight": [1]},
            "M4": {"C": [0.1], "positive_weight": [1]},
            "M7": {"C": [0.1], "positive_weight": [1]},
            "M9": {"n_estimators": [8], "max_depth": [2], "positive_weight": [1]},
        },
    }
    inputs = featurize_inputs(molecules, config["features"], config["cache_dir"], base=features)
    assert inputs.embeddings is None
    artifacts = Path(config["artifact_dir"])
    predictions = Path(config["prediction_dir"])
    legacy = {
        artifacts / "evaluation" / "results.csv": b"legacy evaluation sentinel\n",
        artifacts / "final" / "predictor.joblib": b"legacy model sentinel\n",
        predictions / "submission.csv": b"legacy submission sentinel\n",
    }
    for path, content in legacy.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assignments, _, signature = fold_assignments(molecules, "stratified", config, artifacts / "folds")
    fold_paths = {path: path.read_bytes() for path in (artifacts / "folds").iterdir()}

    results = compare_models(molecules, inputs, config, names)
    assert set(results.model) == set(names)
    assert len(results) == len(names)
    directory = artifacts / "comparison" / "stratified"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["split_signature"] == signature
    assert provenance["encoder"] is None
    oof = pd.read_csv(directory / "oof_predictions.csv")
    pooled = pd.read_csv(directory / "pooled_oof_predictions.csv")
    assert len(oof) == len(molecules) * len(names) * 2
    assert len(pooled) == len(molecules) * len(names)
    assert oof.groupby(["model", "row_index"]).size().eq(2).all()
    assert oof.groupby(["row_index", "repetition"]).fold.nunique().eq(1).all()
    expected = assignments.set_index(["row_index", "repetition"])["fold"]
    for row in oof.itertuples():
        assert row.fold == expected.loc[(row.row_index, row.repetition)]
    for model, part in pooled.groupby("model"):
        expected_logits = oof.loc[oof.model == model].groupby("row_index").logit.mean()
        np.testing.assert_allclose(part.sort_values("row_index").logit, expected_logits, atol=1e-12)
    for plot in ["pr.png", "roc.png", "enrichment.png"]:
        assert (directory / "plots" / plot).stat().st_size > 1000
    assert all(path.read_bytes() == content for path, content in fold_paths.items())

    predictor = fit_final_models(molecules, inputs, config, names)
    restored = ComparisonPredictor.load(config)
    assert predictor.encoder is None and restored.encoder is None
    assert predictor.metadata["encoder_identity"] is None
    for name in names:
        assert (artifacts / "comparison" / "final" / f"{name}.joblib").exists()
        np.testing.assert_array_equal(predictor.predict(inputs)[name], restored.predict(inputs)[name])

    # Arbitrary order must be preserved in every model's file, independent of
    # the canonical structure ordering used by the sparse caches.
    indices = np.array([7, 0, 20, 5, 9, 25])
    frame = molecules.iloc[indices].reset_index(drop=True).copy()
    ordered_inputs = inputs.take(indices)
    scores = predict_models(frame, ordered_inputs, restored, config, official=False)
    assert set(scores) == set(names)
    output = predictions / "comparison"
    for name in names:
        submission = pd.read_csv(output / f"{name}_arbitrary_scores.csv")
        assert list(submission.columns) == ["CatalogID", "ActivityScore"]
        assert submission.CatalogID.tolist() == frame.compound_id.tolist()
        np.testing.assert_allclose(submission.ActivityScore, scores[name], atol=1e-12)
        assert submission.ActivityScore.between(0, 1).all()
        for weight in ["0", "0.5"]:
            selection = pd.read_csv(output / f"{name}_diversity_{weight}_arbitrary_scores.csv")
            assert selection.CatalogID.tolist() == frame.compound_id.tolist()
            assert selection.Selected.sum() == 3
            assert "label" not in selection and "Label" not in selection
    diagnostics = pd.read_csv(output / "arbitrary_diagnostics.csv")
    assert diagnostics.compound_id.tolist() == frame.compound_id.tolist()
    assert set(f"logit_{name}" for name in names) <= set(diagnostics.columns)
    assert "label" not in diagnostics and "Label" not in diagnostics
    assert all(path.read_bytes() == content for path, content in legacy.items())
    assert not (artifacts / "encoder").exists()
