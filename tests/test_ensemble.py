"""Training-only rank normalization, conservative selection and deployment."""
from __future__ import annotations

import joblib
import numpy as np
import pytest
from scipy.special import expit

from dreamq.cross_validation import make_splits
from dreamq.ensemble import EmpiricalRank, build_rank_ensemble
from dreamq.model_zoo import select_model
from dreamq.representations import MolecularInputs


class PredictionSpy:
    """Fails if ensemble selection tries to score fitted training models."""
    def __init__(self):
        self.training_structure_ids_ = np.array(["train-a", "train-b", "train-c", "train-d"])
        self.calls = 0

    def decision_function(self, inputs):
        self.calls += 1
        raise AssertionError("Selection must use provided inner OOF logits, not refitted predictions")


def test_smoothed_midpoint_cdf_ties_and_query_batch_invariance():
    reference = np.array([0., 0., 2., 4.])
    rank = EmpiricalRank().fit(reference)
    np.testing.assert_allclose(rank.transform(np.array([-1., 0., 1., 2., 4., 5.])),
                               [.1, .3, .5, .6, .8, .9])
    before = rank.reference_.copy()
    single = rank.transform(np.array([2.]))[0]
    mixed_batch = rank.transform(np.array([2., -100., 100., 2.]))
    assert single == mixed_batch[0] == mixed_batch[-1]
    reference[:] = 1e9
    np.testing.assert_array_equal(rank.reference_, before)


@pytest.mark.parametrize("query", [np.array([np.nan]), np.array([np.inf]), np.array([[1., 2.]])])
def test_rank_query_rejects_invalid_values(query):
    with pytest.raises(ValueError):
        EmpiricalRank().fit(np.array([1., 2., 3.])).transform(query)


def test_selected_ensemble_uses_only_supplied_oof_and_falls_back_to_single():
    y = np.array([0, 1, 0, 1])
    good = np.array([-2., 2., -1., 1.])
    models = {"good": PredictionSpy(), "opposite": PredictionSpy()}
    ensemble, oof_logits, details = build_rank_ensemble(models, {"good": good, "opposite": -good},
                                                       y, ["good", "opposite"], selected=True,
                                                       alphas=[.25, .5, .75])
    assert details["selected"] == {"good": 1.}
    assert ensemble.weights == {"good": 1.}
    assert set(ensemble.models) == {"good"}
    assert all(model.calls == 0 for model in models.values())
    np.testing.assert_array_equal(ensemble.normalizers["good"].reference_, np.sort(good))
    expected = EmpiricalRank().fit(good).transform(good)
    np.testing.assert_allclose(expit(oof_logits), expected)
    assert any(len(candidate["weights"]) == 2 for candidate in details["search"])


def test_missing_models_and_missing_oof_are_explicit_errors():
    y = np.array([0, 1, 0, 1])
    logits = np.array([-2., 2., -1., 1.])
    with pytest.raises(ValueError, match="not fitted"):
        build_rank_ensemble({"one": PredictionSpy()}, {"one": logits}, y, ["one", "missing"])
    with pytest.raises(ValueError):
        build_rank_ensemble({"one": PredictionSpy()}, {}, y, ["one"])
    with pytest.raises(ValueError):
        build_rank_ensemble({"one": PredictionSpy()}, {"one": logits[:2]}, y, ["one"])
    with pytest.raises(ValueError):
        build_rank_ensemble({"one": PredictionSpy()}, {"one": logits}, np.zeros(4), ["one"])


def test_rank_ensemble_real_models_serialization_and_inference_batch_independence(config, molecules, features, tmp_path):
    inputs = MolecularInputs(features.structure_ids, molecules.canonical_smiles.to_numpy(),
                             {"ecfp": features.ecfp}, features.descriptors, "ensemble-test")
    y = molecules.label.to_numpy()
    splits, _ = make_splits(y, inputs.structure_ids, folds=2)
    models, inner_logits = {}, {}
    for model_id in ["M0", "M1"]:
        models[model_id], inner_logits[model_id], _ = select_model(model_id, inputs, y, splits, config, 2035)
    ensemble, _, details = build_rank_ensemble(models, inner_logits, y, ["M0", "M1"])
    assert details["selected"] == {"M0": .5, "M1": .5}
    full = ensemble.predict_score(inputs)
    first_only = ensemble.predict_score(inputs.take([0]))
    np.testing.assert_allclose(first_only, full[:1])
    assert np.isfinite(full).all() and np.all((full > 0) & (full < 1))
    path = tmp_path / "ensemble.joblib"
    joblib.dump(ensemble, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_score(inputs), full)
    np.testing.assert_allclose(expit(restored.decision_function(inputs)), full)
