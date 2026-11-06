"""Optional graph model contract plus real Chemprop training when installed."""
import importlib.util
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from dreamq.chemprop_model import ChempropClassifier


def _inputs():
    return SimpleNamespace(
        structure_ids=np.array([f"mol-{i}" for i in range(9)]),
        canonical_smiles=np.array(["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCCC", "CCCO", "CCCN", "CCOC"]),
        signature="test-graph-features",
    )


def test_chemprop_missing_dependency_has_install_instructions(monkeypatch):
    import dreamq.chemprop_model as module
    def unavailable():
        raise ImportError(module.INSTALL_MESSAGE)
    monkeypatch.setattr(module, "_chemprop_modules", unavailable)
    with pytest.raises(ImportError, match=r"\[graph\]"):
        ChempropClassifier().fit(_inputs(), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]))


@pytest.mark.parametrize("parameters", [{"epochs": 0}, {"seeds": []}, {"seeds": [0, 0]}, {"depth": -1}, {"dropout": 2}, {"unknown": 1}])
def test_chemprop_configuration_rejected_before_training(parameters):
    with pytest.raises(ValueError):
        ChempropClassifier(parameters)._settings()


def test_chemprop_labels_are_required_and_binary():
    with pytest.raises(ValueError, match="both classes"):
        ChempropClassifier().fit(_inputs(), np.zeros(9))


@pytest.mark.skipif(importlib.util.find_spec("chemprop") is None, reason="optional Chemprop dependency not installed")
def test_chemprop_actual_training_singleton_retention_and_portable_serialization(tmp_path):
    inputs = _inputs()
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    parameters = {"epochs": 1, "batch_size": 4, "hidden_dim": 16, "ffn_hidden_dim": 16,
                  "depth": 2, "seeds": [0, 1], "dropout": 0.0}
    first = ChempropClassifier(parameters, seed=7, threads=1).fit(inputs, target)
    scores = first.predict_score(inputs)
    assert scores.shape == (9,)  # last singleton must not be discarded
    assert np.isfinite(first.decision_function(inputs)).all()
    assert np.all((scores >= 0) & (scores <= 1))
    assert first.training_row_count_ == 9
    np.testing.assert_array_equal(first.training_structure_ids_, inputs.structure_ids)
    assert len(first.member_states_) == 2 and all(isinstance(s, bytes) for s in first.member_states_)
    path = tmp_path / "portable.joblib"
    joblib.dump(first, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(scores, loaded.predict_score(inputs), rtol=0, atol=1e-7)
    repeated = ChempropClassifier(parameters, seed=7, threads=1).fit(inputs, target)
    np.testing.assert_allclose(scores, repeated.predict_score(inputs), rtol=0, atol=1e-7)
    inputs.signature = "different-schema"
    with pytest.raises(ValueError, match="configurations differ"):
        loaded.predict_score(inputs)


@pytest.mark.skipif(importlib.util.find_spec("chemprop") is None, reason="optional Chemprop dependency not installed")
def test_graph_inner_selection_and_portable_refit_via_experiment_runner(config, tmp_path):
    from dreamq.cross_validation import make_splits
    from dreamq.experiments import fit_models
    from dreamq.representations import MolecularInputs
    source = _inputs()
    inputs = MolecularInputs(source.structure_ids, source.canonical_smiles, {},
                             np.empty((9, 0)), source.signature)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    splits, _ = make_splits(y, inputs.structure_ids, folds=2, seed=7)
    config["zoo"] = {"grids": {"M15": {"epochs": [1], "batch_size": [4],
                    "hidden_dim": [16], "ffn_hidden_dim": [16], "depth": [2],
                    "dropout": [0.0], "seeds": [[0]]}}}
    models, oof, details = fit_models(inputs, y, splits, config, seed=7, names=["M15"])
    assert set(models) == set(oof) == set(details) == {"M15"}
    assert oof["M15"].shape == y.shape and np.isfinite(oof["M15"]).all()
    assert details["M15"]["selected"]["epochs"] == 1
    assert len(details["M15"]["search"]) == 1
    np.testing.assert_array_equal(models["M15"].training_structure_ids_, inputs.structure_ids)
    path = tmp_path / "runner_models.joblib"
    joblib.dump(models, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(models["M15"].predict_score(inputs), restored["M15"].predict_score(inputs), atol=1e-7)
