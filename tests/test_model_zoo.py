"""Model-family comparisons, train-only transforms, and inner-selection boundaries."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import sparse

from dreamq.cross_validation import make_splits
from dreamq.model_zoo import (MODEL_SPECS, ZooClassifier, parameter_grid,
                              require_model_dependencies, select_model, tanimoto_kernel)
from dreamq.representations import featurize_inputs


@pytest.fixture
def zoo_inputs(config, molecules, features):
    inputs = featurize_inputs(molecules, config["features"], config["cache_dir"], base=features)
    inputs.embeddings = np.random.default_rng(19).normal(size=(len(molecules), 128)).astype(np.float32)
    return inputs


@pytest.mark.parametrize("model_id", ["M0", "M1", "M2", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M13", "M14", "similarity"])
def test_independent_models_fit_and_serialize(model_id, zoo_inputs, molecules, tmp_path):
    kind = MODEL_SPECS[model_id]["estimator"]
    parameters = {"n_estimators": 12, "max_depth": 3} if kind in {"random_forest", "extra_trees"} else {}
    model = ZooClassifier(model_id, parameters, seed=31, threads=1).fit(zoo_inputs, molecules.label.to_numpy())
    scores = model.predict_score(zoo_inputs)
    assert scores.shape == (len(molecules),)
    assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    assert model.training_structure_ids_.tolist() == zoo_inputs.structure_ids.tolist()
    path = tmp_path / f"{model_id}.joblib"
    model.save(path)
    restored = ZooClassifier.load(path)
    np.testing.assert_array_equal(scores, restored.predict_score(zoo_inputs))


@pytest.mark.parametrize("model_id", ["M11", "M12"])
def test_optional_boosters_when_installed(model_id, zoo_inputs, molecules):
    package = MODEL_SPECS[model_id]["estimator"]
    if importlib.util.find_spec(package) is None:
        with pytest.raises(ImportError, match=package):
            ZooClassifier(model_id).fit(zoo_inputs, molecules.label.to_numpy())
    else:
        model = ZooClassifier(model_id, {"n_estimators": 8, "max_depth": 2}, threads=1)
        model.fit(zoo_inputs, molecules.label.to_numpy())
        assert np.isfinite(model.predict_score(zoo_inputs)).all()


def test_missing_optional_dependency_is_explicit(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "xgboost" else original(name))
    with pytest.raises(ImportError, match="no substitute"):
        require_model_dependencies("M11")


@pytest.mark.parametrize("model_id", ["M1", "M2", "M8"])
def test_training_only_continuous_preprocessing(model_id, zoo_inputs, molecules):
    train = np.arange(20)
    validation = zoo_inputs.take(np.arange(20, 32))
    model = ZooClassifier(model_id).fit(zoo_inputs.take(train), molecules.label.to_numpy()[train])
    mean = model.descriptor_transformer_.named_steps["scale"].mean_.copy()
    expected = np.nanmean(zoo_inputs.descriptors[train], axis=0)
    np.testing.assert_allclose(mean, expected, rtol=1e-6)
    validation.descriptors[:] = 1e9
    if model_id == "M2":
        embed_mean = model.embedding_scaler_.mean_.copy()
        validation.embeddings[:] = 1e9
    assert np.isfinite(model.predict_score(validation)).all()
    np.testing.assert_array_equal(mean, model.descriptor_transformer_.named_steps["scale"].mean_)
    if model_id == "M2":
        np.testing.assert_array_equal(embed_mean, model.embedding_scaler_.mean_)


def test_counts_and_bits_not_standardized(zoo_inputs, molecules):
    model = ZooClassifier("M4").fit(zoo_inputs, molecules.label.to_numpy())
    np.testing.assert_array_equal(model._matrix(zoo_inputs).toarray(), zoo_inputs.matrices["counts"].toarray())
    model = ZooClassifier("M0").fit(zoo_inputs, molecules.label.to_numpy())
    np.testing.assert_array_equal(model._matrix(zoo_inputs).toarray(), zoo_inputs.matrices["ecfp"].toarray())


def test_tree_predictions_reproducible(zoo_inputs, molecules):
    parameters = {"n_estimators": 15, "max_depth": 3}
    first = ZooClassifier("M10", parameters, 2035, 1).fit(zoo_inputs, molecules.label.to_numpy())
    second = ZooClassifier("M10", parameters, 2035, 1).fit(zoo_inputs, molecules.label.to_numpy())
    np.testing.assert_array_equal(first.predict_score(zoo_inputs), second.predict_score(zoo_inputs))


def test_tanimoto_values_empty_bitsets_and_guard(zoo_inputs, molecules):
    bits = sparse.csr_matrix([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    result = tanimoto_kernel(bits, bits, batch_size=1)
    np.testing.assert_allclose(np.diag(result), 1)
    assert result[0, 1] == pytest.approx(1 / 3)
    assert result[0, 2] == 0
    with pytest.raises(ValueError, match="max_train_rows"):
        ZooClassifier("M14", {"max_train_rows": 2}).fit(zoo_inputs, molecules.label.to_numpy())


def test_inner_selection_and_grid_override(config, zoo_inputs, molecules):
    y = molecules.label.to_numpy()
    splits, _ = make_splits(y, zoo_inputs.structure_ids, folds=2)
    config["zoo"] = {"grids": {"M4": [{"C": [0.01, 0.1], "positive_weight": 1}]}}
    model, oof, details = select_model("M4", zoo_inputs, y, splits, config, 2035)
    assert len(details["search"]) == 2
    assert model.parameters == details["selected"]
    assert np.isfinite(oof).all()
    assert len(parameter_grid("M4", config)) == 2
    with pytest.raises(ValueError, match="exactly once"):
        select_model("M4", zoo_inputs, y, splits[:1], config, 2035)
    overlapping = [(train, np.concatenate([valid, train[:1]])) for train, valid in splits]
    with pytest.raises(ValueError, match="leakage"):
        select_model("M4", zoo_inputs, y, overlapping, config, 2035)


def test_inference_signature_and_embedding_requirements(zoo_inputs, molecules):
    model = ZooClassifier("M0").fit(zoo_inputs, molecules.label.to_numpy())
    changed = zoo_inputs.take([0, 1])
    changed.signature = "other-feature-config"
    with pytest.raises(ValueError, match="signatures differ"):
        model.predict_score(changed)
    zoo_inputs.embeddings = None
    with pytest.raises(ValueError, match="frozen molecular embeddings"):
        ZooClassifier("M2").fit(zoo_inputs, molecules.label.to_numpy())
